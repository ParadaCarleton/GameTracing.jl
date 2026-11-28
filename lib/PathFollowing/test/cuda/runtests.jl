# Group storage on CUDA: each parity case traces step for step as a Vector start does on the host, the traces
# allocate nothing on the device, and `TanhChain` timings give µs per correction. Run with `--project=test/cuda`.
using PathFollowing
using CommonSolve: init
using CUDA: CUDA, CUDABackend, CuArray
using ErrorTypes: is_error, unwrap
using KernelAbstractions: KernelAbstractions, @groupsize, @index, @kernel, @localmem
using LinearSolve: LUFactorization
using Printf: @printf
using StaticArrays: SMatrix, SVector
using Test: @test, @testset

include("../common.jl")

const FLOATS = 2048

# One trace per workgroup, every lane of the group running it on the group's shared memory.
# KernelAbstractions hoists `@localmem` above the body, so the element type comes from the `where` clause.
@kernel function group_trace_kernel!(points::AbstractArray{E}, refusals, starts, directions, system, settings, first_step, bound, limit) where {E}
    floats = @localmem E (FLOATS,)
    integers = @localmem Int (16,)
    cursor = @localmem Int (2,)
    memory = GroupMemory(floats, integers, cursor, @index(Local, Linear), prod(@groupsize()), GroupBarrier())
    trace = @index(Group, Linear)
    record_group_trace!(
        view(points, :, :, trace), view(refusals, :, trace), memory, system, view(starts, :, trace), view(directions, :, trace),
        settings, first_step, bound, limit,
    )
end

# Each game's corrector iterations and accepted steps over `limit` steps of `TanhChain`.
@kernel function chain_kernel!(iterations, steps, starts::AbstractMatrix{E}, directions, settings, first_step, limit) where {E}
    floats = @localmem E (FLOATS,)
    integers = @localmem Int (16,)
    cursor = @localmem Int (2,)
    memory = GroupMemory(floats, integers, cursor, @index(Local, Linear), prod(@groupsize()), GroupBarrier())
    game = @index(Group, Linear)
    chain_trace!(iterations, steps, memory, game, view(starts, :, game), view(directions, :, game), settings, first_step, limit)
end

"""
`TanhChain` traced for `limit` steps from `start` in group `memory`; lane 1 writes its corrector iterations and its
accepted steps, or -1 steps when the trace does not start. A failed step ends the trace short of `limit`.
"""
function chain_trace!(
        iterations::AbstractVector{Int}, steps::AbstractVector{Int}, memory::GroupMemory, game::Integer, start::AbstractVector{E},
        direction::AbstractVector{E}, settings::PseudoArclength, first_step::E, limit::Integer,
    )::Nothing where {E <: Real}
    reset_memory!(memory)
    started = init(CurveProblem(TanhChain(), group_vector(memory, start), group_vector(memory, direction)), settings; first_step)
    total, taken = 0, -1
    if !is_error(started)
        taken = 0
        tracer = unwrap(started)
        for _ in 1:limit
            tracer, outcome = step!!(tracer)
            is_error(outcome) && break
            total, taken = total + unwrap(outcome).iterations, taken + 1
        end
    end
    if memory.lane == 1
        iterations[game], steps[game] = total, taken
    end
    return nothing
end

"""`algorithm(E; overrides...)` with static unit weights in `N` coordinates and `GroupLU`: isbits, for a kernel."""
function group_algorithm(::Type{E}, N::Integer, overrides::NamedTuple)::PseudoArclength where {E <: Real}
    return algorithm(E; merge(overrides, (; metric = FixedWeights(SVector{N, E}(ntuple(_ -> one(E), N))), linear_solver = GroupLU()))...)
end

"""The device code of `launch()`, as LLVM IR of every kernel it compiles."""
function device_ir(launch::Function)::String
    buffer = IOBuffer()
    CUDA.@device_code_llvm io = buffer dump_module = true launch()
    return String(take!(buffer))
end

"""Whether `ir` calls an allocator: the GC pool, `malloc`, or CUDA's device heap."""
allocates(ir::AbstractString)::Bool = any(name -> occursin(name, ir), ("gpu_gc_pool_alloc", "gpu_malloc", "@malloc"))

# The configuration the README tells users to launch on.
backend = CUDABackend(; always_inline = true)
println(CUDA.name(CUDA.device()), ", CUDA ", CUDA.runtime_version())

"""The CUDA parity cases: `parity_cases` and the 16-coordinate `wide_chain_case`."""
function cases(::Type{E})::Tuple where {E <: Real}
    return (parity_cases(E)..., wide_chain_case(E))
end

# The case's system and settings types specialize the kernel; the lane count is a launch value, so it shares the compile.
configs = [(E, case) for E in (Float32, Float64) for case in eachindex(cases(E))]
for (E, case) in configs[chosen_configs(ARGS, length(configs))]
    tolerance = sqrt(eps(E))
    system, start, direction, overrides, bound, limit, _ = cases(E)[case]
    @testset "CUDA $E, parity case $case: each lane count traces step for step as a Vector start, with no device allocation or recursion" begin
        N = length(start)
        settings = group_algorithm(E, N, overrides)
        host_points, host_refusals = recorded_trace(system, start, direction, algorithm(E; overrides...), E(1) / 4, bound, limit)
        taken = host_refusals .!= -2
        for lanes in (32, 256)
            points, refusals = CuArray(fill(E(NaN), N, limit, 1)), CuArray(fill(-2, limit, 1))
            launch() = group_trace_kernel!(backend)(points, refusals, CuArray(start[:, :]), CuArray(direction[:, :]), system, settings, E(1) / 4, bound, limit; ndrange = lanes, workgroupsize = lanes)
            # GPUCompiler compiles anew, and caches nothing, under an IR hook: so only the first lane count reads the IR.
            if lanes == 32
                seconds = @elapsed ir = device_ir(launch)
                @test occursin("define", ir) && !allocates(ir)
                @test call_cycle(ir) === nothing
            else
                seconds = @elapsed launch()
            end
            @printf("first launch, %s parity case %d, %d lanes: %.2f s\n", E, case, lanes, seconds)
            KernelAbstractions.synchronize(backend)
            @test Array(refusals)[:, 1] == host_refusals
            @test Array(points)[:, taken, 1] ≈ host_points[:, taken] atol = 10 * tolerance nans = true
        end
    end
end

# Float32 only: both GPUs run Float64 at 1/64 rate.
timing_chosen = isempty(ARGS) || ARGS == ["timing"]
timing_chosen && println("coordinates, lanes, games, steps per game, corrections per game, ms, µs per correction of one game, ns per correction over all games")
for N in (4, 10, 16)
    timing_chosen || break
    E, limit = Float32, 50
    settings = group_algorithm(E, N, (;))
    # One lane on the host first: a trace past `FLOATS` throws here, where a GPU reports only a bad address.
    memory, host_iterations, host_steps = GroupMemory(zeros(E, FLOATS), zeros(Int, 16), zeros(Int, 2), 1, 1, LoneLane()), [0], [0]
    host_trace() = chain_trace!(host_iterations, host_steps, memory, 1, chain_start(E, N, 2π)..., settings, E(1) / 20, limit)
    host_trace()
    host_seconds = minimum(_ -> @elapsed(host_trace()), 1:5)
    host_steps == [limit] || error(lazy"the host TanhChain trace in $N coordinates took $(only(host_steps)) of $limit steps")
    @printf(
        "host, %d coordinates, one lane: %.1f corrections, %.2f µs per correction, %d of %d floats\n", N,
        only(host_iterations), 1.0e6 * host_seconds / only(host_iterations), memory.cursor[1], FLOATS
    )
    for lanes in (32, 256), games in (1, 1024)
        pairs = [chain_start(E, N, 2π * game / games) for game in 1:games]
        starts, directions = CuArray(reduce(hcat, first.(pairs))), CuArray(reduce(hcat, last.(pairs)))
        iterations, steps = CUDA.zeros(Int, games), CUDA.zeros(Int, games)
        launch() = chain_kernel!(backend)(iterations, steps, starts, directions, settings, E(1) / 20, limit; ndrange = lanes * games, workgroupsize = lanes)
        if lanes == 32 && games == 1
            first_seconds = @elapsed ir = device_ir(launch)
            cycle = call_cycle(ir)
            occursin("define", ir) && !allocates(ir) && isnothing(cycle) || error(lazy"the chain kernel in $N coordinates allocates on the device or calls the cycle $cycle")
        else
            first_seconds = @elapsed launch()
        end
        KernelAbstractions.synchronize(backend)
        seconds = minimum(1:5) do _
            CUDA.@elapsed (launch(); KernelAbstractions.synchronize(backend))
        end
        device_steps = Array(steps)
        # A failed trace corrects less, so its timing would read fast.
        all(==(limit), device_steps) || error(lazy"TanhChain traces failed or stopped short of $limit steps in games $(findall(!=(limit), device_steps))")
        corrections = sum(Array(iterations))
        @printf(
            "%d, %d, %d, %.1f, %.1f, %.3f, %.2f, %.1f\n", N, lanes, games, sum(device_steps) / games, corrections / games,
            1.0e3 * seconds, 1.0e6 * seconds / (corrections / games), 1.0e9 * seconds / corrections
        )
        @printf("first launch, %d coordinates, %d lanes, %d games: %.2f s\n", N, lanes, games, first_seconds)
    end
end
