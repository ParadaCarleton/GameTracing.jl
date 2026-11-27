# Test systems and settings for the host suite (`runtests.jl`) and the CUDA script (`cuda/runtests.jl`).
# The includer loads PathFollowing, LinearSolve, StaticArrays and CommonSolve.
using LazyBroadcast: lazy_broadcast

"""
A circle of `radius` about the origin in `(x, y)`: `H = x² + y² - radius²`. Broadcasts and reductions only, so it
also runs on GPU arrays, which refuse scalar indexing, and on static arrays.
"""
struct Circle{E <: Real}
    radius::E
end

function (circle::Circle{E})(residual::R, jacobian::J, point::AbstractVector{E})::Tuple{R, J} where {E <: Real, R <: AbstractVector{E}, J <: AbstractMatrix{E}}
    # With `init`: from 16 entries Base's `sum` calls its recursive `mapreduce_impl`, which a GPU stack cannot bound.
    return (into!!(residual, sum(abs2, point; init = zero(E)) - circle.radius^2), into!!(jacobian, lazy_broadcast.(2 .* transpose(point))))
end

"""`residual` and `jacobian`, one row of two, set to `entry` and `row` (`into!!`): a test system's return on any storage."""
function one_row!!(residual::R, jacobian::J, entry::E, row::NTuple{2, E})::Tuple{R, J} where {E <: Real, R <: AbstractVector{E}, J <: AbstractMatrix{E}}
    return (into!!(residual, SVector(entry)), into!!(jacobian, SMatrix{1, 2, E}(row...)))
end

"""The line `x = y`: `H = x - y`."""
struct Line end

function (::Line)(residual::R, jacobian::J, point::AbstractVector{E})::Tuple{R, J} where {E <: Real, R <: AbstractVector{E}, J <: AbstractMatrix{E}}
    return one_row!!(residual, jacobian, point[1] - point[2], (one(E), -one(E)))
end

"""The curve `λ = u - u³` in `(u, λ)`: λ turns at a minimum, `u = -1/√3`, then at a maximum, `u = 1/√3`."""
struct Wiggle end

function (::Wiggle)(residual::R, jacobian::J, point::AbstractVector{E})::Tuple{R, J} where {E <: Real, R <: AbstractVector{E}, J <: AbstractMatrix{E}}
    return one_row!!(residual, jacobian, point[2] - point[1] + point[1]^3, (3 * point[1]^2 - 1, one(E)))
end

"""The fold `H = u² - λ` in `(u, λ)`: λ turns back at `u = 0`."""
struct Fold end

function (::Fold)(residual::R, jacobian::J, point::AbstractVector{E})::Tuple{R, J} where {E <: Real, R <: AbstractVector{E}, J <: AbstractMatrix{E}}
    return one_row!!(residual, jacobian, point[1]^2 - point[2], (2 * point[1], -one(E)))
end

"""
A pitchfork `H = u³ - λ u - tilt` in `(u, λ)`. Untilted, the branch `u = 0` crosses a simple bifurcation point at the
origin; tilted, its upper branch passes close to the lower ones near `λ = 0`.
"""
struct Pitchfork{E <: Real}
    tilt::E
end

function (pitchfork::Pitchfork{E})(residual::R, jacobian::J, point::AbstractVector{E})::Tuple{R, J} where {E <: Real, R <: AbstractVector{E}, J <: AbstractMatrix{E}}
    (u, lambda) = (point[1], point[2])
    return one_row!!(residual, jacobian, u^3 - lambda * u - pitchfork.tilt, (3 * u^2 - lambda, -u))
end

"""`H = u - λ` up to `λ = edge`; past it a NaN residual, or a zero Jacobian when `singular`."""
struct Edge{E <: Real}
    edge::E
    singular::Bool
end

function (edge::Edge{E})(residual::R, jacobian::J, point::AbstractVector{E})::Tuple{R, J} where {E <: Real, R <: AbstractVector{E}, J <: AbstractMatrix{E}}
    past = point[2] > edge.edge
    row = past && edge.singular ? (zero(E), zero(E)) : (one(E), -one(E))
    return one_row!!(residual, jacobian, past && !edge.singular ? E(NaN32) : point[1] - point[2], row)
end

"""`H = u - λ`, whose `bad_call`-th evaluation is NaN, or has a zero Jacobian when `singular`."""
mutable struct Flaky
    calls::Int
    const bad_call::Int
    const singular::Bool
end

function (flaky::Flaky)(residual::R, jacobian::J, point::AbstractVector{E})::Tuple{R, J} where {E <: Real, R <: AbstractVector{E}, J <: AbstractMatrix{E}}
    flaky.calls += 1
    bad = flaky.calls == flaky.bad_call
    row = bad && flaky.singular ? (zero(E), zero(E)) : (one(E), -one(E))
    return one_row!!(residual, jacobian, bad && !flaky.singular ? E(NaN32) : point[1] - point[2], row)
end

"""A correction's length as the largest move of any coordinate, unbounded."""
struct LargestMove end

function (::LargestMove)(from::AbstractVector{E}, to::AbstractVector{E})::E where {E <: Real}
    return mapreduce((later, earlier) -> abs(later - earlier), max, to, from)
end
function PathFollowing.correction_bound(::LargestMove, ::Type{E})::E where {E <: Real}
    return E(Inf32)
end

"""`LargestMove` capped at one, as a distance between probabilities is."""
struct CappedMove end

function (::CappedMove)(from::AbstractVector{E}, to::AbstractVector{E})::E where {E <: Real}
    return min(LargestMove()(from, to), one(E))
end
function PathFollowing.correction_bound(::CappedMove, ::Type{E})::E where {E <: Real}
    return one(E)
end

"""
The parabola `x = y²/2` and the line `x = -1/5`: `H = (x - y²/2)(x + 1/5)`. Up from the origin, a step of 1 corrects
onto the line at `(-1/5, 1)`. Its half steps keep the parabola and reach `(1/2, 1)`.
"""
struct ParabolaAndLine end

function (::ParabolaAndLine)(residual::R, jacobian::J, point::AbstractVector{E})::Tuple{R, J} where {E <: Real, R <: AbstractVector{E}, J <: AbstractMatrix{E}}
    parabola, line = point[1] - point[2]^2 / 2, point[1] + one(E) / 5
    return one_row!!(residual, jacobian, parabola * line, (parabola + line, -point[2] * line))
end

"""The line `x = 0`, its residual NaN for `y` in `(1/4, 3/4)`: a step of 1 up from the origin sees only `y = 1`."""
struct BandedLine end

function (::BandedLine)(residual::R, jacobian::J, point::AbstractVector{E})::Tuple{R, J} where {E <: Real, R <: AbstractVector{E}, J <: AbstractMatrix{E}}
    return one_row!!(residual, jacobian, one(E) / 4 < point[2] < 3 * one(E) / 4 ? E(NaN32) : point[1], (one(E), zero(E)))
end

"""`LargestMove` that throws a `DomainError` on a point in `BandedLine`'s band, where that line has no residual."""
struct BandAvoidingMove end

function (::BandAvoidingMove)(from::AbstractVector{E}, to::AbstractVector{E})::E where {E <: Real}
    in_band = point -> one(E) / 4 < point[2] < 3 * one(E) / 4
    (in_band(from) || in_band(to)) && throw(DomainError((from, to), "a point in BandedLine's band"))
    return LargestMove()(from, to)
end
function PathFollowing.correction_bound(::BandAvoidingMove, ::Type{E})::E where {E <: Real}
    return E(Inf32)
end

"""A `converged` that never holds."""
never(::AbstractVector, ::AbstractVector, ::AbstractVector)::Bool = false

"""Converged when the correction moved no coordinate more than `tolerance`."""
struct Settled{E <: Real}
    tolerance::E
end

function (settled::Settled{E})(previous::AbstractVector{E}, iterate::AbstractVector{E}, ::AbstractVector{E})::Bool where {E <: Real}
    # A lazy broadcast: one reduction on a GPU array, and no allocation on the CPU.
    moves = Broadcast.instantiate(Broadcast.broadcasted((from, to) -> abs(to - from), previous, iterate))
    return mapreduce(identity, max, moves; init = zero(E)) <= settled.tolerance
end

"""A `PseudoArclength` in `E` with Euclidean metrics and the given overrides."""
function algorithm(::Type{E}; tolerance::Real = sqrt(eps(E)), overrides...)::PseudoArclength where {E <: Real}
    defaults = (;
        corrector = Newton(; max_iterations = 8, converged = Settled(E(tolerance)), contraction_limit = one(E) / 2),
        step_control = StepControl(;
            target_contraction = one(E) / 4, target_correction = one(E) / 10, target_turn = one(E) / 10,
            max_step_growth = E(2), max_step = one(E) / 2,
        ),
        metric = FixedWeights(ones(E, 2)),
        orientation_check = DeterminantSign(),
        split_check = nothing,
        linear_solver = LUFactorization(),
        max_refusals = 60,
    )
    return PseudoArclength(; merge(defaults, overrides)...)
end
"""
A chain in `n` coordinates: the unit circle in `(x₁, x₂)`, then `xₖ₊₁ = tanh(xₖ)` for `k` in `2:n-1`. Broadcasts over
indices only, so it runs on any storage. `chain_start` gives a point on it.
"""
struct TanhChain end

function (::TanhChain)(residual::R, jacobian::J, point::AbstractVector{E})::Tuple{R, J} where {E <: Real, R <: AbstractVector{E}, J <: AbstractMatrix{E}}
    residual = into!!(residual, lazy_broadcast.((row -> chain_residual(point, row)).(eachindex(residual))))
    return residual, into!!(jacobian, lazy_broadcast.((index -> chain_slope(point, index[1], index[2])).(CartesianIndices(jacobian))))
end

"""Row `row` of `TanhChain`'s residual at `point`."""
function chain_residual(point::AbstractVector{E}, row::Integer)::E where {E <: Real}
    return row == 1 ? point[1]^2 + point[2]^2 - 1 : point[row + 1] - tanh(point[row])
end

"""Entry `(row, column)` of `TanhChain`'s Jacobian at `point`."""
function chain_slope(point::AbstractVector{E}, row::Integer, column::Integer)::E where {E <: Real}
    row == 1 && return column <= 2 ? 2 * point[column] : zero(E)
    column == row + 1 && return one(E)
    return column == row ? tanh(point[row])^2 - 1 : zero(E)
end

"""`TanhChain`'s point in `n` coordinates at angle `angle` on its circle, and a direction along the chain from it."""
function chain_start(::Type{E}, n::Integer, angle::Real)::Tuple{Vector{E}, Vector{E}} where {E <: Real}
    point, direction = zeros(E, n), zeros(E, n)
    point[1], point[2], direction[1], direction[2] = cos(angle), sin(angle), -sin(angle), cos(angle)
    for coordinate in 3:n
        point[coordinate] = tanh(point[coordinate - 1])
        direction[coordinate] = (1 - point[coordinate]^2) * direction[coordinate - 1]
    end
    return point, direction
end

"""`entries` copied into a new vector of group `memory`."""
function group_vector(memory::M, entries::AbstractVector{E})::GroupArray{E, 1, M} where {E <: Real, M <: GroupMemory}
    return copyto!(group_array(memory, E, (length(entries),)), entries)
end

"""
Trace `system` from `start` toward `direction` for at most `limit` steps toward `bound`, as one lane of `memory`'s
group. Lane 1 writes each step's point into column `step` of `points` and its refusals into `refusals[step]`: -1
for the failed step that ends the trace, and -3 in `refusals[1]` for a trace that does not start. Entries past the
trace's end keep their values.
"""
function record_group_trace!(
        points::AbstractMatrix{E}, refusals::AbstractVector{Int}, memory::GroupMemory, system::Any, start::AbstractVector{E},
        direction::AbstractVector{E}, settings::PseudoArclength, first_step::E, bound::Union{Nothing, Bound}, limit::Integer,
    )::Nothing where {E <: Real}
    reset_memory!(memory)
    started = init(CurveProblem(system, group_vector(memory, start), group_vector(memory, direction)), settings; first_step)
    if is_error(started)
        memory.lane == 1 && (refusals[1] = -3)
        return nothing
    end
    tracer = unwrap(started)
    for step in 1:limit
        tracer, outcome = step!!(memory, tracer, bound)
        point = group_view(memory, tracer).point
        if memory.lane == 1
            refusals[step] = is_error(outcome) ? -1 : unwrap(outcome).refusals
            for coordinate in eachindex(point)
                points[coordinate, step] = point[coordinate]
            end
        end
        # Every lane holds the same outcome and point, so every lane leaves together.
        is_error(outcome) && return nothing
        isnothing(bound) || point[bound.coordinate] != bound.value || return nothing
    end
    return nothing
end

"""
`record_group_trace!`'s record of a trace with storage `start`'s, from a new `Vector` pair: the points and the
refusals, -2 for a step the trace never took.
"""
function recorded_trace(
        system::Any, start::Vector{E}, direction::Vector{E}, settings::PseudoArclength, first_step::E, bound::Union{Nothing, Bound}, limit::Integer,
    )::Tuple{Matrix{E}, Vector{Int}} where {E <: Real}
    points, refusals = fill(E(NaN), length(start), limit), fill(-2, limit)
    tracer = unwrap(init(CurveProblem(system, start, direction), settings; first_step))
    for step in 1:limit
        tracer, outcome = step!!(tracer, bound)
        refusals[step] = is_error(outcome) ? -1 : unwrap(outcome).refusals
        points[:, step] = tracer.point
        is_error(outcome) && break
        isnothing(bound) || tracer.point[bound.coordinate] != bound.value || break
    end
    return points, refusals
end

"""
Traces for the storage parity tests: a split check, a landing, orientation flips at a bifurcation point, zero pivots
past an edge, turns of a wiggle, and a chain in 4 coordinates. Each holds its system, start, direction, settings
overrides, bound, step limit, and a check of its refusals and failure that it took its path.
"""
function parity_cases(::Type{E})::Tuple where {E <: Real}
    check = HalfStepCheck{E}(; trigger = 0, tolerance = one(E) / 100)
    chain, chain_direction = chain_start(E, 4, 3 // 10)
    return (
        (Circle(one(E)), E[1, 0], E[0, 1], (; split_check = check), nothing, 12, (_, failure) -> isnothing(failure)),
        (Circle(one(E)), E[1, 0], E[0, 1], (; split_check = check), Bound(2, E(3) / 4), 20, (_, failure) -> isnothing(failure)),
        (Pitchfork(zero(E)), E[0, -1], E[0, 1], (;), nothing, 12, (refusals, _) -> refusals > 0),
        (Edge(one(E), true), E[0, 0], E[1, 1], (;), nothing, 200, (_, failure) -> failure.last_refusal == SingularSystem),
        (Wiggle(), [-E(3) / 5, -E(48) / 125], E[1, 0], (;), nothing, 16, (_, failure) -> isnothing(failure)),
        (TanhChain(), chain, chain_direction, (; metric = FixedWeights(ones(E, 4))), nothing, 16, (_, failure) -> isnothing(failure)),
    )
end

"""
A parity case (`parity_cases`) of `TanhChain` in 16 coordinates, a 15-filer stay solve's size. `StaticLU` takes at
most 14, so only mutable and group starts trace it.
"""
function wide_chain_case(::Type{E})::Tuple where {E <: Real}
    chain, chain_direction = chain_start(E, 16, 3 // 10)
    return (TanhChain(), chain, chain_direction, (; metric = FixedWeights(ones(E, 16))), nothing, 50, (_, failure) -> isnothing(failure))
end

"""A called function's name in LLVM IR, bare or quoted, as in `call void @name(`."""
const CALLED_NAME = r"@(\"[^\"]+\"|[^\s(]+)\("

"""
The first cycle of calls among the functions that the LLVM IR `ir` defines, as names from a function back to itself,
or `nothing`. Recursion leaves a kernel's stack size without a static bound.
"""
function call_cycle(ir::AbstractString)::Union{Nothing, Vector{String}}
    callees = Dict{String, Vector{String}}()
    for definition in split(ir, r"^define "m)[2:end]
        header = match(CALLED_NAME, definition)
        isnothing(header) && continue
        body = SubString(definition, header.offset + ncodeunits(header.match))
        callees[header.captures[1]] = [called.captures[1] for called in eachmatch(CALLED_NAME, body)]
    end
    # Depth first: a name met again on the current call path closes a cycle.
    finished, path = Set{String}(), String[]
    function cycle_from(name::String)::Union{Nothing, Vector{String}}
        on_path = findfirst(==(name), path)
        isnothing(on_path) || return [path[on_path:end]; name]
        name in finished && return nothing
        push!(path, name)
        for callee in get(callees, name, String[])
            found = cycle_from(callee)
            isnothing(found) || return found
        end
        pop!(path)
        push!(finished, name)
        return nothing
    end
    for name in keys(callees)
        found = cycle_from(name)
        isnothing(found) || return found
    end
    return nothing
end

"""
The indices of `configs` CUDA parity configurations that `arguments` choose: `parity SHARD SHARDS` takes every
`SHARDS`-th from `SHARD`, so processes split the compiles; `timing` takes none; no arguments take all.
"""
function chosen_configs(arguments::Vector{String}, configs::Integer)::StepRange{Int, Int}
    isempty(arguments) && return 1:1:configs
    arguments == ["timing"] && return 1:1:0
    length(arguments) == 3 && arguments[1] == "parity" || error(lazy"run `parity SHARD SHARDS`, `timing` or no part; got $arguments")
    shard, shards = parse(Int, arguments[2]), parse(Int, arguments[3])
    1 <= shard <= min(shards, configs) || error(lazy"shard $shard lies outside 1:$(min(shards, configs)), so it holds no configuration")
    return shard:shards:configs
end
