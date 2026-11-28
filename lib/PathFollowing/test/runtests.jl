using PathFollowing
using PathFollowing: AcceptAll, factored_sign, metric_angle
using Accessors: @reset
using CommonSolve: init
using ErrorTypes: is_error, unwrap, unwrap_error
using GPUArraysCore: allowscalar
using JLArrays: JLArray, JLBackend
using KernelAbstractions: KernelAbstractions, @groupsize, @index, @kernel, @localmem
using LinearAlgebra: LinearAlgebra, det
using LinearSolve: LUFactorization, QRFactorization
using Moshi.Data: isa_variant
using StaticArrays: SMatrix, SVector
using Supposition: Data, @check
using Test: @inferred, @test, @testset, @test_throws

# A GPU array that indexes one element at a time throws, so the JLArrays tests catch every scalar access.
allowscalar(false)

include("common.jl")

"""`tracer`, of group `memory` or `nothing`, after two steps toward `bound`, and the bytes the second, warm `step!!` allocates."""
function warm_bytes(memory::Union{Nothing, GroupMemory}, tracer::T, bound::Union{Nothing, Bound})::Tuple{T, Int} where {T <: Tracer}
    tracer, _ = step!!(memory, tracer, bound)
    bytes = @allocated((tracer, _) = step!!(memory, tracer, bound))
    return tracer, bytes
end

function warm_bytes(tracer::T, bound::Union{Nothing, Bound})::Tuple{T, Int} where {T <: Tracer}
    return warm_bytes(nothing, tracer, bound)
end

"""`tracer` advanced by one step toward `bound` (`step!!`), and its `AcceptedStep`; a failed step throws."""
function accepted(tracer::T, bound::Union{Nothing, Bound} = nothing)::Tuple{T, AcceptedStep} where {T <: Tracer}
    tracer, outcome = step!!(tracer, bound)
    return tracer, unwrap(outcome)
end

"""A `Circle` whose `system!!` throws a `DomainError` past radius 2."""
struct BrittleCircle{E <: Real}
    radius::E
end

function (circle::BrittleCircle{E})(residual::R, jacobian::J, point::AbstractVector{E})::Tuple{R, J} where {E <: Real, R <: AbstractVector{E}, J <: AbstractMatrix{E}}
    circle.radius > 2 && throw(DomainError(circle.radius, "BrittleCircle past radius 2"))
    return Circle(circle.radius)(residual, jacobian, point)
end

"""The bytes a warm `@into!!` broadcast and a warm `@into!!` copy into `vector` allocate, after one of each to warm them."""
function into_bytes(vector::AbstractVector, source::AbstractVector)::Tuple{Int, Int}
    @into!! vector .= 2 .* vector .+ source
    @into!! vector .= source
    return @allocated(@into!!(vector .= 2 .* vector .+ source)), @allocated(@into!!(vector .= source))
end

"""The bytes a warm `land!!` of `tracer` to `bound` allocates, after one landing to warm it."""
function landing_bytes(tracer::Tracer, bound::Bound, max_steps::Integer)::Int
    land!!(tracer, bound, max_steps)
    return @allocated land!!(tracer, bound, max_steps)
end

"""The `StepFailure` of `tracer`'s next step toward `bound`; an accepted step throws."""
function failed_step(tracer::Tracer, bound::Union{Nothing, Bound} = nothing)::StepFailure.Type
    return unwrap_error(last(step!!(tracer, bound)))
end

"""`algorithm(E)` with a static metric and `StaticLU`: isbits, so `land_batch` takes it."""
function static_algorithm(::Type{E}; overrides...)::PseudoArclength where {E <: Real}
    return algorithm(E; metric = FixedWeights(SVector{2, E}(1, 1)), linear_solver = StaticLU(), overrides...)
end

"""A workgroup memory of one lane on the host, with room for `floats` floats of `E`: a lone lane needs no barrier."""
function host_memory(::Type{E}; floats::Integer = 512)::GroupMemory where {E <: Real}
    return GroupMemory(zeros(E, floats), zeros(Int, 1), 1, 1, LoneLane())
end

"""`γ_n = n u / (1 - n u)`, `u = eps(E) / 2`: the relative error bound of `n` rounded operations in `E` (Higham 2002, Lemma 3.1)."""
function rounding_factor(::Type{E}, steps::Integer)::BigFloat where {E <: Real}
    unit = big(eps(E)) / 2
    steps * unit < 1 || throw(DomainError(steps, "γ_n needs n eps($E) / 2 < 1"))
    return steps * unit / (1 - steps * unit)
end

"""
`‖rhs - matrix solution‖∞` over its bound, exact in BigFloat, for a pivoted elimination whose multipliers are at most 1
and whose upper factor `U` is `strict_upper`'s strict upper triangle with `pivots` on its diagonal.

Higham (2002, Theorems 9.3 and 9.4) bounds `|rhs - matrix x|` by `γ_3n |L||U||x|`, so by `γ_3n sum(|U||x|)` when `|L| <= 1`.
Gradual underflow adds an error `η <= nextfloat(0) / 2` to each product and quotient, never to a sum. Carried through the
elimination and both substitutions, it adds at most `2η (n² + sum|p| + n sum|x| + sum|p x|)`, `p` the pivots.
"""
function lu_residual_ratio(matrix::AbstractMatrix{E}, rhs::AbstractVector{E}, strict_upper::AbstractMatrix{E}, pivots::AbstractVector{E}, solution::AbstractVector{E})::BigFloat where {E <: Real}
    order = length(rhs)
    upper, pivot, x = big.(LinearAlgebra.triu(strict_upper, 1)) + LinearAlgebra.Diagonal(big.(pivots)), big.(pivots), big.(solution)
    residual = maximum(abs, big.(rhs) - big.(matrix) * x)
    # `2η` is the least subnormal.
    underflow = big(nextfloat(zero(E))) * (order^2 + sum(abs, pivot) + order * sum(abs, x) + sum(abs, pivot .* x))
    return residual / (rounding_factor(E, 3 * order) * sum(abs.(upper) * abs.(x)) + underflow)
end

# One trace per workgroup, every lane of the group running it on the group's `@localmem`.
# KernelAbstractions hoists `@localmem` above the body, so the element type comes from the `where` clause.
@kernel function group_trace_kernel!(points::AbstractArray{E}, refusals, starts, directions, system, settings, first_step, bound, limit, barrier) where {E}
    floats = @localmem E (512,)
    cursor = @localmem Int (1,)
    memory = GroupMemory(floats, cursor, @index(Local, Linear), prod(@groupsize()), barrier)
    trace = @index(Group, Linear)
    record_group_trace!(
        view(points, :, :, trace), view(refusals, :, trace), memory, system, view(starts, :, trace), view(directions, :, trace),
        settings, first_step, bound, limit,
    )
end

"""
A metric that counts its freezes in `freezes` and weighs each coordinate by `1 + |xᵢ|` at the point `x` it is frozen at:
a diagonal stand-in for a metric that depends on the point.
"""
struct CountingMetric <: PathFollowing.Metric
    freezes::Base.RefValue{Int}
end

"""`CountingMetric`'s state: its counter and its weights at the point it is frozen at, in the point's storage."""
struct CountingFrozen{E <: Real, V <: AbstractVector{E}}
    freezes::Base.RefValue{Int}
    weights::V
end

function PathFollowing.frozen_metric(metric::CountingMetric, point::V)::CountingFrozen{E, V} where {E <: Real, V <: AbstractVector{E}}
    return freeze_metric!!(CountingFrozen(metric.freezes, zero(point)), point)
end

function PathFollowing.freeze_metric!!(frozen::F, point::AbstractVector{E})::F where {E <: Real, F <: CountingFrozen{E}}
    frozen.freezes[] += 1
    return CountingFrozen(frozen.freezes, @into!!(frozen.weights .= 1 .+ abs.(point)))
end

function PathFollowing.metric_inner(frozen::CountingFrozen{E}, left::AbstractVector{E}, right::AbstractVector{E})::E where {E <: Real}
    return sum(frozen.weights .* left .* right)
end

function PathFollowing.metric_apply!!(out::V, frozen::CountingFrozen{E}, vector::AbstractVector{E})::V where {E <: Real, V <: AbstractVector{E}}
    return @into!! out .= frozen.weights .* vector
end

"""
The tracer and its `AcceptedStep` after a half step up the unit circle from `(1, 0)`, with no step-control targets, its
corrector settling at `settle`.
"""
function half_circle_step(settle::E)::Tuple{Tracer, AcceptedStep} where {E <: Real}
    control = StepControl(; target_contraction = nothing, target_correction = nothing, target_turn = nothing, max_step_growth = E(2), max_step = one(E))
    corrector = Newton(; max_iterations = 8, converged = Settled(settle), contraction_limit = one(E) / 2)
    return accepted(unwrap(init(CurveProblem(Circle(one(E)), E[1, 0], E[0, 1]), algorithm(E; corrector, step_control = control); first_step = one(E) / 2)))
end

"""Accepted points of `steps` steps of `problem` under `settings`, or the failure."""
function traced(problem::CurveProblem, settings::PseudoArclength, steps::Integer; first_step::Real)::Any
    started = init(problem, settings; first_step)
    is_error(started) && return started
    tracer = unwrap(started)
    points = [copy(tracer.point)]
    for _ in 1:steps
        tracer, outcome = step!!(tracer)
        is_error(outcome) && return outcome
        push!(points, copy(tracer.point))
    end
    return points
end

"""A tracer on `system` from `start` toward `direction`, under step control with no targets, and the given overrides."""
function untargeted(system::Any, start::Vector{E}, direction::Vector{E}; first_step::Real, overrides...)::Tracer where {E <: Real}
    control = StepControl(; target_contraction = nothing, target_correction = nothing, target_turn = nothing, max_step_growth = E(2), max_step = one(E))
    return unwrap(init(CurveProblem(system, start, direction), algorithm(E; step_control = control, overrides...); first_step))
end

"""
`tracer` advanced toward `bound` until its point lies on it, at most `limit` steps, and the accepted steps; a failed
step throws.
"""
function bounded_steps(tracer::T, bound::Bound{E}, limit::Integer)::Tuple{T, Vector{AcceptedStep{E}}} where {E <: Real, T <: Tracer{E}}
    steps = AcceptedStep{E}[]
    while tracer.point[bound.coordinate] != bound.value && length(steps) < limit
        tracer, step = accepted(tracer, bound)
        push!(steps, step)
    end
    return tracer, steps
end

"""Whether `Pkg.test`'s `test_args` choose `element_type`. They split the suite over processes; none choose all."""
chosen(element_type::Type{<:Real})::Bool = isempty(ARGS) || string(element_type) in ARGS
all(in(("Float32", "Float64", "BigFloat")), ARGS) || error(lazy"test_args name Float32, Float64 or BigFloat; got $ARGS")

@testset "PathFollowing $E" for E in filter(chosen, (Float32, Float64))
    tolerance = sqrt(eps(E))

    @testset "circle: every point on it, the angle rising around a full turn" begin
        points = traced(CurveProblem(Circle(E(2)), E[2, 0], E[0, 1]), algorithm(E), 80; first_step = E(1) / 10)
        @test all(point -> abs(hypot(point...) - 2) <= 10 * tolerance, points)
        angles = [atan(point[2], point[1]) for point in points]
        unwrapped = cumsum([zero(E); mod.(diff(angles) .+ E(π), 2 * E(π)) .- E(π)])
        @test all(>(0), diff(unwrapped))
        @test last(unwrapped) > 2 * E(π)
    end

    @testset "fold: the trace turns at u = 0 and keeps the branch" begin
        points = traced(CurveProblem(Fold(), E[-1, 1], E[1, -1]), algorithm(E), 60; first_step = E(1) / 10)
        us = first.(points)
        @test all(>(0), diff(us))
        @test last(us) > 1
        @test minimum(last.(points)) >= -tolerance
        @test all(point -> abs(point[1]^2 - point[2]) <= 10 * tolerance, points)
    end

    @testset "unfolded pitchfork: the upper branch stays up" begin
        points = traced(CurveProblem(Pitchfork(E(1) / 100), E[1, 1 - E(1) / 100], E[0, -1]), algorithm(E), 60; first_step = E(1) / 10)
        @test all(point -> point[1] > 0, points)
        @test last(points)[2] < -1
    end

    @testset "a step across a bifurcation point is refused: the trace closes in on it and fails unresolved" begin
        tracer = unwrap(init(CurveProblem(Pitchfork(zero(E)), E[0, -1], E[0, 1]), algorithm(E); first_step = E(3) / 10))
        lambdas = E[]
        failure = nothing
        for _ in 1:10_000
            tracer, outcome = step!!(tracer)
            if is_error(outcome)
                failure = unwrap_error(outcome)
                break
            end
            push!(lambdas, tracer.point[2])
        end
        @test isa_variant(failure, StepFailure.Unresolved)
        @test all(<(0), lambdas) && issorted(lambdas)
        @test tracer.orientation == 1 && tracer.point[1] == 0
    end

    @testset "the orientation sign is the sign of det([J; tᵀ]), from the solve's LU or a second factorization" begin
        for linear_solver in (LUFactorization(), QRFactorization()), (start, expected) in ((E[0, -1], 1), (E[0, 1], -1))
            # The branch u = 0 changes sign at the bifurcation point λ = 0, so starts on either side differ.
            settings = algorithm(E; linear_solver)
            tracer = unwrap(init(CurveProblem(Pitchfork(zero(E)), start, E[0, 1]), settings; first_step = E(3) / 10))
            @test tracer.orientation == expected == sign(det([tracer.work.jacobian; tracer.tangent']))
            for _ in 1:6
                tracer, _ = accepted(tracer)
                @test tracer.orientation == expected == sign(det([tracer.work.jacobian; tracer.tangent']))
            end
            # Only the fallback fills the matrix of the second factorization.
            @test all(iszero, tracer.linear.oriented) == (linear_solver isa LUFactorization)
        end
    end

    @testset "without an orientation check, a bifurcation crossing goes unremarked" begin
        settings = algorithm(E; orientation_check = nothing)
        tracer = unwrap(init(CurveProblem(Pitchfork(zero(E)), E[0, -1], E[0, 1]), settings; first_step = E(3) / 10))
        steps = AcceptedStep{E}[]
        for _ in 1:6
            tracer, step = accepted(tracer)
            push!(steps, step)
        end
        @test !any(step -> step.refusals > 0, steps)
        @test tracer.orientation == 0
        @test tracer.point[2] > 0
    end

    @testset "fold traced backward stays on its branch" begin
        points = traced(CurveProblem(Fold(), E[1, 1], E[-1, -1]), algorithm(E), 60; first_step = E(1) / 10)
        @test all(<(0), diff(first.(points)))
    end

    @testset "the step grows by at most max_step_growth and stays under max_step" begin
        tracer = unwrap(init(CurveProblem(Fold(), E[-1, 1], E[1, -1]), algorithm(E); first_step = E(1) / 100))
        for _ in 1:30
            before = tracer.step_length
            tracer, step = accepted(tracer)
            @test step.step_length <= before
            @test tracer.step_length <= 2 * step.step_length * (1 + eps(E))
            @test tracer.step_length <= one(E) / 2
        end
    end

    @testset "the first step stays under max_step and must be finite" begin
        problem = CurveProblem(Fold(), E[-1, 1], E[1, -1])
        @test unwrap(init(problem, algorithm(E); first_step = E(10))).step_length == one(E) / 2
        @test_throws ArgumentError init(problem, algorithm(E); first_step = typemax(E))
        # Positive and finite in the wider type, 0 and Inf in `E`.
        @test_throws ArgumentError init(problem, algorithm(E); first_step = nextfloat(zero(widen(E))))
        @test_throws ArgumentError init(problem, algorithm(E); first_step = floatmax(widen(E)))
    end

    @testset "a point metric freezes at the start and at every accepted point, and the step uses it" begin
        metric = CountingMetric(Ref(0))
        settings = algorithm(E; metric, turn_metric = FixedWeights(ones(E, 2)))
        tracer = unwrap(init(CurveProblem(Circle(one(E)), E[1, 0], E[0, 1]), settings; first_step = E(1) / 10))
        @test metric.freezes[] == 1
        for steps in 1:3
            tracer, _ = accepted(tracer)
            @test metric.freezes[] == steps + 1
            @test tracer.metric.weights == 1 .+ abs.(tracer.point)
            # The tangent has unit length in the metric frozen at its point.
            @test PathFollowing.metric_norm(tracer.metric, tracer.tangent) ≈ 1
        end
        @test abs(hypot(tracer.point...) - 1) <= 10 * tolerance
    end

    @testset "FixedWeights takes the points' element type and refuses a length that differs" begin
        frozen = frozen_metric(FixedWeights([2, 3]), E[1, 1])
        @test eltype(frozen.weights) == E
        @test metric_inner(frozen, E[1, 2], E[3, 4]) == 2 * 3 + 3 * 8
        @test metric_apply!!(zeros(E, 2), frozen, E[1, 2]) == E[2, 6]
        @test_throws DimensionMismatch frozen_metric(FixedWeights(ones(E, 3)), E[1, 1])
    end

    @testset "a refusing accept ends the trace Unresolved" begin
        settings = algorithm(E; accept = (start, predicted, corrected) -> false)
        failure = unwrap_error(traced(CurveProblem(Fold(), E[-1, 1], E[1, -1]), settings, 1; first_step = E(1) / 10))
        @test isa_variant(failure, StepFailure.Unresolved)
        @test failure.last_refusal == AcceptRefused
    end

    @testset "max_refusals ends the trace TooManyRefusals" begin
        settings = algorithm(E; accept = (start, predicted, corrected) -> false, max_refusals = 3)
        failure = unwrap_error(traced(CurveProblem(Fold(), E[-1, 1], E[1, -1]), settings, 1; first_step = E(1) / 10))
        @test isa_variant(failure, StepFailure.TooManyRefusals)
        @test failure.refusals == 4
    end

    @testset "with no refusal budget, a step refused at every length halves until it is unresolved" begin
        settings = algorithm(E; accept = (start, predicted, corrected) -> false, max_refusals = nothing)
        @test isnothing(settings.max_refusals)
        failure = unwrap_error(traced(CurveProblem(Fold(), E[-1, 1], E[1, -1]), settings, 1; first_step = E(1) / 10))
        @test isa_variant(failure, StepFailure.Unresolved) && failure.last_refusal == AcceptRefused
        # Each refusal at least halves the step: the subnormals bound the count.
        @test 1 < failure.refusals <= exponent(floatmax(E)) - exponent(floatmin(E)) + precision(E)
    end

    @testset "a NaN residual refuses the step until the trace is unresolved" begin
        failure = unwrap_error(traced(CurveProblem(Edge(one(E), false), E[0, 0], E[1, 1]), algorithm(E), 200; first_step = E(1) / 4))
        @test isa_variant(failure, StepFailure.Unresolved)
        @test failure.last_refusal == NotFiniteResidual
    end

    @testset "a singular bordered system refuses the step" begin
        failure = unwrap_error(traced(CurveProblem(Edge(one(E), true), E[0, 0], E[1, 1]), algorithm(E), 200; first_step = E(1) / 4))
        @test isa_variant(failure, StepFailure.Unresolved)
        @test failure.last_refusal == SingularSystem
    end

    # Calls: 1 at the start, 2 for the one correction a line needs, 3 at the corrected point for its tangent.
    @testset "a non-finite system at the corrected point refuses the step" begin
        settings = algorithm(E; max_refusals = 0)
        failure = unwrap_error(traced(CurveProblem(Flaky(0, 3, false), E[0, 0], E[1, 1]), settings, 1; first_step = E(1) / 10))
        @test failure.last_refusal == NotFiniteResidual
    end

    @testset "a singular tangent system at the corrected point refuses the step" begin
        settings = algorithm(E; max_refusals = 0)
        failure = unwrap_error(traced(CurveProblem(Flaky(0, 3, true), E[0, 0], E[1, 1]), settings, 1; first_step = E(1) / 10))
        @test failure.last_refusal == SingularSystem
    end

    @testset "one Newton iteration is not enough on a curved branch" begin
        settings = algorithm(E; corrector = Newton(; max_iterations = 1, converged = never, contraction_limit = one(E) / 2), max_refusals = 0)
        failure = unwrap_error(traced(CurveProblem(Circle(one(E)), E[1, 0], E[0, 1]), settings, 1; first_step = E(1) / 10))
        @test failure.last_refusal == NotConverged
    end

    @testset "a tiny contraction limit refuses as diverging" begin
        settings = algorithm(E; corrector = Newton(; max_iterations = 8, converged = never, contraction_limit = eps(E)), max_refusals = 0)
        failure = unwrap_error(traced(CurveProblem(Circle(one(E)), E[1, 0], E[0, 1]), settings, 1; first_step = E(1) / 10))
        @test failure.last_refusal == Diverging
    end

    @testset "a tight turn limit refuses a long step as too fast" begin
        control = StepControl(;
            target_contraction = nothing, target_correction = nothing, target_turn = one(E) / 1000, max_step_growth = E(2),
            max_step = E(1),
        )
        tracer = unwrap(init(CurveProblem(Circle(one(E)), E[1, 0], E[0, 1]), algorithm(E; step_control = control); first_step = one(E)))
        tracer, step = accepted(tracer)
        @test step.refusals > 0
        @test step.turn <= 2 * control.target_turn
    end

    @testset "a refused step retries at the length its measurements ask for" begin
        # Up the unit circle, 0.8 turns the tangent by atan(4/3) ≈ 0.93 rad, f ≈ 9.3 at a 0.1 target. One retry at
        # 0.8 / f meets the target, where halving would need three refusals.
        control = StepControl(;
            target_contraction = nothing, target_correction = nothing, target_turn = one(E) / 10, max_step_growth = E(2),
            max_step = E(1),
        )
        tracer = unwrap(init(CurveProblem(Circle(one(E)), E[1, 0], E[0, 1]), algorithm(E; step_control = control); first_step = E(4) / 5))
        tracer, step = accepted(tracer)
        @test step.refusals == 1
        @test step.step_length ≈ E(4) / 5 / (atan(E(4) / 3) * 10) rtol = 1 // 100
    end

    @testset "a short step grows by max_step_growth" begin
        # A 1e-3 step turns the tangent by 1e-3 rad, f = 1e-2 at a 0.1 target, so growth caps the next step at 2e-3.
        control = StepControl(;
            target_contraction = nothing, target_correction = nothing, target_turn = one(E) / 10, max_step_growth = E(2),
            max_step = E(1),
        )
        tracer = unwrap(init(CurveProblem(Circle(one(E)), E[1, 0], E[0, 1]), algorithm(E; step_control = control); first_step = E(1) / 1000))
        tracer, _ = accepted(tracer)
        @test tracer.step_length == E(2) / 1000
    end

    @testset "the corrector settles where its last correction passes, and that correction is the settle error" begin
        # A half step up the unit circle corrects by 0.125, 8.9e-3, 4.6e-5 and about 1e-9.
        for (settle, iterations) in ((one(E) / 500, 3), (one(E) / 50_000, 4))
            tracer, step = half_circle_step(settle)
            @test step.iterations == iterations
            @test abs(hypot(tracer.point...) - 1) <= settle
            @test tracer.settle_error ≈ hypot((tracer.point .- tracer.work.previous)...)
            @test tracer.settle_error <= settle
        end
    end

    @testset "property: an accepted point's last correction passed the settle tolerance, and it is the settle error" begin
        stretches = Data.Floats{E}(; minimum = E(1) / 4, maximum = E(4), nans = false, infs = false)
        steps = Data.Floats{E}(; minimum = E(1) / 100, maximum = E(1) / 2, nans = false, infs = false)
        exponents = Data.Integers(2, 6)
        @check max_examples = 50 function settled_last(stretch = stretches, first_step = steps, exponent = exponents)
            settle = max(one(E) / 10^exponent, 100 * eps(E))
            ellipse!! = (residual, jacobian, point) -> one_row!!(residual, jacobian, point[1]^2 + (point[2] / stretch)^2 - 1, (2 * point[1], 2 * point[2] / stretch^2))
            problem = CurveProblem(ellipse!!, E[1, 0], E[0, 1])
            control = StepControl(; target_contraction = nothing, target_correction = nothing, target_turn = nothing, max_step_growth = E(2), max_step = one(E))
            corrector = Newton(; max_iterations = 8, converged = Settled(settle), contraction_limit = one(E) / 2)
            tracer, _ = accepted(unwrap(init(problem, algorithm(E; corrector, step_control = control); first_step)))
            moves = abs.(tracer.point .- tracer.work.previous)
            maximum(moves) <= settle && isapprox(tracer.settle_error, hypot(moves...); rtol = sqrt(eps(E)))
        end
    end

    @testset "a custom distance measures corrections" begin
        settings = algorithm(E; corrector = Newton(; max_iterations = 8, converged = Settled(tolerance), contraction_limit = one(E) / 2, distance = LargestMove()))
        _, step = accepted(unwrap(init(CurveProblem(Circle(one(E)), E[1, 0], E[0, 1]), settings; first_step = E(1) / 10)))
        @test step.first_correction > 0
    end

    @testset "a converged point farther than reach predictor lengths from its predictor refuses the step, which retries shorter" begin
        # On the unit circle a step `h` along the tangent corrects by `1 - √(1 - h²)`: 0.45 of 3/4 and 0.19 of 3/8.
        control = StepControl(;
            target_contraction = nothing, target_correction = nothing, target_turn = nothing, max_step_growth = E(2),
            max_step = one(E), reach = one(E) / 4,
        )
        problem = CurveProblem(Circle(one(E)), E[1, 0], E[0, 1])
        _, step = accepted(unwrap(init(problem, algorithm(E; step_control = control); first_step = E(3) / 4)))
        @test (step.refusals, step.step_length) == (1, E(3) / 8)
        failure = failed_step(unwrap(init(problem, algorithm(E; step_control = control, max_refusals = 0); first_step = E(3) / 4)))
        @test isa_variant(failure, StepFailure.TooManyRefusals) && failure.last_refusal == FarFromPredictor
    end

    @testset "a start settled by a long correction allows no long move: the allowance is the corrector's error estimate" begin
        tracer = unwrap(init(CurveProblem(Circle(one(E)), E[1, 0], E[0, 1]), algorithm(E); first_step = E(1) / 100))
        # A step of 1/100 at reach 3, and a converged point 1/2 from its predictor (50 lengths), its last correction zero.
        tracer.work.predicted .= E[1, 1 // 100]
        tracer.work.iterate .= E[3 // 2, 1 // 100]
        copyto!(tracer.work.previous, tracer.work.iterate)
        # The start settled by a last correction of length 1/1000, which bounds its error.
        @reset tracer.settle_error = PathFollowing.settle_error(nothing, tracer, E[1 + 1 // 1000, 0], E[1, 0])
        @test PathFollowing.far_from_predictor(nothing, tracer, E(1) / 100)
        # A last correction of length 1, as a tolerance of 1 allows, cannot tell that move from the corrector's noise.
        @reset tracer.settle_error = PathFollowing.settle_error(nothing, tracer, E[2, 0], E[1, 0])
        @test !PathFollowing.far_from_predictor(nothing, tracer, E(1) / 100)
    end

    @testset "a step that corrects onto another branch is refused by its half steps, and its retry keeps the branch" begin
        # Without the check the step lands on the line; f is 0 with no targets, so a trigger of 0 runs the check.
        check = HalfStepCheck{E}(; trigger = 0, tolerance = one(E) / 100)
        for (split_check, point, refusals) in ((check, E[1 // 8, 1 // 2], 1), (nothing, E[-1 // 5, 1], 0))
            tracer = untargeted(ParabolaAndLine(), E[0, 0], E[0, 1]; first_step = 1, orientation_check = nothing, split_check)
            tracer, step = accepted(tracer)
            @test tracer.point ≈ point atol = 100 * tolerance
            @test step.refusals == refusals
        end
    end

    @testset "a half step that fails refuses the step as SplitMismatch" begin
        check = HalfStepCheck{E}(; trigger = 0, tolerance = one(E) / 100)
        failure = failed_step(untargeted(BandedLine(), E[0, 0], E[0, 1]; first_step = 1, split_check = check, max_refusals = 0))
        @test isa_variant(failure, StepFailure.TooManyRefusals) && failure.last_refusal == SplitMismatch
        @test last(accepted(untargeted(BandedLine(), E[0, 0], E[0, 1]; first_step = 1, max_refusals = 0))).step_length == 1
    end

    @testset "a failed half step refuses before the distance sees its rejected point, and the buffers come back" begin
        check = HalfStepCheck{E}(; trigger = 0, tolerance = one(E) / 100)
        corrector = Newton(; max_iterations = 8, converged = Settled(tolerance), contraction_limit = one(E) / 2, distance = BandAvoidingMove())
        banded(max_refusals) = untargeted(BandedLine(), E[0, 0], E[0, 1]; first_step = 1, corrector, split_check = check, max_refusals)
        failure = failed_step(banded(0))
        @test isa_variant(failure, StepFailure.TooManyRefusals) && failure.last_refusal == SplitMismatch
        tracer = banded(0)
        tracer.work.iterate .= E[0, 1]
        tracer.work.candidate .= E[0, 1]
        tracer.work.previous .= E[0, 7 // 8]
        # The half predictor at y = 1/2 lies in the band, so the first half step fails.
        tracer, refuses = PathFollowing.split_refused!!(nothing, tracer, check, zero(E), nothing, false, one(E))
        @test (refuses, tracer.work.iterate, tracer.work.candidate, tracer.work.previous) == (true, E[0, 1], E[0, 1], E[0, 7 // 8])
        # The retries: SplitMismatch at 1, NotFiniteResidual at 1/2 (in the band), and 1/4 lies on its edge.
        tracer, step = accepted(banded(60))
        @test (step.refusals, step.step_length, tracer.point) == (2, one(E) / 4, E[0, 1 // 4])
    end

    @testset "half steps of a landing end on the bound, and the step lands" begin
        check = HalfStepCheck{E}(; trigger = 0, tolerance = one(E) / 100)
        tracer = unwrap(init(CurveProblem(Circle(one(E)), E[1, 0], E[0, 1]), algorithm(E; split_check = check); first_step = E(1) / 10))
        bound = Bound(2, E(1) / 20)
        tracer, step = accepted(tracer, bound)
        @test step.refusals == 0
        @test tracer.point[2] == bound.value
        @test tracer.point[1] ≈ sqrt(1 - bound.value^2) atol = 100 * tolerance
    end

    @testset "the settle error is measured in the step metric frozen at the accepted point" begin
        # `CountingMetric` weighs each coordinate by 1 + |xᵢ|, so the metric frozen at the start measures another length.
        problem = CurveProblem(Circle(one(E)), E[1, 0], E[0, 1])
        tracer = unwrap(init(problem, algorithm(E; tolerance = one(E) / 100, metric = CountingMetric(Ref(0))); first_step = E(1) / 2))
        tracer, _ = accepted(tracer)
        scratch = similar(tracer.point)
        at_point = PathFollowing.metric_distance(tracer.metric, scratch, tracer.work.previous, tracer.point)
        at_start = PathFollowing.metric_distance(frozen_metric(CountingMetric(Ref(0)), problem.start), scratch, tracer.work.previous, tracer.point)
        @test tracer.settle_error ≈ at_point
        @test !(at_point ≈ at_start)
    end

    @testset "a first-correction refusal past the distance's largest length cannot fire, and throws" begin
        capped(target) = algorithm(
            E; corrector = Newton(; max_iterations = 8, converged = Settled(tolerance), contraction_limit = one(E) / 2, distance = CappedMove()),
            step_control = StepControl(; target_contraction = nothing, target_correction = target, target_turn = nothing, max_step_growth = E(2), max_step = one(E)),
        )
        # The refusal factor 2 refuses a first correction past 4 times the target.
        @test_throws ArgumentError capped(one(E) / 4)
        @test capped(one(E) / 5).step_control.target_correction == one(E) / 5
        @test capped(nothing).step_control.target_correction === nothing
        @test correction_bound(MetricDistance(), E) === E(Inf32)
    end

    @testset "a predictor that reaches the bound is cut there and lands with the value exact" begin
        tracer = unwrap(init(CurveProblem(Circle(one(E)), E[1, 0], E[0, 1]), algorithm(E); first_step = E(1) / 10))
        bound = Bound(2, E(1) / 20)
        tracer, step = accepted(tracer, bound)
        @test tracer.point[2] == bound.value
        @test tracer.point[1] ≈ sqrt(1 - bound.value^2) atol = 100 * tolerance
        # The next step grows from the cut predictor, by at most max_step_growth.
        @test step.step_length ≈ E(1) / 20 rtol = 10 * eps(E)
        @test tracer.step_length <= 2 * step.step_length
        @test_throws ArgumentError step!!(tracer, bound)
    end

    @testset "a step short of the bound moves as without one" begin
        free = unwrap(init(CurveProblem(Circle(one(E)), E[1, 0], E[0, 1]), algorithm(E); first_step = E(1) / 10))
        bounded = unwrap(init(CurveProblem(Circle(one(E)), E[1, 0], E[0, 1]), algorithm(E); first_step = E(1) / 10))
        free, _ = accepted(free)
        bounded, _ = accepted(bounded, Bound(2, E(1) / 4))
        @test (bounded.point, bounded.tangent, bounded.step_length) == (free.point, free.tangent, free.step_length)
    end

    @testset "a corrected point past the bound refuses the step as PastBound" begin
        # From (-7/25, -24/25) a step of 4/5 passes y's minimum at (0, -1) to (3/5, -4/5), past y = -9/10.
        tracer = untargeted(Circle(one(E)), E[-7, -24] ./ 25, E[1, 0]; first_step = E(4) / 5, max_refusals = 0)
        tracer, outcome = step!!(tracer, Bound(2, -E(9) / 10))
        @test isa_variant(unwrap_error(outcome), StepFailure.TooManyRefusals) && unwrap_error(outcome).last_refusal == PastBound
        @test tracer.point == E[-7, -24] ./ 25
    end

    @testset "a refused landing shrinks from the cut predictor's length" begin
        # A first step of 100 is cut to 9/10 at y = 9/10 and turns too fast. Shrinking 100 repeated that cut until
        # TooManyRefusals.
        control = StepControl(;
            target_contraction = one(E) / 4, target_correction = one(E) / 10, target_turn = one(E) / 10,
            max_step_growth = E(2), max_step = E(100),
        )
        settings = algorithm(E; step_control = control, max_refusals = 2)
        tracer = unwrap(init(CurveProblem(Circle(one(E)), E[1, 0], E[0, 1]), settings; first_step = E(100)))
        bound = Bound(2, E(9) / 10)
        tracer, step = accepted(tracer, bound)
        @test step.refusals >= 1
        @test step.step_length < E(9) / 10
        tracer, _ = bounded_steps(tracer, bound, 40)
        @test tracer.point ≈ [sqrt(E(19)) / 10, E(9) / 10] atol = 100 * tolerance
        # Without retries the failure reports the cut predictor it refused.
        settings = algorithm(E; step_control = control, max_refusals = 0)
        tracer = unwrap(init(CurveProblem(Circle(one(E)), E[1, 0], E[0, 1]), settings; first_step = E(100)))
        failure = failed_step(tracer, bound)
        @test isa_variant(failure, StepFailure.TooManyRefusals)
        @test failure.step_length ≈ E(9) / 10 rtol = 10 * eps(E)
    end

    @testset "a bound one ulp ahead lands: the resolution check judges the step before the cut" begin
        # Up x = y from (1, 1), y = nextfloat(1) is one ulp away. Judging the cut predictor failed this as Unresolved.
        tracer = unwrap(init(CurveProblem(Line(), E[1, 1], E[1, 1]), algorithm(E); first_step = E(1) / 10))
        bound = Bound(2, nextfloat(one(E)))
        tracer, _ = accepted(tracer, bound)
        @test tracer.point[2] == bound.value
        @test tracer.point[1] ≈ bound.value atol = 4 * eps(E)
        # An uncut step ends one ulp short of the bound, and the next step lands on it.
        tracer = unwrap(init(CurveProblem(Line(), E[0, 0], E[1, 1]), algorithm(E); first_step = E(1) / 10))
        tracer, step = accepted(tracer, Bound(2, one(E)))
        @test step.step_length == E(1) / 10
        bound = Bound(2, nextfloat(tracer.point[2]))
        tracer, _ = accepted(tracer, bound)
        @test tracer.point[2] == bound.value
    end

    @testset "a retry that rounds back to the refused step fails Unresolved" begin
        # A shrink one ulp under 1 cannot shorten the smallest subnormal step: each retry repeated it until TooManyRefusals.
        targets = (; target_contraction = nothing, target_correction = nothing, target_turn = nothing)
        control = StepControl(; targets..., max_step_growth = E(2), max_step = one(E), shrink = prevfloat(one(E)))
        settings = algorithm(E; step_control = control, accept = (start, predicted, corrected) -> false)
        tracer, outcome = step!!(unwrap(init(CurveProblem(Line(), E[0, 0], E[1, 1]), settings; first_step = nextfloat(zero(E)))))
        failure = unwrap_error(outcome)
        @test isa_variant(failure, StepFailure.Unresolved) && (failure.last_refusal, failure.refusals) == (AcceptRefused, 1)
        # The returned tracer holds the step the resolution check judged.
        @test tracer.step_length == nextfloat(zero(E))
    end

    @testset "a bounded step over a fold's top stops at the first crossing" begin
        # From (3/5, 4/5) a step of 4/5 passes y's maximum at (0, 1) to (-7/25, 24/25), so it meets y = 9/10 twice.
        tracer = untargeted(Circle(one(E)), E[3, 4] ./ 5, E[-1, 0]; first_step = E(4) / 5)
        tracer, steps = bounded_steps(tracer, Bound(2, E(9) / 10), 1)
        @test length(steps) == 1
        @test tracer.point ≈ [sqrt(E(19)) / 10, E(9) / 10] atol = 100 * tolerance
        # y = 1/2 lies behind the start, so the same step stays above it.
        tracer = untargeted(Circle(one(E)), E[3, 4] ./ 5, E[-1, 0]; first_step = E(4) / 5)
        tracer, _ = accepted(tracer, Bound(2, one(E) / 2))
        @test tracer.point ≈ E[-7, 24] ./ 25 atol = 100 * tolerance
    end

    @testset "a bounded step through a fold's bottom stops at the first crossing" begin
        # From (-7/25, -24/25) the step passes y's minimum at (0, -1); it meets y = -97/100 at x = ∓0.2431, the first at -.
        tracer = untargeted(Circle(one(E)), E[-7, -24] ./ 25, E[1, 0]; first_step = E(4) / 5)
        tracer, steps = bounded_steps(tracer, Bound(2, -E(97) / 100), 1)
        @test length(steps) == 1
        @test tracer.point ≈ [-sqrt(1 - E(97)^2 / 100^2), -E(97) / 100] atol = 100 * tolerance
        # y = -9/10 lies past the step's end, and first behind its start: the refused step shrinks and lands ahead.
        tracer = untargeted(Circle(one(E)), E[-7, -24] ./ 25, E[1, 0]; first_step = E(4) / 5)
        tracer, steps = bounded_steps(tracer, Bound(2, -E(9) / 10), 4)
        @test first(steps).refusals == 1
        @test tracer.point ≈ [sqrt(E(19)) / 10, -E(9) / 10] atol = 100 * tolerance
    end

    @testset "a bound behind the start stays behind: the trace lands on its next crossing ahead" begin
        # From (1, 0) toward rising y the step of 4/5 ends at (3/5, 4/5); y = -1/10 lies behind, at (√99/10, -1/10).
        tracer = untargeted(Circle(one(E)), E[1, 0], E[0, 1]; first_step = E(4) / 5)
        bound = Bound(2, -E(1) / 10)
        tracer, _ = accepted(tracer, bound)
        @test tracer.point ≈ E[3, 4] ./ 5 atol = 100 * tolerance
        tracer, _ = bounded_steps(tracer, bound, 20)
        @test tracer.point ≈ [-sqrt(E(99)) / 10, -E(1) / 10] atol = 100 * tolerance
    end

    @testset "a step over two folds of λ = u - u³ stops at the crossing between them" begin
        # From u = -3/5 a step of 227/200 passes λ's minimum and maximum to u ≈ 3/5, λ ≈ 48/125; λ = 0 lies between.
        tracer = untargeted(Wiggle(), [-E(3) / 5, -E(48) / 125], E[1, 0]; first_step = E(227) / 200, max_refusals = 0)
        failure = failed_step(tracer, Bound(2, zero(E)))
        @test isa_variant(failure, StepFailure.TooManyRefusals) && failure.last_refusal == PastBound
        tracer = untargeted(Wiggle(), [-E(3) / 5, -E(48) / 125], E[1, 0]; first_step = E(227) / 200)
        tracer, _ = bounded_steps(tracer, Bound(2, zero(E)), 20)
        @test tracer.point ≈ E[0, 0] atol = 100 * tolerance
    end

    @testset "a NaN start fails NotFiniteStart" begin
        failure = unwrap_error(init(CurveProblem(Edge(-one(E), false), E[0, 0], E[1, 1]), algorithm(E); first_step = one(E)))
        @test isa_variant(failure, StepFailure.NotFiniteStart)
    end

    @testset "a direction normal to the tangent fails SingularStart" begin
        failure = unwrap_error(init(CurveProblem(Circle(one(E)), E[1, 0], E[1, 0]), algorithm(E); first_step = one(E)))
        @test isa_variant(failure, StepFailure.SingularStart)
    end

    @testset "argument checks" begin
        @test_throws ArgumentError Newton(; max_iterations = 0, converged = Settled(tolerance), contraction_limit = one(E) / 2)
        # Past 1/2, a correction that passes the tolerance can leave more than its own length in its iterate.
        for limit in (zero(E), nextfloat(one(E) / 2), E(NaN32))
            @test_throws ArgumentError Newton(; max_iterations = 1, converged = Settled(tolerance), contraction_limit = limit)
        end
        @test Newton(; max_iterations = 1, converged = Settled(tolerance), contraction_limit = one(E) / 2).contraction_limit === one(E) / 2
        # Checked in `E`: a wider limit an ulp past 1/2 rounds to 1/2 there.
        @test PathFollowing.contraction_limit_in(E, nextfloat(one(widen(E)) / 2)) === one(E) / 2
        @test_throws ArgumentError PathFollowing.contraction_limit_in(E, nextfloat(zero(widen(E))))
        targets = (; target_contraction = nothing, target_correction = nothing, target_turn = nothing)
        @test_throws ArgumentError StepControl(; targets..., max_step_growth = one(E) / 2, max_step = one(E))
        @test_throws ArgumentError StepControl(; targets..., max_step_growth = one(E), max_step = zero(E))
        @test_throws ArgumentError StepControl(; targets..., max_step_growth = one(E), max_step = one(E), refusal_factor = 0)
        @test_throws ArgumentError StepControl(; targets..., max_step_growth = one(E), max_step = one(E), reach = 0)
        @test_throws ArgumentError StepControl(; targets..., max_step_growth = one(E), max_step = one(E), reach = NaN32)
        # A reach positive and finite in a wider type can round to zero, which refuses every move, or to Inf in `E`.
        @test_throws ArgumentError StepControl(; targets..., max_step_growth = one(E), max_step = one(E), reach = nextfloat(zero(widen(E))))
        @test_throws ArgumentError StepControl(; targets..., max_step_growth = one(E), max_step = one(E), reach = floatmax(widen(E)))
        @test StepControl(; targets..., max_step_growth = one(E), max_step = one(E), reach = Inf32).reach == Inf32
        @test_throws ArgumentError StepControl(; targets..., max_step_growth = one(E), max_step = one(E), shrink = 1)
        # A shrink an eighth of eps under 1 rounds to 1 in `E`, and each retry repeated its step.
        @test_throws ArgumentError StepControl(; targets..., max_step_growth = one(E), max_step = one(E), shrink = 1 - big(eps(E)) / 8)
        # A step control in a wider type computes its shrink there, where it can round back to the same step in `E`.
        wide = StepControl(; targets..., max_step_growth = widen(E)(2), max_step = one(E))
        @test_throws ArgumentError init(CurveProblem(Fold(), E[-1, 1], E[1, -1]), algorithm(E; step_control = wide); first_step = one(E) / 10)
        @test_throws ArgumentError StepControl(; targets..., max_step_growth = E(Inf), max_step = E(Inf))
        @test StepControl(; targets..., max_step_growth = E(2), max_step = E(Inf)).max_step == E(Inf)
        @test_throws ArgumentError StepControl(;
            target_contraction = nothing, target_correction = nothing, target_turn = -one(E), max_step_growth = one(E),
            max_step = one(E),
        )
        @test_throws ArgumentError HalfStepCheck{E}(; trigger = -1, tolerance = one(E))
        @test_throws ArgumentError HalfStepCheck{E}(; trigger = NaN32, tolerance = one(E))
        @test_throws ArgumentError HalfStepCheck{E}(; trigger = Inf32, tolerance = one(E))
        @test_throws ArgumentError HalfStepCheck{E}(; trigger = one(E), tolerance = 0)
        @test_throws ArgumentError HalfStepCheck{E}(; trigger = one(E), tolerance = NaN32)
        # Numbers finite in a wider type can round to zero or to Inf in `E`.
        @test_throws ArgumentError HalfStepCheck{E}(; trigger = one(E), tolerance = nextfloat(zero(widen(E))))
        @test_throws ArgumentError HalfStepCheck{E}(; trigger = one(E), tolerance = floatmax(widen(E)))
        @test_throws ArgumentError HalfStepCheck{E}(; trigger = floatmax(widen(E)), tolerance = one(E))
        @test HalfStepCheck{E}(; trigger = 0, tolerance = 1 // 3).tolerance === E(1 // 3)
        # A step past the refusal factor 2 is refused before the check, so a trigger there never fires; one at 2 can.
        @test_throws ArgumentError algorithm(E; split_check = HalfStepCheck{E}(; trigger = nextfloat(E(2)), tolerance = one(E)))
        @test algorithm(E; split_check = HalfStepCheck{E}(; trigger = 2, tolerance = one(E))).split_check.trigger == 2
        # A step at the trigger runs the check; one an ulp under it does not.
        at_refusal = HalfStepCheck{E}(; trigger = 2, tolerance = one(E))
        @test PathFollowing.split_fires(at_refusal, E(2))
        @test !PathFollowing.split_fires(at_refusal, prevfloat(E(2)))
        # Without a type parameter, the check takes its numbers' promoted float type.
        @test HalfStepCheck(; trigger = 13 // 10, tolerance = one(E) / 1000) === HalfStepCheck{E}(; trigger = 13 // 10, tolerance = one(E) / 1000)
        @test HalfStepCheck(; trigger = 2, tolerance = 1) isa HalfStepCheck{float(Int)}
        wide_check = HalfStepCheck{widen(E)}(; trigger = 1, tolerance = 1)
        @test_throws ArgumentError init(CurveProblem(Fold(), E[-1, 1], E[1, -1]), algorithm(E; split_check = wide_check); first_step = one(E) / 10)
        @test_throws ArgumentError algorithm(E; min_step_ulps = -1)
        @test_throws ArgumentError algorithm(E; max_refusals = -1)
        @test_throws ArgumentError init(CurveProblem(Fold(), E[0, 0], E[1, 0]), algorithm(E); first_step = zero(E))
        @test_throws DimensionMismatch init(CurveProblem(Fold(), E[0, 0], E[1, 0, 0]), algorithm(E); first_step = one(E))
        @test AcceptAll()(E[0], E[0], E[0])
        euclidean, scratch = FixedWeights(ones(E, 2)), zeros(E, 2)
        @test metric_angle(euclidean, scratch, E[1, 0], E[0, 1]) ≈ E(π) / 2
        @test metric_angle(euclidean, scratch, E[1, 0], E[-1, 0]) ≈ E(π)
        # A small angle keeps its digits: acos of the cosine would lose half of them.
        @test metric_angle(euclidean, scratch, E[1, 0], [cos(E(1) / 10^4), sin(E(1) / 10^4)]) ≈ E(1) / 10^4 rtol = 8 * eps(E)
        @test metric_angle(FixedWeights(E[4, 1]), scratch, E[1, 0], E[1, 2]) ≈ E(π) / 4
        @test factored_sign(E[1 0; 0 NaN32], [1, 2]) == 0
    end

    @testset "a warm step allocates nothing, nor does one that lands on a bound" begin
        tracer = unwrap(init(CurveProblem(Circle(one(E)), E[1, 0], E[0, 1]), algorithm(E); first_step = E(1) / 10))
        @test last(warm_bytes(tracer, nothing)) == 0
        tracer = unwrap(init(CurveProblem(Circle(one(E)), E[1, 0], E[0, 1]), algorithm(E); first_step = E(1) / 10))
        bound = Bound(2, E(3) / 20)
        tracer, bytes = warm_bytes(tracer, bound)
        @test bytes == 0
        @test tracer.point[2] == bound.value
        check = HalfStepCheck{E}(; trigger = 0, tolerance = one(E) / 100)
        for bound in (nothing, Bound(2, E(3) / 20))
            tracer = unwrap(init(CurveProblem(Circle(one(E)), E[1, 0], E[0, 1]), algorithm(E; split_check = check); first_step = E(1) / 10))
            @test last(warm_bytes(tracer, bound)) == 0
            # A static tracer allocates nothing either, and its steps infer.
            static_problem = CurveProblem(Circle(one(E)), SVector{2, E}(1, 0), SVector{2, E}(0, 1))
            static = unwrap(init(static_problem, algorithm(E; split_check = check, linear_solver = StaticLU()); first_step = E(1) / 10))
            @test last(warm_bytes(static, bound)) == 0
            @inferred step!!(static, bound)
        end
    end

    @testset "a trace on a GPU array type stays on it and follows the CPU trace, bounded or not" begin
        # JLArrays refuses scalar indexing as a GPU does, so any element-by-element access throws here.
        for linear_solver in (LUFactorization(), QRFactorization()), bound in (nothing, Bound(2, E(3) / 4))
            # LU keeps the factors the orientation sign reads; QR makes the tracer factor `[J; tᵀ]` itself.
            settings = algorithm(E; linear_solver)
            host = unwrap(init(CurveProblem(Circle(one(E)), E[1, 0], E[0, 1]), settings; first_step = E(1) / 10))
            device = unwrap(init(CurveProblem(Circle(one(E)), JLArray(E[1, 0]), JLArray(E[0, 1])), settings; first_step = E(1) / 10))
            @test all(array -> array isa JLArray, (device.point, device.work.jacobian, device.work.bordered, device.linear.oriented, device.metric.weights))
            # The bound at y = 3/4 lies 0.85 of arc away, 9 steps of 0.1: the cap only stops a trace that never lands.
            limit = isnothing(bound) ? 8 : 20
            while device.steps < limit && (isnothing(bound) || host.point[bound.coordinate] != bound.value)
                host, _ = accepted(host, bound)
                device, _ = accepted(device, bound)
                @test Array(device.point) ≈ host.point rtol = 10 * tolerance
                @test device.orientation == host.orientation == 1
            end
            # The bound stops both traces on it, its value exact.
            @test isnothing(bound) || Array(device.point)[2] == host.point[2] == bound.value
            @test all(iszero, Array(device.linear.oriented)) == (linear_solver isa LUFactorization)
        end
        # Host weights of any vector type reach the device in one copy, never element by element.
        for weights in (E[1, 2], E(1):E(1):E(2), view(E[1, 2, 3], 1:2))
            frozen = frozen_metric(FixedWeights(weights), JLArray(E[1, 0]))
            @test frozen.weights isa JLArray && Array(frozen.weights) == collect(weights)
        end
    end

    @testset "static and group starts trace as a mutable one, step for step, through refusals and failures" begin
        # A point metric, beside the parity cases. Each case's last entry checks its refusals and failure, if any.
        point_metric = (; metric = CountingMetric(Ref(0)), turn_metric = FixedWeights(ones(E, 2)))
        cases = (parity_cases(E)..., (Fold(), E[-1, 1], E[1, -1], point_metric, nothing, 12, (_, failure) -> isnothing(failure)))
        for (system, start, direction, overrides, bound, limit, took_its_path) in cases
            in_place = unwrap(init(CurveProblem(system, start, direction), algorithm(E; overrides...); first_step = E(1) / 4))
            N = length(start)
            static_problem = CurveProblem(system, SVector{N, E}(start), SVector{N, E}(direction))
            static = unwrap(init(static_problem, algorithm(E; linear_solver = StaticLU(), overrides...); first_step = E(1) / 4))
            memory = host_memory(E)
            group_problem = CurveProblem(system, group_vector(memory, start), group_vector(memory, direction))
            group = unwrap(init(group_problem, algorithm(E; linear_solver = GroupLU(), overrides...); first_step = E(1) / 4))
            @test static.point isa SVector{N, E} && static.work.bordered isa SMatrix{N, N, E} && static.linear isa PathFollowing.StaticSolve
            @test group.point isa GroupHandle{E, 1} && group.work.bordered isa GroupHandle{E, 2} && group.linear isa PathFollowing.GroupSolve
            @test group_view(memory, group).point isa GroupVector{E} && group_view(memory, group).work.bordered isa GroupMatrix{E}
            @test static.orientation == group.orientation == in_place.orientation
            refusals, failure = 0, nothing
            for _ in 1:limit
                in_place, expected = step!!(in_place, bound)
                static, outcome = step!!(static, bound)
                group, grouped = step!!(memory, group, bound)
                @test is_error(outcome) == is_error(grouped) == is_error(expected)
                if is_error(expected)
                    failure = unwrap_error(outcome)
                    @test failure == unwrap_error(grouped) == unwrap_error(expected)
                    break
                end
                refusals += unwrap(outcome).refusals
                @test unwrap(outcome).refusals == unwrap(grouped).refusals == unwrap(expected).refusals
                @test static.point ≈ in_place.point atol = 10 * tolerance
                @test collect(group_view(memory, group).point) ≈ in_place.point atol = 10 * tolerance
                @test static.orientation == group.orientation == in_place.orientation
                isnothing(bound) || static.point[bound.coordinate] != bound.value || break
            end
            @test took_its_path(refusals, failure)
            @test isnothing(bound) || static.point[bound.coordinate] == group_view(memory, group).point[bound.coordinate] == in_place.point[bound.coordinate] == bound.value
        end
    end

    @testset "a warm step in group memory allocates nothing, nor does one that lands on a bound" begin
        check = HalfStepCheck{E}(; trigger = 0, tolerance = one(E) / 100)
        for bound in (nothing, Bound(2, E(3) / 20)), split_check in (nothing, check)
            memory = host_memory(E)
            problem = CurveProblem(Circle(one(E)), group_vector(memory, E[1, 0]), group_vector(memory, E[0, 1]))
            settings = algorithm(E; split_check, metric = FixedWeights(SVector{2, E}(1, 1)), linear_solver = GroupLU())
            # A fresh tracer each: tracers in one group memory share their arrays, and the warm one ends on the bound.
            tracer = unwrap(init(problem, settings; first_step = E(1) / 10))
            # Handles only: a copy of the tracer copies no array.
            @test isbits(tracer)
            @inferred step!!(memory, tracer, bound)
            @test last(warm_bytes(memory, unwrap(init(problem, settings; first_step = E(1) / 10)), bound)) == 0
        end
    end

    @testset "a trace's steps take no group memory past its start's, and 16 coordinates fit the CUDA tests' 2048 floats" begin
        for (system, start, direction, overrides, bound, limit, _) in (parity_cases(E)..., wide_chain_case(E))
            settings = algorithm(E; overrides..., linear_solver = GroupLU())
            first_step = system isa TanhChain ? E(1) / 20 : E(1) / 4
            started = host_memory(E; floats = 2048)
            init(CurveProblem(system, group_vector(started, start), group_vector(started, direction)), settings; first_step)
            traced, refusals = host_memory(E; floats = 2048), fill(-2, limit)
            record_group_trace!(fill(E(NaN), length(start), limit), refusals, traced, system, start, direction, settings, first_step, bound, limit)
            @test traced.cursor == started.cursor
            system isa TanhChain && length(start) == 16 && @test all(>=(0), refusals)
        end
    end

    @testset "GroupLU solves within its backward error bound and gives the sign of the determinant, its row swaps counted" begin
        entry = Data.Floats{E}(; minimum = -E(4), maximum = E(4), nans = false, infs = false)
        @check max_examples = 60 function within_bound(order = Data.Integers(1, 6), entries = Data.Vectors(entry; min_size = 49, max_size = 49))
            matrix, rhs = reshape(entries[1:(order^2)], order, order), entries[43:(42 + order)]
            memory = host_memory(E)
            bordered, column = copyto!(group_array(memory, E, (order, order)), matrix), group_vector(memory, rhs)
            state = PathFollowing.GroupSolve(Int8(0), PathFollowing.stored(similar(column)))
            state, solution, solved = PathFollowing.linear_solve!!(state, bordered, column, similar(column))
            # A failed solve met a zero pivot, so a zero sign, or overflowed to a solution that is not finite.
            solved || return iszero(state.sign) || !all(isfinite, collect(solution))
            pivots = collect(PathFollowing.stored_view(memory, state.pivots))
            lu_residual_ratio(matrix, rhs, collect(bordered), pivots, collect(solution)) <= 1 || return false
            # Rounding can flip the determinant's sign only near a singular matrix.
            LinearAlgebra.cond(Float64.(matrix)) < 100 || return true
            # A Float64 determinant of subnormal entries underflows to zero, so the exact sign comes from BigFloat.
            return state.sign == sign(det(big.(matrix)))
        end
        # Two row swaps cancel in the sign; a zero column fails the solve.
        memory = host_memory(E)
        cycle = copyto!(group_array(memory, E, (3, 3)), E[0 1 0; 0 0 1; 1 0 0])
        rhs = group_vector(memory, E[1, 2, 3])
        state, solution, solved = PathFollowing.linear_solve!!(PathFollowing.GroupSolve(Int8(0), PathFollowing.stored(similar(rhs))), cycle, rhs, similar(rhs))
        @test solved && state.sign == 1 && collect(solution) == E[3, 1, 2]
        singular = copyto!(group_array(memory, E, (2, 2)), E[1 0; 1 0])
        state, _, solved = PathFollowing.linear_solve!!(state, singular, group_vector(memory, E[1, 1]), group_vector(memory, E[0, 0]))
        @test !solved && state.sign == 0
        # A CI draw: the least subnormal pivot overflows the solution, so the solve fails with the pivot's sign.
        tiny = nextfloat(zero(E))
        lone, past_range = copyto!(group_array(memory, E, (1, 1)), fill(-tiny, 1, 1)), group_vector(memory, [ldexp(one(E), exponent(floatmax(E)) + 1 + exponent(tiny))])
        state, solution, solved = PathFollowing.linear_solve!!(state, lone, past_range, group_vector(memory, E[0]))
        @test !solved && state.sign == -1 && collect(solution) == E[-Inf]
    end

    @testset "group memory hands out its entries once, refuses a foreign type and a full buffer, and needs GroupLU" begin
        memory = host_memory(E; floats = 8)
        taken = group_array(memory, E, (3,))
        @test (taken.handle, memory.cursor) == (GroupHandle{E, 1}(0, (3,)), [3])
        @test group_array(memory, E, (5,)).handle.offset == 3
        @test_throws "the group memory is full" group_array(memory, E, (1,))
        # A dimension past `Int32` throws before the cursor moves; a `LinRange` stands in for a buffer of 2^31 entries.
        huge = GroupMemory(LinRange{E}(0, 0, 2^31), zeros(Int, 1), 1, 1, LoneLane())
        @test_throws InexactError group_array(huge, E, (2^31,))
        @test huge.cursor == [0]
        other = E === Float32 ? Float64 : Float32
        @test_throws "its own float type" group_array(memory, other, (1,))
        @test_throws "its own float type" group_array(memory, Int, (1,))
        @test reset_memory!(memory).cursor == [0] && group_array(memory, E, (8,)).handle.offset == 0
        # A handle views only through a memory of its own float type.
        @test PathFollowing.stored_view(memory, PathFollowing.stored(taken)) === taken && PathFollowing.stored(taken) === taken.handle
        @test_throws "takes its GroupMemory beside it" PathFollowing.stored_view(nothing, taken.handle)
        @test_throws "its memory's float type" PathFollowing.stored_view(GroupMemory(zeros(other, 4), zeros(Int, 1), 1, 1, LoneLane()), taken.handle)
        memory = host_memory(E)
        problem = CurveProblem(Circle(one(E)), group_vector(memory, E[1, 0]), group_vector(memory, E[0, 1]))
        @test_throws "needs linear_solver = GroupLU()" init(problem, algorithm(E); first_step = E(1) / 10)
        @test_throws "GroupLU needs a start in a workgroup's memory" init(CurveProblem(Circle(one(E)), E[1, 0], E[0, 1]), algorithm(E; linear_solver = GroupLU()); first_step = E(1) / 10)
        vector = group_vector(memory, E[1, 2, 3])
        @test PathFollowing.set_element!!(vector, 2, E(7)) === vector && collect(vector) == E[1, 7, 3]
        @test @into!!(vector .= 2 .* vector) === vector && collect(vector) == E[2, 14, 6]
        @test collect(fill!(vector, E(5))) == E[5, 5, 5] && collect(PathFollowing.unit!!(vector, 3)) == E[0, 0, 1]
    end

    @testset "a group trace steps and lands only in the memory it started in, and lands as a Vector trace lands" begin
        memory, other = host_memory(E), host_memory(E)
        settings = algorithm(E; linear_solver = GroupLU())
        tracer = unwrap(init(CurveProblem(Circle(one(E)), group_vector(memory, E[1, 0]), group_vector(memory, E[0, 1])), settings; first_step = E(1) / 10))
        @test_throws "the group memory it started in" step!!(other, tracer)
        @test_throws "the group memory it started in" step!!(tracer)
        @test_throws "the group memory it started in" group_view(other, tracer)
        # A wrong memory throws before the landing reads its point or its budget.
        @test_throws "the group memory it started in" land!!(other, tracer, Bound(2, zero(E)), 0)
        # Two memories over one buffer under separate cursors are separate memories.
        @test_throws "the group memory it started in" land!!(GroupMemory(memory.floats, zeros(Int, 1), 1, 1, LoneLane()), tracer, Bound(2, zero(E)), 0)
        # A device memory's tag reads device pointers, which the host builds from an integer address.
        @test PathFollowing.address(reinterpret(Core.LLVMPtr{E, 1}, UInt(64))) == PathFollowing.address(Ptr{E}(64)) == UInt(64)
        split_problem = CurveProblem(Circle(one(E)), group_vector(memory, E[1, 0]), group_vector(other, E[0, 1]))
        @test_throws "lies in its start's memory" init(split_problem, settings; first_step = E(1) / 10)
        bound = Bound(2, E(1) / 2)
        tracer, landed = land!!(memory, tracer, bound, 40)
        _, expected = land!!(unwrap(init(CurveProblem(Circle(one(E)), E[1, 0], E[0, 1]), algorithm(E); first_step = E(1) / 10)), bound, 40)
        @test unwrap(landed) isa GroupVector{E} && unwrap(landed)[2] == bound.value
        @test collect(unwrap(landed)) ≈ unwrap(expected) atol = 10 * tolerance
    end

    @testset "a group array checks its own bounds, has no setindex!, and refuses a broadcast from entries it overlaps" begin
        memory = host_memory(E)
        first_array, second_array = group_vector(memory, E[1, 2]), group_vector(memory, E[3, 4])
        @test_throws BoundsError first_array[3]
        @test_throws BoundsError PathFollowing.store!(first_array, E(9), 3)
        @test collect(second_array) == E[3, 4]
        @test_throws "setindex!" (first_array[1] = E(9))
        vector = group_vector(memory, E[1, 2, 3])
        @test_throws "reads an array that overlaps it" @into!! vector .= view(vector, 3:-1:1)
        @test_throws "reads an array that overlaps it" @into!! vector .= view(memory.floats, (vector.handle.offset + 3):-1:(vector.handle.offset + 1))
        floats = zeros(E, 8)
        viewed = GroupMemory(view(floats, 1:8), zeros(Int, 1), 1, 1, LoneLane())
        @test_throws "reads an array that overlaps it" @into!! group_vector(viewed, E[1, 2, 3]) .= view(floats, 3:-1:1)
        @test_throws "reads an array that overlaps it" @into!! group_vector(viewed, E[1, 2]) .= group_vector(GroupMemory(view(floats, 5:8), zeros(Int, 1), 1, 1, LoneLane()), E[1, 2])
        @test collect(vector) == E[1, 2, 3]
        @test_throws "not negative" group_array(memory, E, (-1,))
        @test_throws OverflowError group_array(memory, E, (typemax(Int), 2))
        # A broadcast inlines the overlap check for views, so only a direct call covers its signature line.
        @test PathFollowing.shares_entries(vector, view(vector, 1:2)) && PathFollowing.shares_entries(vector, memory.floats)
        @test !PathFollowing.shares_entries(vector, E[1, 2, 3]) && !PathFollowing.shares_entries(vector, view(E[1, 2, 3], 1:2))
        refusals = fill(-2, 3)
        record_group_trace!(fill(E(NaN), 2, 3), refusals, host_memory(E), Circle(one(E)), E[NaN, 0], E[0, 1], algorithm(E; linear_solver = GroupLU()), E(1) / 4, nothing, 3)
        @test refusals == [-3, -2, -2]
        @test collect(@into!!(vector .= vector .+ first_array[2] .* E[1, 2, 3])) == E[3, 6, 9]
        @test collect(copyto!(vector, view(second_array, [2, 1, 2]))) == E[4, 3, 4]
        @test collect(copyto!(vector, E[7])) == E[7, 3, 4] && collect(copyto!(vector, E[])) == E[7, 3, 4]
        @test_throws BoundsError copyto!(first_array, E[1, 2, 3])
        @test_throws "reads an array that overlaps it" copyto!(vector, view(vector, 3:-1:1))
    end

    @testset "a group memory refuses a lane outside its group, LoneLane() for several lanes, and a cursor not of one entry" begin
        floats, cursor = zeros(E, 4), zeros(Int, 1)
        @test_throws "lies in 1:lanes" GroupMemory(floats, cursor, 0, 1, LoneLane())
        @test_throws "lies in 1:lanes" GroupMemory(floats, cursor, 3, 2, GroupBarrier())
        @test_throws "a group of one lane" GroupMemory(floats, cursor, 1, 2, LoneLane())
        @test_throws "one entry" GroupMemory(floats, zeros(Int, 2), 1, 1, LoneLane())
        @test_throws "separate storage" GroupMemory(floats, view(reinterpret(Int, floats), 1:1), 1, 1, LoneLane())
        @test GroupMemory(floats, cursor, 1, 1, LoneLane()).cursor === cursor
        @test GroupMemory(floats, cursor, 2, 2, GroupBarrier()).lane == 2
    end

    @testset "a GroupBarrier throws when it synchronizes outside a kernel" begin
        @test_throws "@synchronize used outside kernel" PathFollowing.synchronize_lanes(GroupBarrier())
    end

    @testset "a CPU-backend kernel of one-lane workgroups traces each start in group memory as a Vector start traces" begin
        backend = KernelAbstractions.CPU()
        for (system, start, direction, overrides, bound, limit, _) in parity_cases(E)
            # Four starts a little apart, one per workgroup.
            starts = reduce(hcat, (start .+ E(shift) .* direction for shift in (0, 1 // 100, 2 // 100, 3 // 100)))
            directions = repeat(direction, 1, 4)
            N = length(start)
            settings = algorithm(E; merge(overrides, (; metric = FixedWeights(SVector{N, E}(ntuple(_ -> one(E), N))), linear_solver = GroupLU()))...)
            @test isbits(settings)
            points, refusals = fill(E(NaN), N, limit, 4), fill(-2, limit, 4)
            group_trace_kernel!(backend, 1)(points, refusals, starts, directions, system, settings, E(1) / 4, bound, limit, LoneLane(); ndrange = 4)
            KernelAbstractions.synchronize(backend)
            for trace in 1:4
                host_points, host_refusals = recorded_trace(system, starts[:, trace], directions[:, trace], algorithm(E; overrides...), E(1) / 4, bound, limit)
                @test refusals[:, trace] == host_refusals
                taken = host_refusals .!= -2
                @test points[:, taken, trace] ≈ host_points[:, taken] atol = 10 * tolerance nans = true
            end
        end
    end

    @testset "StaticLU takes a static start of at most 14 coordinates, and a mutable start takes LinearSolve" begin
        static_start = (SVector{2, E}(1, 0), SVector{2, E}(0, 1))
        @test_throws ArgumentError init(CurveProblem(Circle(one(E)), static_start...), algorithm(E); first_step = E(1) / 10)
        @test_throws ArgumentError init(CurveProblem(Circle(one(E)), E[1, 0], E[0, 1]), algorithm(E; linear_solver = StaticLU()); first_step = E(1) / 10)
        wide = (SVector{15, E}(ntuple(index -> E(index == 1), 15)), SVector{15, E}(ntuple(index -> E(index == 2), 15)))
        @test_throws ArgumentError init(CurveProblem(Circle(one(E)), wide...), algorithm(E; linear_solver = StaticLU(), metric = FixedWeights(ones(E, 15))); first_step = E(1) / 10)
        # A system that answers a static tracer with heap arrays would change the tracer's type.
        heap_answer = (residual, jacobian, _) -> (collect(residual), collect(jacobian))
        @test_throws ArgumentError init(CurveProblem(heap_answer, static_start...), algorithm(E; linear_solver = StaticLU()); first_step = E(1) / 10)
    end

    @testset "static settings are isbits with any target dropped, so a GPU kernel can take them" begin
        check = HalfStepCheck{E}(; trigger = 13 // 10, tolerance = one(E) / 100)
        for targets in ((one(E) / 4, one(E) / 10, one(E) / 10), (nothing, nothing, one(E) / 10), (one(E) / 4, nothing, nothing)), split_check in (nothing, check)
            (target_contraction, target_correction, target_turn) = targets
            step_control = StepControl(; target_contraction, target_correction, target_turn, max_step_growth = E(2), max_step = one(E) / 2)
            settings = algorithm(E; step_control, split_check, metric = FixedWeights(SVector{2, E}(1, 1)), linear_solver = StaticLU())
            @test isbits(step_control) && isbits(settings)
            @test (step_control.target_contraction, step_control.target_correction, step_control.target_turn) === targets
        end
    end

    @testset "land!! steps to the bound and returns a copy of its point; on the bound it lands at once" begin
        bound = Bound(2, E(3) / 4)
        for (start, direction, settings) in ((E[1, 0], E[0, 1], algorithm(E)), (SVector{2, E}(1, 0), SVector{2, E}(0, 1), static_algorithm(E)))
            tracer = unwrap(init(CurveProblem(Circle(one(E)), start, direction), settings; first_step = E(1) / 10))
            tracer, landed = land!!(tracer, bound, 20)
            point = unwrap(landed)
            @test point == tracer.point && point[2] == bound.value
            @test point[1] ≈ sqrt(1 - bound.value^2) atol = 100 * tolerance
            start isa SVector || @test point !== tracer.point
            steps = tracer.steps
            tracer, again = land!!(tracer, bound, steps)
            @test unwrap(again) == point && tracer.steps == steps
            # The returned copy of a mutable point allocates; a static landing allocates nothing.
            start isa SVector && @test landing_bytes(tracer, Bound(2, E(4) / 5), 40) == 0
        end
    end

    @testset "land!! fails TooManySteps past its budget, and a failed step with that step's StepFailure" begin
        bound = Bound(2, E(3) / 4)
        tracer = unwrap(init(CurveProblem(Circle(one(E)), E[1, 0], E[0, 1]), algorithm(E); first_step = E(1) / 10))
        tracer, landed = land!!(tracer, bound, 2)
        @test tracer.steps == 2 && unwrap_error(landed) == StepFailure.TooManySteps{E}()
        edge = unwrap(init(CurveProblem(Edge(one(E), true), E[0, 0], E[1, 1]), algorithm(E); first_step = E(1) / 4))
        _, landed = land!!(edge, Bound(2, E(2)), 200)
        @test unwrap_error(landed).last_refusal == SingularSystem
        # No budget by default: the trace ends on the bound or on a failed step.
        tracer = unwrap(init(CurveProblem(Circle(one(E)), E[1, 0], E[0, 1]), algorithm(E); first_step = E(1) / 10))
        tracer, landed = land!!(tracer, bound)
        @test unwrap(landed)[2] == bound.value && tracer.steps > 2
        _, landed = land!!(unwrap(init(CurveProblem(Edge(one(E), true), E[0, 0], E[1, 1]), algorithm(E); first_step = E(1) / 4)), Bound(2, E(2)))
        @test unwrap_error(landed).last_refusal == SingularSystem
    end

    @testset "land_batch traces each problem as land!! does, one work item each, on Vector and JLArray storage" begin
        settings = static_algorithm(E)
        bound = Bound(2, E(3) / 4)
        # Radius 1/2 never reaches the bound and runs out of steps; a NaN radius has no finite start.
        radii = E[1, 2, 1 // 2, 3, NaN]
        problems = [CurveProblem(Circle(radius), SVector{2, E}(radius, 0), SVector{2, E}(0, 1)) for radius in radii]
        first_steps = E[1 // 10, 1 // 5, 1 // 10, 1 // 4, 1 // 10]
        expected = [PathFollowing.landed(problem, settings, first_step, bound, 40) for (problem, first_step) in zip(problems, first_steps)]
        outcomes = landing_result.(expected)
        @test !any(is_error, outcomes[[1, 2, 4]])
        @test unwrap_error(outcomes[3]) == StepFailure.TooManySteps{E}() && unwrap_error(outcomes[5]) == StepFailure.NotFiniteStart{E}()
        # A failure at the start holds no step; one past the budget holds its tracer's.
        @test (expected[5].step_length, expected[5].steps) == (0, 0) && expected[3].step_length > 0
        @test all(landing -> landing.point[2] == bound.value, expected[[1, 2, 4]]) && expected[3].steps == 40
        tracer = unwrap(init(problems[2], settings; first_step = first_steps[2]))
        @test unwrap(landing_result(expected[2])) == unwrap(last(land!!(tracer, bound, 40)))
        @test unwrap_error(landing_result(expected[5])) == unwrap_error(init(problems[5], settings; first_step = first_steps[5]))
        landings = land_batch(problems, settings, first_steps, bound, 40; workgroupsize = 2)
        @test landings isa Vector{eltype(expected)} && isequal(landings, expected)
        # JLArrays 0.3 defines no `synchronize`, so land_batch refuses it; its kernel launch returns once it has finished.
        @test_throws "defines no KernelAbstractions.synchronize" land_batch(JLArray(problems), settings, JLArray(first_steps), bound, 40; workgroupsize = 2)
        device_landings = similar(JLArray(problems), eltype(expected))
        PathFollowing.land_kernel!(JLBackend(), 2)(device_landings, JLArray(problems), settings, JLArray(first_steps), bound, 40; ndrange = length(problems))
        @test isequal(collect(device_landings), expected)
        # KernelAbstractions 0.9 drops the kernel's source lines. Once this fails, delete the COV_EXCL markers in landing.jl.
        @test only(methods(PathFollowing.cpu_land_kernel!)).line == 0
    end

    @testset "land_batch aborts loud when a callback throws in a work item past the host's check" begin
        # The host's `init` sees radius 1; the work item for radius 3 throws.
        problems = [CurveProblem(BrittleCircle(radius), SVector{2, E}(radius, 0), SVector{2, E}(0, 1)) for radius in E[1, 3]]
        thrown = try
            land_batch(problems, static_algorithm(E), E[1 // 10, 1 // 10], Bound(2, E(3) / 4), 40; workgroupsize = 1)
        catch caught
            caught
        end
        @test occursin("BrittleCircle past radius 2", sprint(showerror, thrown))
    end

    @testset "land_batch checks on the host every input that init or a work item would throw on" begin
        circle = CurveProblem(Circle(one(E)), SVector{2, E}(1, 0), SVector{2, E}(0, 1))
        bound = Bound(2, E(3) / 4)
        for storage in (identity, JLArray)
            checked(settings, first_steps; bound = bound, workgroupsize = 1, problems = storage([circle, circle])) =
                PathFollowing.check_batch(problems, settings, storage(first_steps), bound, workgroupsize)
            @test_throws DimensionMismatch checked(static_algorithm(E), E[1 // 10])
            @test_throws "a static start needs linear_solver = StaticLU()" checked(static_algorithm(E; linear_solver = LUFactorization()), E[1 // 10, 1 // 10])
            # A `Vector` of metric weights keeps the algorithm off the device.
            @test_throws "an isbits algorithm" checked(algorithm(E; linear_solver = StaticLU()), E[1 // 10, 1 // 10])
            @test_throws "workgroupsize must be positive" checked(static_algorithm(E), E[1 // 10, 1 // 10]; workgroupsize = 0)
            @test_throws "the bound's coordinate" checked(static_algorithm(E), E[1 // 10, 1 // 10]; bound = Bound(3, E(3) / 4))
            @test_throws "every first step" checked(static_algorithm(E), E[1 // 10, 0])
            @test_throws "every first step" checked(static_algorithm(E), E[NaN, 1 // 10])
            other = E === Float32 ? Float64 : Float32
            @test_throws "element type must be the start's" checked(static_algorithm(other; metric = FixedWeights(SVector{2, E}(1, 1))), E[1 // 10, 1 // 10])
            wide = CurveProblem(Circle(one(E)), SVector{3, E}(1, 0, 0), SVector{3, E}(0, 1, 0))
            @test_throws DimensionMismatch checked(static_algorithm(E), E[1 // 10, 1 // 10]; problems = storage([wide, wide]))
            fifteen = CurveProblem(Circle(one(E)), SVector{15, E}(ntuple(index -> E(index == 1), 15)), SVector{15, E}(ntuple(index -> E(index == 2), 15)))
            fifteen_weights = FixedWeights(SVector{15, E}(ntuple(_ -> one(E), 15)))
            @test_throws "at most 14 × 14" checked(static_algorithm(E; metric = fifteen_weights), E[1 // 10, 1 // 10]; problems = storage([fifteen, fifteen]))
            @test isempty(land_batch(storage(typeof(circle)[]), static_algorithm(E), storage(E[]), bound, 20; workgroupsize = 1))
            @test_throws ArgumentError land_batch(storage(typeof(circle)[]), static_algorithm(E), storage(E[]), bound, -1; workgroupsize = 1)
        end
        @test isnothing(PathFollowing.check_batch([circle, circle], static_algorithm(E), E[1 // 10, 1 // 5], bound, 1))
        @test_throws "defines no KernelAbstractions.synchronize" PathFollowing.check_batch(JLArray([circle]), static_algorithm(E), JLArray(E[1 // 10]), bound, 1)
        @test_throws "the problems' device" PathFollowing.check_batch(JLArray([circle]), static_algorithm(E), E[1 // 10], bound, 1)
        # A work item would read coordinate 3 of a 2-coordinate point and throw a BoundsError.
        @test_throws ArgumentError land_batch([circle], static_algorithm(E), E[1 // 10], Bound(3, E(3) / 4), 20; workgroupsize = 1)
    end

    @testset "metric_apply!! returns the array it wrote on every storage, whatever a `.=` expression's value" begin
        # Julia 1.14 evaluates `Into(out) .= x` to the wrapper, so a `!!` function that returned it threw.
        weights, vector, static, device = E[2, 3], E[1, 2], SVector{2, E}(1, 2), JLArray(E[1, 2])
        out, device_out = zeros(E, 2), JLArray(zeros(E, 2))
        @test metric_apply!!(out, frozen_metric(FixedWeights(weights), vector), vector) === out == E[2, 6]
        @test metric_apply!!(zero(static), frozen_metric(FixedWeights(weights), static), static) === SVector{2, E}(2, 6)
        @test metric_apply!!(device_out, frozen_metric(FixedWeights(weights), device), device) === device_out
        @test Array(device_out) == E[2, 6]
    end

    @testset "@into!! writes a mutable array in place and rebuilds a static one; an eager source throws" begin
        vector = E[1, 2]
        @test @into!!(vector .= 2 .* vector) === vector == E[2, 4]
        # A fused broadcast and a copy build no temporary.
        @test into_bytes(vector, E[3, 4]) == (0, 0)
        # An eager broadcast reaches `into!!` as a built array, and a number as a value: neither is a broadcast.
        @test_throws MethodError PathFollowing.into!!(vector, 2 .* vector)
        @test_throws MethodError PathFollowing.into!!(vector, zero(E))
        @test_throws "@into!! takes `destination .= expression`" macroexpand(@__MODULE__, :(@into!! vector = 2 .* vector))
        # A dotted call reaches `lazy_broadcast` bare, so it fuses as written; any other source takes `identity.`.
        lowers_through_identity(expression) = occursin("identity", string(macroexpand(@__MODULE__, :(@into!! vector .= $expression))))
        @test !any(lowers_through_identity, (:(2 .* vector), :(.-vector), :(abs.(vector)), :(vector .+ 1 .- vector)))
        @test all(lowers_through_identity, (:vector, :(zero(E)), :(tracer.work.point), :(transpose(vector)), :(a .. b)))
        # Base's `x[2:2] .= 0` writes `x` through a view; `@into!!` refuses an indexed destination, and takes a view.
        @test_throws "not an indexed Vector{$E}; write `@into!! (@view x[i]) .= ...`" @into!! vector[2:2] .= 0
        # The refusal comes before any index runs, and `end` keeps its meaning inside the brackets.
        indices = [1]
        @test_throws "not an indexed Vector{$E}" @into!! vector[pop!(indices)] .= 0
        @test_throws "not an indexed Vector{$E}" @into!! vector[1:end] .= 0
        @test indices == [1] && vector == E[3, 4]
        @test @into!!((@view vector[2:end]) .= 0) == E[0] && vector == E[3, 0]
        # A number is no array to write into, as with Base's `.=`.
        @test_throws MethodError @into!! zero(E) .= 1
        static = SVector{2, E}(1, 2)
        @test @into!!(static .= 2 .* static) === SVector{2, E}(2, 4)
        @test static == SVector{2, E}(1, 2)
        # A scalar or a singleton fills every entry, on static and mutable arrays alike.
        @test @into!!(static .= zero(E)) === SVector{2, E}(0, 0)
        @test @into!!(static .= SVector{1, E}(3)) === SVector{2, E}(3, 3)
        @test @into!!(SMatrix{1, 2, E}(1, 2) .= 2 .* SMatrix{1, 2, E}(3, 4)) === SMatrix{1, 2, E}(6, 8)
        @test @into!!(SMatrix{2, 2, E}(1, 2, 3, 4) .= SVector{2, E}(5, 6)) === SMatrix{2, 2, E}(5, 6, 5, 6)
        @test @into!!(E[1, 2] .= zero(E)) == E[0, 0]
        @test_throws DimensionMismatch @into!! static .= SVector{3, E}(1, 2, 3)
        @test_throws DimensionMismatch @into!! E[1, 2] .= E[1, 2, 3]
        device = JLArray(E[1, 2])
        @test @into!!(device .= 2 .* device) === device
        @test Array(device) == E[2, 4]
        @test PathFollowing.unit!!(SVector{3, E}(5, 5, 5), 2) === SVector{3, E}(0, 1, 0)
        @test PathFollowing.set_element!!(SVector{2, E}(1, 2), 1, E(7)) === SVector{2, E}(7, 2)
    end

    @testset "the static orientation sign is the sign of the determinant, its row swaps counted" begin
        @test !PathFollowing.odd_permutation(SVector(1, 2, 3)) && !PathFollowing.odd_permutation(SVector(2, 3, 1))
        @test PathFollowing.odd_permutation(SVector(2, 1, 3)) && PathFollowing.odd_permutation(SVector(3, 2, 1))
        for matrix in (SMatrix{3, 3, E}(0, 1, 0, 1, 0, 0, 0, 0, 1), SMatrix{3, 3, E}(0, 0, 1, 1, 0, 0, 0, 1, 0), SMatrix{3, 3, E}(2, 1, 1, 1, -3, 1, 4, 1, 5))
            factorization = LinearAlgebra.lu(matrix, LinearAlgebra.RowMaximum(); check = false)
            @test factored_sign(factorization) == sign(det(matrix))
        end
        @test factored_sign(LinearAlgebra.lu(SMatrix{2, 2, E}(1, 0, 0, NaN32), LinearAlgebra.RowMaximum(); check = false)) == 0
    end

    @testset "property: random ellipses keep every point on the curve" begin
        ratio = Data.Floats{E}(; minimum = E(1) / 4, maximum = E(4), nans = false, infs = false)
        @check max_examples = 30 function on_ellipse(stretch = ratio)
            ellipse!! = (residual, jacobian, point) -> one_row!!(residual, jacobian, point[1]^2 + (point[2] / stretch)^2 - 1, (2 * point[1], 2 * point[2] / stretch^2))
            points = traced(CurveProblem(ellipse!!, E[1, 0], E[0, 1]), algorithm(E), 20; first_step = E(1) / 10)
            points isa Vector && all(point -> abs(point[1]^2 + (point[2] / stretch)^2 - 1) <= 100 * tolerance, points)
        end
    end

    @testset "a metric sum keeps small terms that large ones cancel around, on host, static and group storage" begin
        # Fourteen terms each under half an ulp of one: a plain fold from zero loses them all.
        small = eps(E) / 3
        terms = [one(E); fill(small, 14); -one(E)]
        for storage in (terms, SVector{16, E}(terms), group_vector(host_memory(E), terms))
            @test PathFollowing.elementwise_sum(E, identity, storage) ≈ 14 * big(small) rtol = 8 * eps(E)
        end
        # GPU arrays take one GPU reduction, with no compensation.
        @test PathFollowing.elementwise_sum(E, *, JLArray(E[1, 2, 3]), JLArray(E[4, 5, 6])) == 32
        # Float64 weights on points in E sum in their promoted type; an overflow stays infinite, and a NaN stays NaN.
        @test metric_inner(FixedWeights([1.0, 1.0]), E[1, 2], E[1, 2]) === E(5)
        @test metric_inner(FixedWeights(ones(E, 1)), [floatmax(E)], [floatmax(E)]) === E(Inf)
        @test isnan(PathFollowing.elementwise_sum(E, identity, E[1, NaN, 2])) && isnan(PathFollowing.elementwise_sum(E, identity, E[Inf, -Inf]))
    end
end

@testset "CUDA parity shards split the configurations and refuse a shard past them" begin
    @test chosen_configs(String[], 12) == 1:12 && isempty(chosen_configs(["timing"], 12))
    @test sort(reduce(vcat, [collect(chosen_configs(["parity", string(shard), "5"], 12)) for shard in 1:5])) == 1:12
    @test_throws "holds no configuration" chosen_configs(["parity", "13", "13"], 12)
    @test_throws "holds no configuration" chosen_configs(["parity", "0", "3"], 12)
    @test_throws "`parity SHARD SHARDS`" chosen_configs(["parity", "1"], 12)
end

@testset "the CUDA tests' IR check names a call cycle through quoted or bare names, and passes a call chain" begin
    definition(name::AbstractString, callees::AbstractString...)::String =
        "define internal void @$name(ptr %a) {\n" * join("  call void @$callee(ptr %a)\n" for callee in callees) * "  ret void\n}\n"
    quoted = "\"julia_#impl_1\""
    @test call_cycle(definition("kernel", quoted) * definition(quoted, quoted)) == [quoted, quoted]
    @test call_cycle(definition("kernel", "a") * definition("a", "b", "llvm.nvvm.barrier0") * definition("b", "a")) in (["a", "b", "a"], ["b", "a", "b"])
    @test call_cycle(definition("kernel", "a", "b") * definition("a", "b") * definition("b", "llvm.nvvm.barrier0")) === nothing
    @test call_cycle(definition("kernel", quoted) * definition(quoted)) === nothing
end

chosen(BigFloat) && @testset "PathFollowing BigFloat: the orientation check runs on a generic LU" begin
    for linear_solver in (LUFactorization(), QRFactorization())
        settings = algorithm(BigFloat; linear_solver)
        tracer = unwrap(init(CurveProblem(Pitchfork(zero(BigFloat)), BigFloat[0, -1], BigFloat[0, 1]), settings; first_step = big(3) / 10))
        steps = AcceptedStep{BigFloat}[]
        for _ in 1:6
            tracer, step = accepted(tracer)
            push!(steps, step)
        end
        # Steps across the bifurcation point flip the sign and are refused.
        @test sum(step -> step.refusals, steps) > 0
        @test tracer.orientation == 1 && tracer.point[2] < 0
        @test all(iszero, tracer.linear.oriented) == (linear_solver isa LUFactorization)
    end
end
