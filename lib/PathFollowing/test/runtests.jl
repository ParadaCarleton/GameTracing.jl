using PathFollowing
using PathFollowing: AcceptAll, factored_sign, metric_angle
using CommonSolve: init, step!
using ErrorTypes: is_error, unwrap, unwrap_error
using LinearAlgebra: det
using LinearSolve: LUFactorization, QRFactorization
using Supposition: Data, @check
using Test: @test, @testset, @test_throws

"""A circle of `radius` about the origin in `(x, y)`: `H = x² + y² - radius²`."""
struct Circle{E <: Real}
    radius::E
end

function (circle::Circle{E})(residual::AbstractVector{E}, jacobian::AbstractMatrix{E}, point::AbstractVector{E})::Nothing where {E <: Real}
    residual[1] = point[1]^2 + point[2]^2 - circle.radius^2
    jacobian[1, 1] = 2 * point[1]
    jacobian[1, 2] = 2 * point[2]
    return nothing
end

"""The line `x = y`: `H = x - y`."""
struct Line end

function (::Line)(residual::AbstractVector{E}, jacobian::AbstractMatrix{E}, point::AbstractVector{E})::Nothing where {E <: Real}
    residual[1] = point[1] - point[2]
    jacobian[1, 1] = one(E)
    jacobian[1, 2] = -one(E)
    return nothing
end

"""The curve `λ = u - u³` in `(u, λ)`: λ turns at a minimum, `u = -1/√3`, then at a maximum, `u = 1/√3`."""
struct Wiggle end

function (::Wiggle)(residual::AbstractVector{E}, jacobian::AbstractMatrix{E}, point::AbstractVector{E})::Nothing where {E <: Real}
    residual[1] = point[2] - point[1] + point[1]^3
    jacobian[1, 1] = 3 * point[1]^2 - 1
    jacobian[1, 2] = one(E)
    return nothing
end

"""The fold `H = u² - λ` in `(u, λ)`: λ turns back at `u = 0`."""
struct Fold end

function (::Fold)(residual::AbstractVector{E}, jacobian::AbstractMatrix{E}, point::AbstractVector{E})::Nothing where {E <: Real}
    residual[1] = point[1]^2 - point[2]
    jacobian[1, 1] = 2 * point[1]
    jacobian[1, 2] = -one(E)
    return nothing
end

"""
A pitchfork `H = u³ - λ u - tilt` in `(u, λ)`. Untilted, the branch `u = 0` crosses a simple bifurcation point at the
origin; tilted, its upper branch passes close to the lower ones near `λ = 0`.
"""
struct Pitchfork{E <: Real}
    tilt::E
end

function (pitchfork::Pitchfork{E})(residual::AbstractVector{E}, jacobian::AbstractMatrix{E}, point::AbstractVector{E})::Nothing where {E <: Real}
    (u, lambda) = (point[1], point[2])
    residual[1] = u^3 - lambda * u - pitchfork.tilt
    jacobian[1, 1] = 3 * u^2 - lambda
    jacobian[1, 2] = -u
    return nothing
end

"""`H = u - λ` up to `λ = edge`; past it a NaN residual, or a zero Jacobian when `singular`."""
struct Edge{E <: Real}
    edge::E
    singular::Bool
end

function (edge::Edge{E})(residual::AbstractVector{E}, jacobian::AbstractMatrix{E}, point::AbstractVector{E})::Nothing where {E <: Real}
    past = point[2] > edge.edge
    residual[1] = past && !edge.singular ? E(NaN32) : point[1] - point[2]
    jacobian[1, 1] = past && edge.singular ? zero(E) : one(E)
    jacobian[1, 2] = past && edge.singular ? zero(E) : -one(E)
    return nothing
end

"""`H = u - λ`, whose `bad_call`-th evaluation is NaN, or has a zero Jacobian when `singular`."""
mutable struct Flaky
    calls::Int
    const bad_call::Int
    const singular::Bool
end

function (flaky::Flaky)(residual::AbstractVector{E}, jacobian::AbstractMatrix{E}, point::AbstractVector{E})::Nothing where {E <: Real}
    flaky.calls += 1
    bad = flaky.calls == flaky.bad_call
    residual[1] = bad && !flaky.singular ? E(NaN32) : point[1] - point[2]
    jacobian[1, 1] = bad && flaky.singular ? zero(E) : one(E)
    jacobian[1, 2] = bad && flaky.singular ? zero(E) : -one(E)
    return nothing
end

"""A `converged` that never holds."""
never(::AbstractVector, ::AbstractVector, ::AbstractVector, ::Real)::Bool = false

"""The bytes a warm `step!` of `tracer` toward `bound` allocates."""
function warm_bytes(tracer::Tracer, bound::Union{Nothing, Bound})::Int
    step!(tracer, bound)
    return @allocated step!(tracer, bound)
end

"""Converged when the correction, times `scale`, moved no coordinate more than `tolerance`."""
struct Settled{E <: Real}
    tolerance::E
end

function (settled::Settled{E})(previous::AbstractVector{E}, iterate::AbstractVector{E}, ::AbstractVector{E}, scale::E)::Bool where {E <: Real}
    largest = zero(E)
    for index in eachindex(previous, iterate)
        largest = max(largest, abs(iterate[index] - previous[index]))
    end
    return scale * largest <= settled.tolerance
end

"""
A metric that counts its freezes in `freezes` and weighs each coordinate by `1 + |xᵢ|` at the point `x` it is frozen at:
a diagonal stand-in for a metric that depends on the point.
"""
struct CountingMetric <: PathFollowing.Metric
    freezes::Base.RefValue{Int}
end

"""`CountingMetric`'s state: its counter and its weights at the point it is frozen at."""
struct CountingFrozen{E <: Real}
    freezes::Base.RefValue{Int}
    weights::Vector{E}
end

function PathFollowing.frozen_metric(metric::CountingMetric, point::AbstractVector{E})::CountingFrozen{E} where {E <: Real}
    return freeze_metric!(CountingFrozen(metric.freezes, zeros(E, length(point))), point)
end

function PathFollowing.freeze_metric!(frozen::CountingFrozen{E}, point::AbstractVector{E})::CountingFrozen{E} where {E <: Real}
    frozen.freezes[] += 1
    frozen.weights .= 1 .+ abs.(point)
    return frozen
end

function PathFollowing.metric_inner(frozen::CountingFrozen{E}, left::AbstractVector{E}, right::AbstractVector{E})::E where {E <: Real}
    return sum(frozen.weights .* left .* right)
end

function PathFollowing.metric_apply!(out::AbstractVector{E}, frozen::CountingFrozen{E}, vector::AbstractVector{E})::AbstractVector{E} where {E <: Real}
    out .= frozen.weights .* vector
    return out
end

"""A `PseudoArclength` in `E` with Euclidean metrics and the given overrides."""
function algorithm(::Type{E}; tolerance::Real = sqrt(eps(E)), overrides...)::PseudoArclength where {E <: Real}
    defaults = (;
        corrector = Newton(; max_iterations = 8, converged = Settled(E(tolerance)), contraction_limit = one(E) / 2, error_estimate = LastCorrection()),
        step_control = StepControl(;
            target_contraction = one(E) / 4, target_correction = one(E) / 10, target_turn = one(E) / 10,
            max_step_growth = E(2), max_step = one(E) / 2,
        ),
        metric = FixedWeights(ones(E, 2)),
        orientation_check = DeterminantSign(),
        linear_solver = LUFactorization(),
        max_refusals = 60,
    )
    return PseudoArclength(; merge(defaults, overrides)...)
end

"""Accepted points of `steps` steps of `problem` under `settings`, or the failure."""
function traced(problem::CurveProblem, settings::PseudoArclength, steps::Integer; first_step::Real)::Any
    started = init(problem, settings; first_step)
    is_error(started) && return started
    tracer = unwrap(started)
    points = [copy(tracer.point)]
    for _ in 1:steps
        outcome = step!(tracer)
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

"""The accepted steps of `tracer` toward `bound` until its point lies on it, at most `limit`; a failed step throws."""
function bounded_steps(tracer::Tracer{E}, bound::Bound{E}, limit::Integer)::Vector{AcceptedStep{E}} where {E <: Real}
    steps = AcceptedStep{E}[]
    while tracer.point[bound.coordinate] != bound.value && length(steps) < limit
        push!(steps, unwrap(step!(tracer, bound)))
    end
    return steps
end

@testset "PathFollowing $E" for E in (Float32, Float64)
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
            outcome = step!(tracer)
            if is_error(outcome)
                failure = unwrap_error(outcome)
                break
            end
            push!(lambdas, tracer.point[2])
        end
        @test failure.cause == Unresolved
        @test all(<(0), lambdas) && issorted(lambdas)
        @test tracer.orientation == 1 && tracer.point[1] == 0
    end

    @testset "the orientation sign is the sign of det([J; tᵀ]), from the solve's LU or a second factorization" begin
        for linear_solver in (LUFactorization(), QRFactorization()), (start, expected) in ((E[0, -1], 1), (E[0, 1], -1))
            # The branch u = 0 changes sign at the bifurcation point λ = 0, so starts on either side differ.
            settings = algorithm(E; linear_solver)
            tracer = unwrap(init(CurveProblem(Pitchfork(zero(E)), start, E[0, 1]), settings; first_step = E(3) / 10))
            @test tracer.orientation == expected == sign(det([tracer.jacobian; tracer.tangent']))
            for _ in 1:6
                unwrap(step!(tracer))
                @test tracer.orientation == expected == sign(det([tracer.jacobian; tracer.tangent']))
            end
            # Only the fallback fills the matrix of the second factorization.
            @test all(iszero, tracer.oriented) == (linear_solver isa LUFactorization)
        end
    end

    @testset "without an orientation check, a bifurcation crossing goes unremarked" begin
        settings = algorithm(E; orientation_check = nothing)
        tracer = unwrap(init(CurveProblem(Pitchfork(zero(E)), E[0, -1], E[0, 1]), settings; first_step = E(3) / 10))
        steps = [unwrap(step!(tracer)) for _ in 1:6]
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
            step = unwrap(step!(tracer))
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
            unwrap(step!(tracer))
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
        @test metric_apply!(zeros(E, 2), frozen, E[1, 2]) == E[2, 6]
        @test_throws DimensionMismatch frozen_metric(FixedWeights(ones(E, 3)), E[1, 1])
    end

    @testset "a refusing accept ends the trace Unresolved" begin
        settings = algorithm(E; accept = (start, predicted, corrected) -> false)
        failure = unwrap_error(traced(CurveProblem(Fold(), E[-1, 1], E[1, -1]), settings, 1; first_step = E(1) / 10))
        @test failure.cause == Unresolved
        @test failure.last_refusal == AcceptRefused
    end

    @testset "max_refusals ends the trace TooManyRefusals" begin
        settings = algorithm(E; accept = (start, predicted, corrected) -> false, max_refusals = 3)
        failure = unwrap_error(traced(CurveProblem(Fold(), E[-1, 1], E[1, -1]), settings, 1; first_step = E(1) / 10))
        @test failure.cause == TooManyRefusals
        @test failure.refusals == 4
    end

    @testset "a NaN residual refuses the step until the trace is unresolved" begin
        failure = unwrap_error(traced(CurveProblem(Edge(one(E), false), E[0, 0], E[1, 1]), algorithm(E), 200; first_step = E(1) / 4))
        @test failure.cause == Unresolved
        @test failure.last_refusal == NotFiniteResidual
    end

    @testset "a singular bordered system refuses the step" begin
        failure = unwrap_error(traced(CurveProblem(Edge(one(E), true), E[0, 0], E[1, 1]), algorithm(E), 200; first_step = E(1) / 4))
        @test failure.cause == Unresolved
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
        settings = algorithm(E; corrector = Newton(; max_iterations = 1, converged = never, contraction_limit = one(E), error_estimate = LastCorrection()), max_refusals = 0)
        failure = unwrap_error(traced(CurveProblem(Circle(one(E)), E[1, 0], E[0, 1]), settings, 1; first_step = E(1) / 10))
        @test failure.last_refusal == NotConverged
    end

    @testset "a tiny contraction limit refuses as diverging" begin
        settings = algorithm(E; corrector = Newton(; max_iterations = 8, converged = never, contraction_limit = eps(E), error_estimate = LastCorrection()), max_refusals = 0)
        failure = unwrap_error(traced(CurveProblem(Circle(one(E)), E[1, 0], E[0, 1]), settings, 1; first_step = E(1) / 10))
        @test failure.last_refusal == Diverging
    end

    @testset "a tight turn limit refuses a long step as too fast" begin
        control = StepControl(;
            target_contraction = nothing, target_correction = nothing, target_turn = one(E) / 1000, max_step_growth = E(2),
            max_step = E(1),
        )
        tracer = unwrap(init(CurveProblem(Circle(one(E)), E[1, 0], E[0, 1]), algorithm(E; step_control = control); first_step = one(E)))
        step = unwrap(step!(tracer))
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
        step = unwrap(step!(tracer))
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
        unwrap(step!(tracer))
        @test tracer.step_length == E(2) / 1000
    end

    @testset "ContractionEstimate settles on the error left in the iterate, in fewer iterations" begin
        halved = ContractionEstimate(; safety = one(E) / 2)
        @test PathFollowing.error_scale(LastCorrection(), E(1) / 4, one(E)) == 1
        @test PathFollowing.error_scale(halved, E(1) / 4, E(NaN32)) == 1
        @test PathFollowing.error_scale(halved, E(1) / 4, one(E)) ≈ E(2) / 3
        @test PathFollowing.error_scale(halved, one(E), one(E)) == Inf
        @test_throws ArgumentError ContractionEstimate(; safety = zero(E))
        @test_throws ArgumentError ContractionEstimate(; safety = E(2))
        @test_throws TypeError Newton(; max_iterations = 1, converged = Settled(tolerance), contraction_limit = one(E), error_estimate = ContractionEstimate(; safety = big(1)))
        # A half step up the unit circle corrects by 0.125, 8.9e-3 and 4.6e-5. At tolerance 2e-3, θ/(1-θ) times the
        # second correction, 6.9e-4, settles; over safety 1/10 it does not, nor does the second correction itself.
        settle = E(1) / 500
        control = StepControl(;
            target_contraction = nothing, target_correction = nothing, target_turn = nothing, max_step_growth = E(2),
            max_step = E(1),
        )
        estimates = (LastCorrection(), ContractionEstimate(; safety = one(E) / 10), ContractionEstimate(; safety = one(E)))
        iterations = map(estimates) do error_estimate
            corrector = Newton(; max_iterations = 8, converged = Settled(settle), contraction_limit = one(E) / 2, error_estimate)
            tracer = unwrap(init(CurveProblem(Circle(one(E)), E[1, 0], E[0, 1]), algorithm(E; corrector, step_control = control); first_step = E(1) / 2))
            step = unwrap(step!(tracer))
            @test abs(hypot(tracer.point...) - 1) <= settle
            step.iterations
        end
        @test iterations == (3, 3, 2)
    end

    @testset "a custom distance measures corrections" begin
        largest = (from, to) -> maximum(abs.(to .- from))
        settings = algorithm(E; corrector = Newton(; max_iterations = 8, converged = Settled(tolerance), contraction_limit = one(E) / 2, error_estimate = LastCorrection(), distance = largest))
        step = unwrap(unwrap(init(CurveProblem(Circle(one(E)), E[1, 0], E[0, 1]), settings; first_step = E(1) / 10)) |> step!)
        @test step.first_correction > 0
    end

    @testset "a converged point farther than reach predictor lengths from its predictor refuses the step, which retries shorter" begin
        # On the unit circle a step `h` along the tangent corrects by `1 - √(1 - h²)`: 0.45 of 3/4 and 0.19 of 3/8.
        control = StepControl(;
            target_contraction = nothing, target_correction = nothing, target_turn = nothing, max_step_growth = E(2),
            max_step = one(E), reach = one(E) / 4,
        )
        problem = CurveProblem(Circle(one(E)), E[1, 0], E[0, 1])
        step = unwrap(step!(unwrap(init(problem, algorithm(E; step_control = control); first_step = E(3) / 4))))
        @test (step.refusals, step.step_length) == (1, E(3) / 8)
        failure = unwrap_error(step!(unwrap(init(problem, algorithm(E; step_control = control, max_refusals = 0); first_step = E(3) / 4))))
        @test (failure.cause, failure.last_refusal) == (TooManyRefusals, FarFromPredictor)
    end

    @testset "a predictor that reaches the bound is cut there and lands with the value exact" begin
        tracer = unwrap(init(CurveProblem(Circle(one(E)), E[1, 0], E[0, 1]), algorithm(E); first_step = E(1) / 10))
        bound = Bound(2, E(1) / 20)
        accepted = unwrap(step!(tracer, bound))
        @test tracer.point[2] == bound.value
        @test tracer.point[1] ≈ sqrt(1 - bound.value^2) atol = 100 * tolerance
        # The next step grows from the cut predictor, by at most max_step_growth.
        @test accepted.step_length ≈ E(1) / 20 rtol = 10 * eps(E)
        @test tracer.step_length <= 2 * accepted.step_length
        @test_throws ArgumentError step!(tracer, bound)
    end

    @testset "a step short of the bound moves as without one" begin
        free = unwrap(init(CurveProblem(Circle(one(E)), E[1, 0], E[0, 1]), algorithm(E); first_step = E(1) / 10))
        bounded = unwrap(init(CurveProblem(Circle(one(E)), E[1, 0], E[0, 1]), algorithm(E); first_step = E(1) / 10))
        unwrap(step!(free))
        unwrap(step!(bounded, Bound(2, E(1) / 4)))
        @test (bounded.point, bounded.tangent, bounded.step_length) == (free.point, free.tangent, free.step_length)
    end

    @testset "a corrected point past the bound refuses the step as PastBound" begin
        # From (-7/25, -24/25) a step of 4/5 passes y's minimum at (0, -1) to (3/5, -4/5), past y = -9/10.
        tracer = untargeted(Circle(one(E)), E[-7, -24] ./ 25, E[1, 0]; first_step = E(4) / 5, max_refusals = 0)
        failure = unwrap_error(step!(tracer, Bound(2, -E(9) / 10)))
        @test (failure.cause, failure.last_refusal) == (TooManyRefusals, PastBound)
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
        accepted = unwrap(step!(tracer, bound))
        @test accepted.refusals >= 1
        @test accepted.step_length < E(9) / 10
        bounded_steps(tracer, bound, 40)
        @test tracer.point ≈ [sqrt(E(19)) / 10, E(9) / 10] atol = 100 * tolerance
        # Without retries the failure reports the cut predictor it refused.
        settings = algorithm(E; step_control = control, max_refusals = 0)
        tracer = unwrap(init(CurveProblem(Circle(one(E)), E[1, 0], E[0, 1]), settings; first_step = E(100)))
        failure = unwrap_error(step!(tracer, bound))
        @test failure.cause == TooManyRefusals
        @test failure.step_length ≈ E(9) / 10 rtol = 10 * eps(E)
    end

    @testset "a bound one ulp ahead lands: the resolution check judges the step before the cut" begin
        # Up x = y from (1, 1), y = nextfloat(1) is one ulp away. Judging the cut predictor failed this as Unresolved.
        tracer = unwrap(init(CurveProblem(Line(), E[1, 1], E[1, 1]), algorithm(E); first_step = E(1) / 10))
        bound = Bound(2, nextfloat(one(E)))
        unwrap(step!(tracer, bound))
        @test tracer.point[2] == bound.value
        @test tracer.point[1] ≈ bound.value atol = 4 * eps(E)
        # An uncut step ends one ulp short of the bound, and the next step lands on it.
        tracer = unwrap(init(CurveProblem(Line(), E[0, 0], E[1, 1]), algorithm(E); first_step = E(1) / 10))
        @test unwrap(step!(tracer, Bound(2, one(E)))).step_length == E(1) / 10
        bound = Bound(2, nextfloat(tracer.point[2]))
        unwrap(step!(tracer, bound))
        @test tracer.point[2] == bound.value
    end

    @testset "a retry that rounds back to the refused step fails Unresolved" begin
        # A shrink one ulp under 1 cannot shorten the smallest subnormal step: each retry repeated it until TooManyRefusals.
        targets = (; target_contraction = nothing, target_correction = nothing, target_turn = nothing)
        control = StepControl(; targets..., max_step_growth = E(2), max_step = one(E), shrink = prevfloat(one(E)))
        settings = algorithm(E; step_control = control, accept = (start, predicted, corrected) -> false)
        tracer = unwrap(init(CurveProblem(Line(), E[0, 0], E[1, 1]), settings; first_step = nextfloat(zero(E))))
        failure = unwrap_error(step!(tracer))
        @test (failure.cause, failure.last_refusal, failure.refusals) == (Unresolved, AcceptRefused, 1)
        @test failure.step_length == nextfloat(zero(E))
    end

    @testset "a bounded step over a fold's top stops at the first crossing" begin
        # From (3/5, 4/5) a step of 4/5 passes y's maximum at (0, 1) to (-7/25, 24/25), so it meets y = 9/10 twice.
        tracer = untargeted(Circle(one(E)), E[3, 4] ./ 5, E[-1, 0]; first_step = E(4) / 5)
        @test length(bounded_steps(tracer, Bound(2, E(9) / 10), 1)) == 1
        @test tracer.point ≈ [sqrt(E(19)) / 10, E(9) / 10] atol = 100 * tolerance
        # y = 1/2 lies behind the start, so the same step stays above it.
        tracer = untargeted(Circle(one(E)), E[3, 4] ./ 5, E[-1, 0]; first_step = E(4) / 5)
        unwrap(step!(tracer, Bound(2, one(E) / 2)))
        @test tracer.point ≈ E[-7, 24] ./ 25 atol = 100 * tolerance
    end

    @testset "a bounded step through a fold's bottom stops at the first crossing" begin
        # From (-7/25, -24/25) the step passes y's minimum at (0, -1); it meets y = -97/100 at x = ∓0.2431, the first at -.
        tracer = untargeted(Circle(one(E)), E[-7, -24] ./ 25, E[1, 0]; first_step = E(4) / 5)
        @test length(bounded_steps(tracer, Bound(2, -E(97) / 100), 1)) == 1
        @test tracer.point ≈ [-sqrt(1 - E(97)^2 / 100^2), -E(97) / 100] atol = 100 * tolerance
        # y = -9/10 lies past the step's end, and first behind its start: the refused step shrinks and lands ahead.
        tracer = untargeted(Circle(one(E)), E[-7, -24] ./ 25, E[1, 0]; first_step = E(4) / 5)
        steps = bounded_steps(tracer, Bound(2, -E(9) / 10), 4)
        @test first(steps).refusals == 1
        @test tracer.point ≈ [sqrt(E(19)) / 10, -E(9) / 10] atol = 100 * tolerance
    end

    @testset "a bound behind the start stays behind: the trace lands on its next crossing ahead" begin
        # From (1, 0) toward rising y the step of 4/5 ends at (3/5, 4/5); y = -1/10 lies behind, at (√99/10, -1/10).
        tracer = untargeted(Circle(one(E)), E[1, 0], E[0, 1]; first_step = E(4) / 5)
        bound = Bound(2, -E(1) / 10)
        unwrap(step!(tracer, bound))
        @test tracer.point ≈ E[3, 4] ./ 5 atol = 100 * tolerance
        bounded_steps(tracer, bound, 20)
        @test tracer.point ≈ [-sqrt(E(99)) / 10, -E(1) / 10] atol = 100 * tolerance
    end

    @testset "a step over two folds of λ = u - u³ stops at the crossing between them" begin
        # From u = -3/5 a step of 227/200 passes λ's minimum and maximum to u ≈ 3/5, λ ≈ 48/125; λ = 0 lies between.
        tracer = untargeted(Wiggle(), [-E(3) / 5, -E(48) / 125], E[1, 0]; first_step = E(227) / 200, max_refusals = 0)
        failure = unwrap_error(step!(tracer, Bound(2, zero(E))))
        @test (failure.cause, failure.last_refusal) == (TooManyRefusals, PastBound)
        tracer = untargeted(Wiggle(), [-E(3) / 5, -E(48) / 125], E[1, 0]; first_step = E(227) / 200)
        bounded_steps(tracer, Bound(2, zero(E)), 20)
        @test tracer.point ≈ E[0, 0] atol = 100 * tolerance
    end

    @testset "a NaN start fails NotFiniteStart" begin
        failure = unwrap_error(init(CurveProblem(Edge(-one(E), false), E[0, 0], E[1, 1]), algorithm(E); first_step = one(E)))
        @test failure.cause == NotFiniteStart
    end

    @testset "a direction normal to the tangent fails SingularStart" begin
        failure = unwrap_error(init(CurveProblem(Circle(one(E)), E[1, 0], E[1, 0]), algorithm(E); first_step = one(E)))
        @test failure.cause == SingularStart
    end

    @testset "argument checks" begin
        @test_throws ArgumentError Newton(; max_iterations = 0, converged = Settled(tolerance), contraction_limit = one(E), error_estimate = LastCorrection())
        @test_throws ArgumentError Newton(; max_iterations = 1, converged = Settled(tolerance), contraction_limit = zero(E), error_estimate = LastCorrection())
        targets = (; target_contraction = nothing, target_correction = nothing, target_turn = nothing)
        @test_throws ArgumentError StepControl(; targets..., max_step_growth = one(E) / 2, max_step = one(E))
        @test_throws ArgumentError StepControl(; targets..., max_step_growth = one(E), max_step = zero(E))
        @test_throws ArgumentError StepControl(; targets..., max_step_growth = one(E), max_step = one(E), refusal_factor = 0)
        @test_throws ArgumentError StepControl(; targets..., max_step_growth = one(E), max_step = one(E), reach = 0)
        @test_throws ArgumentError StepControl(; targets..., max_step_growth = one(E), max_step = one(E), reach = NaN32)
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
        @test warm_bytes(tracer, nothing) == 0
        tracer = unwrap(init(CurveProblem(Circle(one(E)), E[1, 0], E[0, 1]), algorithm(E); first_step = E(1) / 10))
        bound = Bound(2, E(3) / 20)
        @test warm_bytes(tracer, bound) == 0
        @test tracer.point[2] == bound.value
    end

    @testset "property: random ellipses keep every point on the curve" begin
        ratio = Data.Floats{E}(; minimum = E(1) / 4, maximum = E(4), nans = false, infs = false)
        @check max_examples = 30 function on_ellipse(stretch = ratio)
            ellipse! = (residual, jacobian, point) -> begin
                residual[1] = point[1]^2 + (point[2] / stretch)^2 - 1
                jacobian[1, 1] = 2 * point[1]
                jacobian[1, 2] = 2 * point[2] / stretch^2
                nothing
            end
            points = traced(CurveProblem(ellipse!, E[1, 0], E[0, 1]), algorithm(E), 20; first_step = E(1) / 10)
            points isa Vector && all(point -> abs(point[1]^2 + (point[2] / stretch)^2 - 1) <= 100 * tolerance, points)
        end
    end
end

@testset "PathFollowing BigFloat: the orientation check runs on a generic LU" begin
    for linear_solver in (LUFactorization(), QRFactorization())
        settings = algorithm(BigFloat; linear_solver)
        tracer = unwrap(init(CurveProblem(Pitchfork(zero(BigFloat)), BigFloat[0, -1], BigFloat[0, 1]), settings; first_step = big(3) / 10))
        steps = [unwrap(step!(tracer)) for _ in 1:6]
        # Steps across the bifurcation point flip the sign and are refused.
        @test sum(step -> step.refusals, steps) > 0
        @test tracer.orientation == 1 && tracer.point[2] < 0
        @test all(iszero, tracer.oriented) == (linear_solver isa LUFactorization)
    end
end
