using PathFollowing
using PathFollowing: AcceptAll, factored_sign, weighted_angle
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

"""The bytes a warm `step!` of `tracer` allocates."""
function warm_bytes(tracer::Tracer)::Int
    step!(tracer)
    return @allocated step!(tracer)
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

"""A `PseudoArclength` in `E` with Euclidean metrics and the given overrides."""
function algorithm(::Type{E}; tolerance::Real = sqrt(eps(E)), overrides...)::PseudoArclength where {E <: Real}
    defaults = (;
        corrector = Newton(; max_iterations = 8, converged = Settled(E(tolerance)), contraction_limit = one(E) / 2, error_estimate = LastCorrection()),
        step_control = StepControl(;
            target_contraction = one(E) / 4, target_correction = one(E) / 10, target_turn = one(E) / 10,
            max_step_growth = E(2), max_first_growth = E(2), max_step = one(E) / 2,
        ),
        metric = FixedWeights(ones(E, 2)),
        orientation_check = DeterminantSign(; persistence = 2),
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

    @testset "a bifurcation crossing is refused persistence times, then accepted as flipped" begin
        for persistence in (0, 2)
            settings = algorithm(E; orientation_check = DeterminantSign(; persistence))
            tracer = unwrap(init(CurveProblem(Pitchfork(zero(E)), E[0, -1], E[0, 1]), settings; first_step = E(3) / 10))
            @test tracer.orientation == 1
            steps = [unwrap(step!(tracer)) for _ in 1:12]
            crossing = only(findall(step -> step.flipped, steps))
            @test sum(step -> step.refusals, steps[1:crossing]) >= persistence
            @test persistence > 0 || steps[crossing].refusals == 0
            @test tracer.orientation == -1
            @test tracer.flip_streak == 0
            @test tracer.point[1] == 0
            @test tracer.point[2] > 0
        end
    end

    @testset "the orientation sign is the sign of det([J; tᵀ]), from the solve's LU or a second factorization" begin
        for linear_solver in (LUFactorization(), QRFactorization())
            settings = algorithm(E; orientation_check = DeterminantSign(; persistence = 0), linear_solver)
            tracer = unwrap(init(CurveProblem(Pitchfork(zero(E)), E[0, -1], E[0, 1]), settings; first_step = E(3) / 10))
            signs = [tracer.orientation]
            @test tracer.orientation == sign(det([tracer.jacobian; tracer.tangent']))
            for _ in 1:6
                unwrap(step!(tracer))
                push!(signs, tracer.orientation)
                @test tracer.orientation == sign(det([tracer.jacobian; tracer.tangent']))
            end
            @test signs[1] == 1 && signs[end] == -1
            # Only the fallback fills the matrix of the second factorization.
            @test all(iszero, tracer.oriented) == (linear_solver isa LUFactorization)
        end
    end

    @testset "without an orientation check, a bifurcation crossing goes unremarked" begin
        settings = algorithm(E; orientation_check = nothing)
        tracer = unwrap(init(CurveProblem(Pitchfork(zero(E)), E[0, -1], E[0, 1]), settings; first_step = E(3) / 10))
        steps = [unwrap(step!(tracer)) for _ in 1:6]
        @test !any(step -> step.flipped || step.refusals > 0, steps)
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

    @testset "point weights reweight at every accepted point" begin
        visits = Ref(0)
        scale! = (weights, point) -> (visits[] += 1; fill!(weights, one(eltype(point))))
        settings = algorithm(E; metric = PointWeights(scale!), turn_metric = FixedWeights(ones(E, 2)))
        tracer = unwrap(init(CurveProblem(Circle(one(E)), E[1, 0], E[0, 1]), settings; first_step = E(1) / 10))
        before = visits[]
        unwrap(step!(tracer))
        @test visits[] == before + 1
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
            max_first_growth = E(2), max_step = E(1),
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
            max_first_growth = E(2), max_step = E(1),
        )
        tracer = unwrap(init(CurveProblem(Circle(one(E)), E[1, 0], E[0, 1]), algorithm(E; step_control = control); first_step = E(4) / 5))
        step = unwrap(step!(tracer))
        @test step.refusals == 1
        @test step.step_length ≈ E(4) / 5 / (atan(E(4) / 3) * 10) rtol = 1 // 100
    end

    @testset "the first step only sets a scale: the second grows by up to max_first_growth" begin
        # A 1e-3 step turns the tangent by 1e-3 rad, f = 1e-2 at a 0.1 target, so the second step is 0.1 long.
        control = StepControl(;
            target_contraction = nothing, target_correction = nothing, target_turn = one(E) / 10, max_step_growth = E(2),
            max_first_growth = E(1000), max_step = E(1),
        )
        tracer = unwrap(init(CurveProblem(Circle(one(E)), E[1, 0], E[0, 1]), algorithm(E; step_control = control); first_step = E(1) / 1000))
        unwrap(step!(tracer))
        @test tracer.step_length ≈ one(E) / 10 rtol = 1 // 100
        # Later steps grow by at most max_step_growth.
        before = tracer.step_length
        unwrap(step!(tracer))
        @test tracer.step_length <= 2 * before
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
            max_first_growth = E(2), max_step = E(1),
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

    @testset "land! corrects onto a coordinate's value and keeps the tracer's point" begin
        tracer = unwrap(init(CurveProblem(Circle(one(E)), E[1, 0], E[0, 1]), algorithm(E); first_step = E(1) / 10))
        step!(tracer)
        before = copy(tracer.point)
        for value in (E(1) / 4, E(1) / 20)
            landed = unwrap(land!(tracer, 2, value))
            @test landed[2] ≈ value atol = 10 * eps(E)
            @test landed[1] ≈ sqrt(1 - value^2) atol = 100 * tolerance
        end
        @test tracer.point == before
    end

    @testset "land! refuses a hyperplane along the tangent, and a corrector that does not converge" begin
        tracer = unwrap(init(CurveProblem(Circle(one(E)), E[1, 0], E[0, 1]), algorithm(E); first_step = E(1) / 10))
        @test unwrap_error(land!(tracer, 1, one(E) / 2)) == SingularSystem
        stubborn = algorithm(E; corrector = Newton(; max_iterations = 2, converged = never, contraction_limit = one(E), error_estimate = LastCorrection()))
        tracer = unwrap(init(CurveProblem(Circle(one(E)), E[1, 0], E[0, 1]), stubborn; first_step = E(1) / 10))
        @test unwrap_error(land!(tracer, 2, one(E) / 4)) == NotConverged
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
        @test_throws ArgumentError StepControl(; targets..., max_step_growth = one(E) / 2, max_first_growth = one(E), max_step = one(E))
        @test_throws ArgumentError StepControl(; targets..., max_step_growth = one(E), max_first_growth = one(E) / 2, max_step = one(E))
        @test_throws ArgumentError StepControl(; targets..., max_step_growth = one(E), max_first_growth = one(E), max_step = zero(E))
        @test_throws ArgumentError StepControl(; targets..., max_step_growth = one(E), max_first_growth = one(E), max_step = one(E), refusal_factor = 0)
        @test_throws ArgumentError StepControl(; targets..., max_step_growth = one(E), max_first_growth = one(E), max_step = one(E), shrink = 1)
        @test_throws ArgumentError StepControl(; targets..., max_step_growth = E(Inf), max_first_growth = one(E), max_step = E(Inf))
        @test_throws ArgumentError StepControl(; targets..., max_step_growth = one(E), max_first_growth = E(Inf), max_step = E(Inf))
        @test StepControl(; targets..., max_step_growth = E(2), max_first_growth = E(100), max_step = E(Inf)).max_step == E(Inf)
        @test_throws ArgumentError StepControl(;
            target_contraction = nothing, target_correction = nothing, target_turn = -one(E), max_step_growth = one(E),
            max_first_growth = one(E), max_step = one(E),
        )
        @test_throws ArgumentError DeterminantSign(; persistence = -1)
        @test_throws ArgumentError algorithm(E; min_step_ulps = -1)
        @test_throws ArgumentError algorithm(E; max_refusals = -1)
        @test_throws ArgumentError init(CurveProblem(Fold(), E[0, 0], E[1, 0]), algorithm(E); first_step = zero(E))
        @test_throws DimensionMismatch init(CurveProblem(Fold(), E[0, 0], E[1, 0, 0]), algorithm(E); first_step = one(E))
        @test AcceptAll()(E[0], E[0], E[0])
        @test weighted_angle(ones(E, 2), E[1, 0], E[0, 1]) ≈ E(π) / 2
        @test weighted_angle(ones(E, 2), E[1, 0], E[-1, 0]) ≈ E(π)
        # A small angle keeps its digits: acos of the cosine would lose half of them.
        @test weighted_angle(ones(E, 2), E[1, 0], [cos(E(1) / 10^4), sin(E(1) / 10^4)]) ≈ E(1) / 10^4 rtol = 8 * eps(E)
        @test weighted_angle(E[4, 1], E[1, 0], E[1, 2]) ≈ E(π) / 4
        @test factored_sign(E[1 0; 0 NaN32], [1, 2]) == 0
    end

    @testset "a warm step allocates nothing" begin
        tracer = unwrap(init(CurveProblem(Circle(one(E)), E[1, 0], E[0, 1]), algorithm(E); first_step = E(1) / 10))
        @test warm_bytes(tracer) == 0
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
        settings = algorithm(BigFloat; orientation_check = DeterminantSign(; persistence = 0), linear_solver)
        tracer = unwrap(init(CurveProblem(Pitchfork(zero(BigFloat)), BigFloat[0, -1], BigFloat[0, 1]), settings; first_step = big(3) / 10))
        steps = [unwrap(step!(tracer)) for _ in 1:6]
        @test count(step -> step.flipped, steps) == 1
        @test tracer.orientation == -1
        @test all(iszero, tracer.oriented) == (linear_solver isa LUFactorization)
    end
end
