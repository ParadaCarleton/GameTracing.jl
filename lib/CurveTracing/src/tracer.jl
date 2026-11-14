"""
Why a trace ended: `Unresolved` (a predictor below the float type's resolution), `TooManyRefusals` (more refusals in
one step than `max_refusals`), `NotFiniteStart` (`H` or its Jacobian is not finite at the start) or `SingularStart`
(no tangent at the start along `direction`).
"""
@enum FailureCause Unresolved TooManyRefusals NotFiniteStart SingularStart

"""
Why a step attempt was refused: the corrector ran out of iterations (`NotConverged`) or diverged (`Diverging`), `H`
was not finite at an iterate (`NotFiniteResidual`), a bordered system was singular (`SingularSystem`), `accept`
returned `false` (`AcceptRefused`), or step control found the step too long (`TooFast`). `NoRefusal` marks an
accepted attempt.
"""
@enum Refusal NoRefusal NotConverged Diverging NotFiniteResidual SingularSystem AcceptRefused TooFast

"""
    AcceptedStep(step_length, iterations, refusals, contraction, first_correction, turn)

One accepted step: its predictor's length in the step metric, the corrector's iterations, the refusals before it, the
ratio of the corrector's second correction to its first (NaN after one correction), the first correction's length
and the tangent's turn in radians.
"""
struct AcceptedStep{E <: Real}
    step_length::E
    iterations::Int
    refusals::Int
    contraction::E
    first_correction::E
    turn::E
end

"""
    StepFailure(cause, last_refusal, step_length, refusals)

A trace's end short of its caller's goal: the `FailureCause`, the last `Refusal` before it (`NoRefusal` at the start),
the step length then and the refusals in the failed step.
"""
struct StepFailure{E <: Real}
    cause::FailureCause
    last_refusal::Refusal
    step_length::E
    refusals::Int
end

"""
    Tracer

The state of a trace: the last accepted `point`, its unit `tangent` in the step metric, the next `step_length`,
the metrics' weights at `point`, the accepted `steps`, and work arrays for the bordered systems. `init` builds one and
`step!` advances it. Read its fields; do not write them.
"""
mutable struct Tracer{E <: Real, P <: CurveProblem, A <: PseudoArclength, C}
    const problem::P
    const algorithm::A
    const point::Vector{E}
    const tangent::Vector{E}
    step_length::E
    steps::Int
    const weights::Vector{E}
    const turn_weights::Vector{E}
    const residual::Vector{E}
    const jacobian::Matrix{E}
    const bordered::Matrix{E}
    const rhs::Vector{E}
    const predicted::Vector{E}
    const iterate::Vector{E}
    const previous::Vector{E}
    const candidate::Vector{E}
    const linear::C
end

"""
    init(problem, algorithm; first_step)

A `Tracer` at `problem.start`, its tangent oriented along `problem.direction`, its first step `first_step` long in the
step metric; or the `StepFailure` (`NotFiniteStart`, `SingularStart`) that stops it there.
"""
function CommonSolve.init(
        problem::CurveProblem{S, V},
        algorithm::PseudoArclength;
        first_step::Real,
    )::Result{<:Tracer{E}, StepFailure{E}} where {S, E <: Real, V <: AbstractVector{E}}  # concrete in `started`
    first_step > 0 || throw(ArgumentError("first_step must be positive; got $first_step"))
    size = length(problem.start)
    length(problem.direction) == size || throw(DimensionMismatch("start and direction differ in length"))
    bordered = zeros(E, size, size)
    rhs = zeros(E, size)
    linear = LinearSolve.init(LinearSolve.LinearProblem(bordered, rhs), algorithm.linear_solver)
    tracer = Tracer(
        problem, algorithm, collect(problem.start), zeros(E, size), E(first_step), 0, zeros(E, size), zeros(E, size),
        zeros(E, size - 1), zeros(E, size - 1, size), bordered, rhs, zeros(E, size), zeros(E, size), zeros(E, size),
        zeros(E, size), linear,
    )
    return started(tracer)
end

"""`tracer` with its weights and unit tangent at its start, or the `StepFailure` there."""
function started(tracer::T)::Result{T, StepFailure{E}} where {E <: Real, T <: Tracer{E}}
    (; algorithm, problem) = tracer
    reweight!(tracer.weights, algorithm.metric, tracer.point)
    reweight!(tracer.turn_weights, algorithm.turn_metric, tracer.point)
    problem.system!(tracer.residual, tracer.jacobian, tracer.point)
    if !(all(isfinite, tracer.residual) && all(isfinite, tracer.jacobian))
        return Err(StepFailure(NotFiniteStart, NoRefusal, tracer.step_length, 0))
    end
    if !tangent_solve!(tracer, problem.direction)
        return Err(StepFailure(SingularStart, NoRefusal, tracer.step_length, 0))
    end
    tracer.tangent .= tracer.candidate ./ weighted_norm(tracer.weights, tracer.candidate)
    return Ok(tracer)
end

"""
    step!(tracer)

Advance `tracer` by one accepted point and return its `AcceptedStep`, or the `StepFailure` that ends the trace. A
refused attempt shrinks the step and tries again.
"""
function CommonSolve.step!(tracer::Tracer{E})::Result{AcceptedStep{E}, StepFailure{E}} where {E <: Real}
    (; algorithm) = tracer
    refusals = 0
    last_refusal = NoRefusal
    while true
        tracer.predicted .= tracer.point .+ tracer.step_length .* tracer.tangent
        if unresolved(tracer.point, tracer.predicted, algorithm.resolution_ulps)
            return Err(StepFailure(Unresolved, last_refusal, tracer.step_length, refusals))
        end
        attempt = attempt_step!(tracer)
        if attempt.refusal == NoRefusal
            taken = tracer.step_length
            accept_point!(tracer, attempt.factor)
            return Ok(
                AcceptedStep(taken, attempt.iterations, refusals, attempt.contraction, attempt.first_correction, attempt.turn)
            )
        end
        refusals += 1
        last_refusal = attempt.refusal
        if refusals > algorithm.max_refusals
            break
        end
        tracer.step_length *= algorithm.step_control.shrink
    end
    return Err(StepFailure(TooManyRefusals, last_refusal, tracer.step_length, refusals))
end

"""One step attempt's verdict and, when accepted, its corrector's figures, its turn and its step factor."""
struct Attempt{E <: Real}
    refusal::Refusal
    iterations::Int
    contraction::E
    first_correction::E
    turn::E
    factor::E
end

"""A refused `Attempt` for `refusal`."""
function refused(::Type{E}, refusal::Refusal)::Attempt{E} where {E <: Real}
    return Attempt{E}(refusal, 0, E(NaN32), E(NaN32), E(NaN32), E(NaN32))
end

"""
Correct `tracer.predicted`, then test the corrected point: `accept`, a finite system there, its tangent, and step
control's factor.
"""
function attempt_step!(tracer::Tracer{E})::Attempt{E} where {E <: Real}
    (; algorithm) = tracer
    corrected = correct!(tracer)
    if corrected.refusal != NoRefusal
        return corrected
    end
    if !algorithm.accept(tracer.point, tracer.predicted, tracer.iterate)
        return refused(E, AcceptRefused)
    end
    tracer.problem.system!(tracer.residual, tracer.jacobian, tracer.iterate)
    if !(all(isfinite, tracer.residual) && all(isfinite, tracer.jacobian))
        return refused(E, NotFiniteResidual)
    end
    if !tangent_solve!(tracer, tracer.tangent)
        return refused(E, SingularSystem)
    end
    turn = weighted_angle(tracer.turn_weights, tracer.tangent, tracer.candidate)
    factor = step_factor(algorithm.step_control, corrected.contraction, corrected.first_correction, turn)
    if factor > algorithm.step_control.refusal_factor
        return refused(E, TooFast)
    end
    return Attempt{E}(NoRefusal, corrected.iterations, corrected.contraction, corrected.first_correction, turn, factor)
end

"""
Newton's method from `tracer.predicted` on `[H(x); ⟨t, x - x̂⟩] = 0`, leaving its last iterate in `tracer.iterate`.
An `Attempt` with the iterations, the contraction and the first correction once `converged` holds, else its refusal.
"""
function correct!(tracer::Tracer{E})::Attempt{E} where {E <: Real}
    (; corrector) = tracer.algorithm
    width = length(tracer.residual)
    copyto!(tracer.iterate, tracer.predicted)
    before, first_correction, contraction = E(NaN32), E(NaN32), E(NaN32)
    for iteration in 1:corrector.max_iterations
        tracer.problem.system!(tracer.residual, tracer.jacobian, tracer.iterate)
        if !(all(isfinite, tracer.residual) && all(isfinite, tracer.jacobian))
            return refused(E, NotFiniteResidual)
        end
        border!(tracer, tracer.weights, tracer.tangent)
        @views tracer.rhs[1:width] .= .-tracer.residual
        tracer.rhs[end] = -weighted_offset(tracer.weights, tracer.tangent, tracer.predicted, tracer.iterate)
        if !bordered_solve!(tracer, tracer.candidate)
            return refused(E, SingularSystem)
        end
        copyto!(tracer.previous, tracer.iterate)
        tracer.iterate .+= tracer.candidate
        correction = correction_length(corrector.distance, tracer.weights, tracer.previous, tracer.iterate)
        if iteration == 1
            first_correction = correction
        elseif iteration == 2
            contraction = correction / first_correction
        end
        # Negated so a NaN correction counts as diverging.
        if iteration > 1 && !(correction <= corrector.contraction_limit * before)
            return refused(E, Diverging)
        end
        before = correction
        if corrector.converged(tracer.previous, tracer.iterate, tracer.residual)
            return Attempt{E}(NoRefusal, iteration, contraction, first_correction, E(NaN32), E(NaN32))
        end
    end
    return refused(E, NotConverged)
end

"""A correction's length from `from` to `to`: the step metric's norm, or the corrector's own `distance`."""
function correction_length(::MetricDistance, weights::AbstractVector{E}, from::AbstractVector{E}, to::AbstractVector{E})::E where {E <: Real}
    return weighted_distance(weights, from, to)
end

function correction_length(distance::Any, ::AbstractVector{E}, from::AbstractVector{E}, to::AbstractVector{E})::E where {E <: Real}
    return E(distance(from, to))
end

"""
The tangent at the point whose Jacobian `tracer.jacobian` holds, into `tracer.candidate`: the solution of
`[J; (w ∘ previous)ᵀ] t = [0; 1]` with `w` the turn metric's weights, so `⟨t, previous⟩` is one, positive. Whether
the solve succeeded.
"""
function tangent_solve!(tracer::Tracer{E}, previous::AbstractVector{E})::Bool where {E <: Real}
    border!(tracer, tracer.turn_weights, previous)
    fill!(tracer.rhs, zero(E))
    tracer.rhs[end] = one(E)
    return bordered_solve!(tracer, tracer.candidate)
end

"""Fill `tracer.bordered` with the Jacobian over the row `weights ∘ row`."""
function border!(tracer::Tracer{E}, weights::AbstractVector{E}, row::AbstractVector{E})::Matrix{E} where {E <: Real}
    width = length(tracer.residual)
    @views tracer.bordered[1:width, :] .= tracer.jacobian
    @views tracer.bordered[end, :] .= weights .* row
    return tracer.bordered
end

"""Solve `tracer.bordered \\ tracer.rhs` into `destination`; whether the solve succeeded with a finite solution."""
function bordered_solve!(tracer::Tracer{E}, destination::AbstractVector{E})::Bool where {E <: Real}
    tracer.linear.A = tracer.bordered
    tracer.linear.b = tracer.rhs
    solution = LinearSolve.solve!(tracer.linear)
    if !SciMLBase.successful_retcode(solution)
        return false
    end
    copyto!(destination, solution.u)
    return all(isfinite, destination)
end

"""
Whether `predicted` moves no coordinate of `point` by more than `ulps` ulps of its own size: the per-coordinate test
`|Δxᵢ| ≤ ulps · eps · |xᵢ|`, so coordinates of different scales each count, and any move of a zero coordinate
resolves.
"""
function unresolved(point::AbstractVector{E}, predicted::AbstractVector{E}, ulps::Integer)::Bool where {E <: Real}
    for index in eachindex(point, predicted)
        # Negated so a NaN move counts as resolved, as the comparison is false for it.
        if !(abs(predicted[index] - point[index]) <= ulps * eps(E) * abs(point[index]))
            return false
        end
    end
    return true
end

"""Allgower and Georg's step factor `f` (`StepControl`), at least `1 / growth_limit`."""
function step_factor(control::StepControl{E}, contraction::E, first_correction::E, turn::E)::E where {E <: Real}
    corrector_term = max(
        sqrt(factor_term(control.nominal_contraction, contraction)),
        sqrt(factor_term(control.nominal_distance, first_correction)),
    )
    return max(one(E) / control.growth_limit, max(corrector_term, factor_term(control.nominal_turn, turn)))
end

"""`measured / nominal`, or zero when the term is off (`nothing`) or `measured` is NaN."""
function factor_term(::Nothing, ::E)::E where {E <: Real}
    return zero(E)
end

function factor_term(nominal::E, measured::E)::E where {E <: Real}
    return isnan(measured) ? zero(E) : measured / nominal
end

"""Move `tracer` to its corrected point with the new tangent, reweight both metrics there and size the next step."""
function accept_point!(tracer::Tracer{E}, factor::E)::Tracer{E} where {E <: Real}
    (; algorithm) = tracer
    copyto!(tracer.point, tracer.iterate)
    reweight!(tracer.weights, algorithm.metric, tracer.point)
    reweight!(tracer.turn_weights, algorithm.turn_metric, tracer.point)
    tracer.tangent .= tracer.candidate ./ weighted_norm(tracer.weights, tracer.candidate)
    tracer.step_length = min(tracer.step_length / factor, algorithm.step_control.max_step)
    tracer.steps += 1
    return tracer
end
