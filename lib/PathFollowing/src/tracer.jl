"""
Why a trace ended: `Unresolved` (a predictor below the float type's resolution), `TooManyRefusals` (more refusals in
one step than `max_refusals`), `NotFiniteStart` (`H` or its Jacobian is not finite at the start) or `SingularStart`
(no tangent at the start along `direction`).
"""
@enum FailureCause Unresolved TooManyRefusals NotFiniteStart SingularStart

"""
Why a step attempt was refused: the corrector ran out of iterations (`NotConverged`) or diverged (`Diverging`), `H`
was not finite at an iterate (`NotFiniteResidual`), a bordered system was singular (`SingularSystem`), `accept`
returned `false` (`AcceptRefused`), step control found the step too long (`TooFast`), or the orientation check found
a changed sign (`OrientationFlip`). `NoRefusal` marks an accepted attempt. `land!` also refuses `NoCrossing`: neither
tangent of the last step leads to the value.
"""
@enum Refusal NoRefusal NotConverged Diverging NotFiniteResidual SingularSystem AcceptRefused TooFast OrientationFlip NoCrossing

"""
    AcceptedStep(step_length, iterations, refusals, contraction, first_correction, turn)

One accepted step: its predictor's length in the step metric, the corrector's iterations, the refusals before it, the
ratio of the corrector's second correction to its first (NaN after one correction), the first correction's length,
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

The state of a trace: the last accepted `point`, its unit `tangent` in the step metric, the point the last step
started from and its tangent (`step_start`, `start_tangent`; the start itself before any step), the next
`step_length`, the accepted `steps`, the `orientation` sign at `point` (zero without an orientation check), the step
and turn metrics frozen at `point` (`frozen_metric`), and work arrays; `row` is the border row of the next bordered
system. `init` builds one and `step!` advances it. Read its fields; do not write them.
"""
mutable struct Tracer{E <: Real, P <: CurveProblem, A <: PseudoArclength, C, M, T}
    const problem::P
    const algorithm::A
    const point::Vector{E}
    const tangent::Vector{E}
    const step_start::Vector{E}
    const start_tangent::Vector{E}
    step_length::E
    steps::Int
    orientation::Int8
    const metric::M
    const turn_metric::T
    const scratch::Vector{E}
    const residual::Vector{E}
    const jacobian::Matrix{E}
    const bordered::Matrix{E}
    const row::Vector{E}
    const rhs::Vector{E}
    const predicted::Vector{E}
    const iterate::Vector{E}
    const previous::Vector{E}
    const candidate::Vector{E}
    const oriented::Matrix{E}
    const pivots::Vector{LinearAlgebra.BlasInt}
    const linear::C
end

"""
    init(problem, algorithm; first_step)

A `Tracer` at `problem.start`, its tangent oriented along `problem.direction`, its first step `first_step` long in the
step metric and at most the step control's `max_step`; or the `StepFailure` (`NotFiniteStart`, `SingularStart`) that
stops it there. `first_step` must be positive and finite.
"""
function CommonSolve.init(
        problem::CurveProblem{S, V},
        algorithm::PseudoArclength;
        first_step::Real,
    )::Result{<:Tracer{E}, StepFailure{E}} where {S, E <: Real, V <: AbstractVector{E}}  # concrete in `started`
    # Checked in `E`, where a tiny or huge `first_step` rounds to 0 or Inf.
    0 < E(first_step) < Inf || throw(ArgumentError("first_step must be positive and finite in $E; got $first_step"))
    size = length(problem.start)
    length(problem.direction) == size || throw(DimensionMismatch("start and direction differ in length"))
    bordered = zeros(E, size, size)
    rhs = zeros(E, size)
    # A failed solve is a refusal (`SingularSystem`), so LinearSolve's own warning would only repeat it.
    linear = LinearSolve.init(LinearSolve.LinearProblem(bordered, rhs), algorithm.linear_solver; verbose = false)
    tracer = Tracer(
        problem, algorithm, collect(problem.start), zeros(E, size), collect(problem.start), zeros(E, size), min(E(first_step), algorithm.step_control.max_step), 0, Int8(0),
        frozen_metric(algorithm.metric, problem.start), frozen_metric(algorithm.turn_metric, problem.start), zeros(E, size),
        zeros(E, size - 1), zeros(E, size - 1, size), bordered, zeros(E, size), rhs, zeros(E, size), zeros(E, size),
        zeros(E, size), zeros(E, size), zeros(E, size, size), zeros(LinearAlgebra.BlasInt, size), linear,
    )
    return started(tracer)
end

"""`tracer` with its unit tangent at its start, where `init` froze its metrics, or the `StepFailure` there."""
function started(tracer::T)::Result{T, StepFailure{E}} where {E <: Real, T <: Tracer{E}}
    (; problem) = tracer
    problem.system!(tracer.residual, tracer.jacobian, tracer.point)
    if !(all(isfinite, tracer.residual) && all(isfinite, tracer.jacobian))
        return Err(StepFailure(NotFiniteStart, NoRefusal, tracer.step_length, 0))
    end
    oriented = tangent_solve!(tracer, problem.direction)
    if is_error(oriented)
        return Err(StepFailure(SingularStart, NoRefusal, tracer.step_length, 0))
    end
    tracer.tangent .= tracer.candidate ./ metric_norm(tracer.metric, tracer.candidate)
    copyto!(tracer.start_tangent, tracer.tangent)
    tracer.orientation = unwrap(oriented)
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
        if unresolved(tracer.point, tracer.predicted, algorithm.min_step_ulps)
            return Err(StepFailure(Unresolved, last_refusal, tracer.step_length, refusals))
        end
        attempt = attempt_step!(tracer)
        if attempt.refusal == NoRefusal
            taken = tracer.step_length
            accept_point!(tracer, attempt)
            return Ok(AcceptedStep(taken, attempt.iterations, refusals, attempt.contraction, attempt.first_correction, attempt.turn))
        end
        refusals += 1
        last_refusal = attempt.refusal
        if refusals > algorithm.max_refusals
            break
        end
        # An infinite factor measured nothing usable, so that refusal shrinks like an unmeasured one.
        shrink = inv(algorithm.step_control.shrink)
        tracer.step_length /= ifelse(isfinite(attempt.factor), max(attempt.factor, shrink), shrink)
    end
    return Err(StepFailure(TooManyRefusals, last_refusal, tracer.step_length, refusals))
end

"""
One step attempt's verdict and, when accepted, its corrector's figures, its turn, its step factor and the orientation
sign at its point.
"""
struct Attempt{E <: Real}
    refusal::Refusal
    iterations::Int
    contraction::E
    first_correction::E
    turn::E
    factor::E
    orientation::Int8
end

"""A refused `Attempt` for `refusal`, with the step `factor` its measurements gave (zero for none)."""
function refused(::Type{E}, refusal::Refusal, factor::E = zero(E))::Attempt{E} where {E <: Real}
    return Attempt{E}(refusal, 0, E(NaN32), E(NaN32), E(NaN32), factor, Int8(0))
end

"""
Correct `tracer.predicted`, then test the corrected point: `accept`, a finite system there, its tangent, step
control's factor and its orientation sign.
"""
function attempt_step!(tracer::Tracer{E})::Attempt{E} where {E <: Real}
    (; algorithm) = tracer
    metric_apply!(tracer.row, tracer.metric, tracer.tangent)
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
    oriented = tangent_solve!(tracer, tracer.tangent)
    if is_error(oriented)
        return refused(E, SingularSystem)
    end
    turn = metric_angle(tracer.turn_metric, tracer.scratch, tracer.tangent, tracer.candidate)
    factor = step_factor(algorithm.step_control, corrected.contraction, corrected.first_correction, turn)
    if factor > algorithm.step_control.refusal_factor
        return refused(E, TooFast, factor)
    end
    orientation = unwrap(oriented)
    if flip_refused(algorithm.orientation_check, tracer.orientation, orientation)
        return refused(E, OrientationFlip)
    end
    return Attempt{E}(NoRefusal, corrected.iterations, corrected.contraction, corrected.first_correction, turn, factor, orientation)
end

"""
The sign of `det([J; tᵀ])` under `check` for `J = tracer.jacobian` and the tangent `t` that the bordered solve just
put in `tracer.candidate`; zero without a check. Call it only from `tangent_solve!`, which holds that solve.
"""
orientation_sign!(::Tracer, ::Nothing)::Int8 = Int8(0)
orientation_sign!(tracer::Tracer, ::DeterminantSign)::Int8 = solved_sign!(tracer, tracer.linear.cacheval)

"""
The sign from the bordered solve's own LU of `[J; rᵀ]`. With `rᵀt = 1`, the matrix determinant lemma gives
`det([J; tᵀ]) = det([J; rᵀ]) · tᵀt`, so both signs agree.
"""
solved_sign!(::Tracer, solved::LinearAlgebra.LU)::Int8 = factored_sign(solved.factors, solved.ipiv)

"""The sign from a second factorization, of `[J; tᵀ]` itself, for a linear solver that keeps no LU."""
function solved_sign!(tracer::Tracer, ::Any)::Int8
    width = length(tracer.residual)
    @views tracer.oriented[1:width, :] .= tracer.jacobian
    @views tracer.oriented[end, :] .= tracer.candidate
    (factors, pivots) = factor!(tracer.oriented, tracer.pivots)
    return factored_sign(factors, pivots)
end

"""`matrix`'s pivoted LU factors in place and its row swaps: LAPACK's `getrf!` for BLAS floats, else `lu!`."""
function factor!(
        matrix::Matrix{E},
        pivots::Vector{LinearAlgebra.BlasInt},
    )::Tuple{Matrix{E}, Vector{LinearAlgebra.BlasInt}} where {E <: LinearAlgebra.BlasReal}
    LinearAlgebra.LAPACK.getrf!(matrix, pivots; check = false)
    return (matrix, pivots)
end

function factor!(matrix::Matrix{E}, ::Vector{LinearAlgebra.BlasInt})::Tuple{Matrix{E}, Vector{LinearAlgebra.BlasInt}} where {E <: Real}
    factored = LinearAlgebra.lu!(matrix; check = false)
    return (factored.factors, factored.ipiv)
end

"""The sign of a determinant from its pivoted LU `factors` and row swaps `pivots`; zero for a zero or NaN pivot."""
function factored_sign(factors::AbstractMatrix{E}, pivots::AbstractVector{<:Integer})::Int8 where {E <: Real}
    signed = Int8(1)
    for index in eachindex(pivots)
        pivot = factors[index, index]
        signed *= (Int8(pivot > 0) - Int8(pivot < 0)) * ifelse(pivots[index] == index, Int8(1), Int8(-1))
    end
    return signed
end

"""Whether `check` refuses a step whose orientation sign went from `before` to `after`."""
flip_refused(::Nothing, ::Int8, ::Int8)::Bool = false
flip_refused(::DeterminantSign, before::Int8, after::Int8)::Bool = after != before

"""
Newton's method from `tracer.predicted` on `[H(x); r ⋅ (x - x̂)] = 0`, `r = tracer.row`, leaving its last iterate in
`tracer.iterate`. An `Attempt` with the iterations, the contraction and the first correction once `converged` holds,
else its refusal.
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
        border!(tracer)
        @views tracer.rhs[1:width] .= .-tracer.residual
        tracer.rhs[end] = -row_offset(tracer.row, tracer.predicted, tracer.iterate)
        if !bordered_solve!(tracer, tracer.candidate)
            return refused(E, SingularSystem)
        end
        copyto!(tracer.previous, tracer.iterate)
        tracer.iterate .+= tracer.candidate
        correction = correction_length(corrector.distance, tracer, tracer.previous, tracer.iterate)
        if iteration == 1
            first_correction = correction
        elseif iteration == 2
            contraction = correction / first_correction
        end
        # Negated so a NaN correction counts as diverging.
        if iteration > 1 && !(correction <= corrector.contraction_limit * before)
            return refused(E, Diverging, step_factor(tracer.algorithm.step_control, contraction, first_correction, E(NaN32)))
        end
        scale = error_scale(corrector.error_estimate, correction, before)
        before = correction
        if corrector.converged(tracer.previous, tracer.iterate, tracer.residual, scale)
            return Attempt{E}(NoRefusal, iteration, contraction, first_correction, E(NaN32), E(NaN32), Int8(0))
        end
    end
    return refused(E, NotConverged, step_factor(tracer.algorithm.step_control, contraction, first_correction, E(NaN32)))
end

"""A correction's length from `from` to `to`: `tracer`'s step metric norm, or the corrector's own `distance`."""
function correction_length(::MetricDistance, tracer::Tracer{E}, from::AbstractVector{E}, to::AbstractVector{E})::E where {E <: Real}
    return metric_distance(tracer.metric, tracer.scratch, from, to)
end

function correction_length(distance::Any, ::Tracer{E}, from::AbstractVector{E}, to::AbstractVector{E})::E where {E <: Real}
    return E(distance(from, to))
end

"""
The tangent at the point whose Jacobian `tracer.jacobian` holds, into `tracer.candidate`: the solution of
`[J; (M previous)ᵀ] t = [0; 1]` with `M` the turn metric, so `⟨t, previous⟩` is one, positive. The orientation sign
at that tangent (`orientation_sign!`), or `none` when the solve fails.
"""
function tangent_solve!(tracer::Tracer{E}, previous::AbstractVector{E})::Option{Int8} where {E <: Real}
    metric_apply!(tracer.row, tracer.turn_metric, previous)
    border!(tracer)
    fill!(tracer.rhs, zero(E))
    tracer.rhs[end] = one(E)
    if !bordered_solve!(tracer, tracer.candidate)
        return none(Int8)
    end
    return some(orientation_sign!(tracer, tracer.algorithm.orientation_check))
end

"""Fill `tracer.bordered` with the Jacobian over `tracer.row`."""
function border!(tracer::Tracer{E})::Matrix{E} where {E <: Real}
    width = length(tracer.residual)
    @views tracer.bordered[1:width, :] .= tracer.jacobian
    @views tracer.bordered[end, :] .= tracer.row
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

"""Allgower and Georg's step factor `f` (`StepControl`); a NaN measurement drops its term."""
function step_factor(control::StepControl{E}, contraction::E, first_correction::E, turn::E)::E where {E <: Real}
    corrector_term = max(
        sqrt(factor_term(control.target_contraction, contraction)),
        sqrt(factor_term(control.target_correction, first_correction)),
    )
    return max(corrector_term, factor_term(control.target_turn, turn))
end

"""`measured / target`, or zero when the term is off (`nothing`) or `measured` is NaN."""
function factor_term(::Nothing, ::E)::E where {E <: Real}
    return zero(E)
end

function factor_term(target::E, measured::E)::E where {E <: Real}
    return isnan(measured) ? zero(E) : measured / target
end

"""
Move `tracer` to `attempt`'s corrected point with the new tangent and orientation, keep the step's start and its
tangent, freeze both metrics there and size the next step.
"""
function accept_point!(tracer::Tracer{E}, attempt::Attempt{E})::Tracer{E} where {E <: Real}
    (; algorithm) = tracer
    copyto!(tracer.step_start, tracer.point)
    copyto!(tracer.start_tangent, tracer.tangent)
    copyto!(tracer.point, tracer.iterate)
    tracer.orientation = attempt.orientation
    freeze_metric!(tracer.metric, tracer.point)
    freeze_metric!(tracer.turn_metric, tracer.point)
    tracer.tangent .= tracer.candidate ./ metric_norm(tracer.metric, tracer.candidate)
    (; max_step_growth, max_step) = algorithm.step_control
    tracer.step_length = min(tracer.step_length / max(attempt.factor, inv(max_step_growth)), max_step)
    tracer.steps += 1
    return tracer
end

"""
    land!(tracer, coordinate, value)

The curve's point where `x[coordinate] = value` near the last step, or the `Refusal` that stopped its corrector. The
predictor follows the point's tangent back to that hyperplane, or else the step start's tangent forward to it
(`landing_origin`), and Newton's method corrects on it. A step over a fold's top can cross the value twice, and the
point's tangent then leads on to a later crossing; the start's leads to the first. The tracer keeps its point, tangent
and step.
"""
function land!(tracer::Tracer{E}, coordinate::Integer, value::Real)::Result{Vector{E}, Refusal} where {E <: Real}
    origin = landing_origin(tracer, coordinate, E(value))
    if is_error(origin)
        return Err(unwrap_error(origin))
    end
    (from, along) = unwrap(origin)
    tracer.predicted .= from .+ ((E(value) - from[coordinate]) / along[coordinate]) .* along
    tracer.predicted[coordinate] = E(value)
    fill!(tracer.row, zero(E))
    tracer.row[coordinate] = one(E)
    corrected = correct!(tracer)
    if corrected.refusal != NoRefusal
        return Err(corrected.refusal)
    end
    return Ok(copy(tracer.iterate))
end

"""
The point and tangent a landing on `x[coordinate] = value` predicts from: the point when its tangent leads back to the
value, else the step's start when its tangent leads forward to it. Where both lead there, the point predicts better:
from the start, a long step's landing diverged on a random game. Two tangents parallel to the hyperplane refuse as
`SingularSystem`, and two that lead away from the value as `NoCrossing`.
"""
function landing_origin(tracer::Tracer{E}, coordinate::Integer, value::E)::Result{NTuple{2, Vector{E}}, Refusal} where {E <: Real}
    (; step_start, start_tangent, point, tangent) = tracer
    if leads_to(point, tangent, coordinate, value, -one(E))
        return Ok((point, tangent))
    elseif leads_to(step_start, start_tangent, coordinate, value, one(E))
        return Ok((step_start, start_tangent))
    end
    return Err(iszero(start_tangent[coordinate]) && iszero(tangent[coordinate]) ? SingularSystem : NoCrossing)
end

"""Whether `tangent` leads from `from` to `x[coordinate] = value` in the sense `sense`: `1` forward, `-1` back."""
function leads_to(from::AbstractVector{E}, tangent::AbstractVector{E}, coordinate::Integer, value::E, sense::E)::Bool where {E <: Real}
    return !iszero(tangent[coordinate]) && sense * (value - from[coordinate]) * tangent[coordinate] >= 0
end
