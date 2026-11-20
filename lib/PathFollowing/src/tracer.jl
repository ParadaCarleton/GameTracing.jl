"""
Why a trace ended: `Unresolved` (a predictor below the float type's resolution), `TooManyRefusals` (more refusals in
one step than `max_refusals`), `NotFiniteStart` (`H` or its Jacobian is not finite at the start), `SingularStart`
(no tangent at the start along `direction`), or `TooManySteps` (a `land!!` past its step budget). `NoFailure`
marks a landing in a `Landing`; a `StepFailure` never holds it.
"""
@enum FailureCause Unresolved TooManyRefusals NotFiniteStart SingularStart TooManySteps NoFailure

"""
Why a step attempt was refused: the corrector ran out of iterations (`NotConverged`) or diverged (`Diverging`), `H`
was not finite at an iterate (`NotFiniteResidual`), a bordered system was singular (`SingularSystem`), `accept`
returned `false` (`AcceptRefused`), step control found the step too long (`TooFast`), or the orientation check found
a changed sign (`OrientationFlip`), the corrected point went past the step's `Bound` (`PastBound`), it lies
farther from its predictor than step control's `reach` (`FarFromPredictor`), or two half steps did not reach it
(`SplitMismatch`, `HalfStepCheck`). `NoRefusal` marks an accepted attempt.
"""
@enum Refusal NoRefusal NotConverged Diverging NotFiniteResidual SingularSystem AcceptRefused TooFast OrientationFlip PastBound FarFromPredictor SplitMismatch

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
the step length its failing check judged and the refusals in the failed step. That length is the first step at the
start, and the last refused predictor, cut at a `Bound`, for `TooManyRefusals`. For `Unresolved` it is the controller's
step, or the retry that did not shrink below the predictor it refuses.
"""
struct StepFailure{E <: Real}
    cause::FailureCause
    last_refusal::Refusal
    step_length::E
    refusals::Int
    function StepFailure(cause::FailureCause, last_refusal::Refusal, step_length::E, refusals::Integer)::StepFailure{E} where {E <: Real}
        cause == NoFailure && throw(ArgumentError("a StepFailure needs a cause; NoFailure marks a landing"))
        return new{E}(cause, last_refusal, step_length, refusals)
    end
end

"""
    Tracer

The state of a trace: the last accepted `point`, its unit `tangent` in the step metric, the next `step_length`, the
accepted `steps`, the `orientation` sign at `point` (zero without an orientation check), the corrector's bound on the
error left at `point` (`settle_error`; zero at the start), the step and turn metrics frozen at `point`, the work arrays
(`Workspace`) and the linear solver's state. `init` builds one and `step!!` advances it.

Its arrays take the start's array type: `similar` arrays on the start's device for a mutable start, static arrays for a
static one. A step changes mutable arrays in place, so use only the tracer it returns. Read its fields; do not write
them.
"""
struct Tracer{E <: Real, V <: AbstractVector{E}, P <: CurveProblem, A <: PseudoArclength, M, T, W <: Workspace, L}
    problem::P
    algorithm::A
    point::V
    tangent::V
    step_length::E
    steps::Int
    orientation::Int8
    settle_error::E
    metric::M
    turn_metric::T
    work::W
    linear::L
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
    0 < E(first_step) < Inf || throw(ArgumentError(lazy"first_step must be positive and finite in $E; got $first_step"))
    # The step control computes in its own type: a shrink below 1 there can round back to the same step in `E`.
    algorithm.step_control isa StepControl{E} ||
        throw(ArgumentError(lazy"the step control's element type must be the start's $E; got $(typeof(algorithm.step_control))"))
    algorithm.split_check isa Union{Nothing, HalfStepCheck{E}} ||
        throw(ArgumentError(lazy"the split check's element type must be the start's $E; got $(typeof(algorithm.split_check))"))
    (; start) = problem
    length(problem.direction) == length(start) || throw(DimensionMismatch("start and direction differ in length"))
    work = workspace(start, E)
    tracer = Tracer(
        problem, algorithm, Into(zero_point(start, E)) .= start, zero_point(start, E),
        min(E(first_step), algorithm.step_control.max_step), 0, Int8(0), zero(E), frozen_metric(algorithm.metric, start),
        frozen_metric(algorithm.turn_metric, start), work, linear_state(algorithm.linear_solver, work),
    )
    return started(tracer)
end

"""`tracer` with its unit tangent at its start, where `init` froze its metrics, or the `StepFailure` there."""
function started(tracer::T)::Result{T, StepFailure{E}} where {E <: Real, T <: Tracer{E}}
    tracer, finite = evaluate!!(tracer, tracer.point)
    finite || return Err(StepFailure(NotFiniteStart, NoRefusal, tracer.step_length, 0))
    tracer, oriented = tangent_solve!!(tracer, tracer.problem.direction)
    is_error(oriented) && return Err(StepFailure(SingularStart, NoRefusal, tracer.step_length, 0))
    @reset tracer.tangent = Into(tracer.tangent) .= tracer.work.candidate ./ metric_norm(tracer.metric, tracer.work.candidate)
    @reset tracer.orientation = unwrap(oriented)
    return Ok(tracer)
end

"""
`tracer` with `H` and its Jacobian at `point` in its work arrays (`CurveProblem`'s `system!!`), and whether both are
finite. A system that returns arrays of other types than it got throws an `ArgumentError`.
"""
function evaluate!!(tracer::T, point::AbstractVector)::Tuple{T, Bool} where {T <: Tracer}
    (; work) = tracer
    residual, jacobian = tracer.problem.system!!(work.residual, work.jacobian, point)
    # Folded away when the types match, as they must for the tracer to keep its type.
    residual isa typeof(work.residual) && jacobian isa typeof(work.jacobian) || throw(
        ArgumentError(lazy"system!! must return its residual and Jacobian as a $(typeof(work.residual)) and a $(typeof(work.jacobian))")
    )
    @reset tracer.work.residual = residual
    @reset tracer.work.jacobian = jacobian
    return tracer, all(isfinite, residual) && all(isfinite, jacobian)
end

"""
    Bound(coordinate, value)

A value of one coordinate that a step (`step!!`) stops at: `x[coordinate] = value`.
"""
struct Bound{E <: Real}
    coordinate::Int
    value::E
end

"""
    tracer, outcome = step!!(tracer[, bound])

Advance `tracer` by one accepted point. Return the advanced tracer and the step's `AcceptedStep`, or the `StepFailure`
that ends the trace. Use only the returned tracer: the step changes the old one's mutable arrays in place. A refused
attempt shrinks the predictor it took, cut or not, and tries again. With a `Bound`, the step stays on the point's side
of the bound or stops on it. A predictor that reaches the bound is cut where its line meets it, and Newton's method
corrects at the bound's value, under every test of a step. A corrected point past the bound refuses as `PastBound`. The
step sees the bound only at its predictor and corrected point: a curve that passes the bound and comes back within one
step goes unseen. The resolution check judges the controller's step before the cut, as a landing always moves the
point. A retry no shorter than the predictor it refuses fails `Unresolved`. The point must lie off the bound.
"""
function step!!(
        tracer::T,
        bound::Union{Nothing, Bound{E}} = nothing,
    )::Tuple{T, Result{AcceptedStep{E}, StepFailure{E}}} where {E <: Real, T <: Tracer{E}}
    (; algorithm) = tracer
    on_bound(bound, tracer.point) && throw(ArgumentError(lazy"the point lies on the bound $bound"))
    refusals = 0
    last_refusal = NoRefusal
    taken = tracer.step_length
    while true
        @reset tracer.work.predicted = Into(tracer.work.predicted) .= tracer.point .+ tracer.step_length .* tracer.tangent
        # Checked before the cut: a cut predictor lands on the bound, so its length shows no stagnation.
        if unresolved(tracer.point, tracer.work.predicted, algorithm.min_step_ulps)
            return tracer, Err(StepFailure(Unresolved, last_refusal, tracer.step_length, refusals))
        end
        lands = reaches(bound, tracer.point, tracer.work.predicted)
        tracer, taken = lands ? cut_predictor!!(tracer, bound) : (tracer, tracer.step_length)
        tracer, attempt = attempt_step!!(tracer, bound, lands, taken)
        if attempt.refusal == NoRefusal
            tracer = accept_point!!(tracer, attempt, taken)
            return tracer, Ok(AcceptedStep(taken, attempt.iterations, refusals, attempt.contraction, attempt.first_correction, attempt.turn))
        end
        refusals += 1
        last_refusal = attempt.refusal
        if refusals > algorithm.max_refusals
            break
        end
        # The retry shrinks the predictor taken, cut or not; an infinite factor shrinks like an unmeasured one.
        shrink = inv(algorithm.step_control.shrink)
        @reset tracer.step_length = taken / ifelse(isfinite(attempt.factor), max(attempt.factor, shrink), shrink)
        # A retry that rounds back to the refused length would repeat it: the step is below resolution.
        if !(tracer.step_length < taken)
            return tracer, Err(StepFailure(Unresolved, last_refusal, tracer.step_length, refusals))
        end
    end
    return tracer, Err(StepFailure(TooManyRefusals, last_refusal, taken, refusals))
end

"""
One step attempt's verdict and, when accepted, its corrector's figures, the `scale` its convergence test saw, its
turn, its step factor and the orientation sign at its point.
"""
struct Attempt{E <: Real}
    refusal::Refusal
    iterations::Int
    contraction::E
    first_correction::E
    error_scale::E
    turn::E
    factor::E
    orientation::Int8
end

"""A refused `Attempt` for `refusal`, with the step `factor` its measurements gave (zero for none)."""
function refused(::Type{E}, refusal::Refusal, factor::E = zero(E))::Attempt{E} where {E <: Real}
    return Attempt{E}(refusal, 0, E(NaN32), E(NaN32), E(NaN32), E(NaN32), factor, Int8(0))
end

"""Whether `point` lies on `bound`; never without one."""
on_bound(::Nothing, ::AbstractVector)::Bool = false
function on_bound(bound::Bound{E}, point::AbstractVector{E})::Bool where {E <: Real}
    return element(point, bound.coordinate) == bound.value
end

"""Whether `to` lies on `bound` or on its other side from `from`; never without one."""
reaches(::Nothing, ::AbstractVector, ::AbstractVector)::Bool = false
function reaches(bound::Bound{E}, from::AbstractVector{E}, to::AbstractVector{E})::Bool where {E <: Real}
    (; coordinate, value) = bound
    reached = element(to, coordinate)
    return signbit(element(from, coordinate) - value) != signbit(reached - value) || reached == value
end

"""
Cut the predictor where its line from the point meets `bound`, and border the corrector's system with the bound's
coordinate. The cut predictor's length in the step metric.
"""
function cut_predictor!!(tracer::T, bound::Bound{E})::Tuple{T, E} where {E <: Real, T <: Tracer{E}}
    (; coordinate, value) = bound
    from = element(tracer.point, coordinate)
    fraction = (value - from) / (element(tracer.work.predicted, coordinate) - from)
    taken = fraction * tracer.step_length
    @reset tracer.work.predicted = Into(tracer.work.predicted) .= tracer.point .+ taken .* tracer.tangent
    @reset tracer.work.predicted = set_element!!(tracer.work.predicted, coordinate, value)
    @reset tracer.work.row = unit!!(tracer.work.row, coordinate)
    return tracer, taken
end

"""
After a correction against `bound`: a landing (`lands`) keeps the bound's value exactly; else whether the corrected
point went past the bound.
"""
function past_bound!!(tracer::T, ::Nothing, ::Bool)::Tuple{T, Bool} where {T <: Tracer}
    return tracer, false
end

function past_bound!!(tracer::T, bound::Bound{E}, lands::Bool)::Tuple{T, Bool} where {E <: Real, T <: Tracer{E}}
    if lands
        # The bordered row holds the coordinate at the value; this drops the solves' rounding.
        @reset tracer.work.iterate = set_element!!(tracer.work.iterate, bound.coordinate, bound.value)
        return tracer, false
    end
    return tracer, reaches(bound, tracer.point, tracer.work.iterate) && !on_bound(bound, tracer.work.iterate)
end

"""
Correct the predictor, `taken` long, then test the corrected point: its distance from the predictor, `bound`,
`accept`, a finite system there, its tangent, step control's factor, its orientation sign and the split check. A
landing (`lands`) holds its row from `cut_predictor!!`; any other attempt borders with the tangent.
"""
function attempt_step!!(tracer::T, bound::Union{Nothing, Bound{E}}, lands::Bool, taken::E)::Tuple{T, Attempt{E}} where {E <: Real, T <: Tracer{E}}
    (; algorithm) = tracer
    if !lands
        @reset tracer.work.row = metric_apply!!(tracer.work.row, tracer.metric, tracer.tangent)
    end
    tracer, corrected = correct!!(tracer)
    corrected.refusal == NoRefusal || return tracer, corrected
    far_from_predictor(tracer, taken, corrected.error_scale) && return tracer, refused(E, FarFromPredictor)
    tracer, past = past_bound!!(tracer, bound, lands)
    past && return tracer, refused(E, PastBound)
    algorithm.accept(tracer.point, tracer.work.predicted, tracer.work.iterate) || return tracer, refused(E, AcceptRefused)
    tracer, finite = evaluate!!(tracer, tracer.work.iterate)
    finite || return tracer, refused(E, NotFiniteResidual)
    tracer, oriented = tangent_solve!!(tracer, tracer.tangent)
    is_error(oriented) && return tracer, refused(E, SingularSystem)
    turn = metric_angle(tracer.turn_metric, tracer.work.scratch, tracer.tangent, tracer.work.candidate)
    factor = step_factor(algorithm.step_control, corrected.contraction, corrected.first_correction, turn)
    factor > algorithm.step_control.refusal_factor && return tracer, refused(E, TooFast, factor)
    orientation = unwrap(oriented)
    flip_refused(algorithm.orientation_check, tracer.orientation, orientation) && return tracer, refused(E, OrientationFlip)
    tracer, split = split_refused!!(tracer, algorithm.split_check, factor, bound, lands, taken)
    split && return tracer, refused(E, SplitMismatch)
    return tracer, Attempt{E}(NoRefusal, corrected.iterations, corrected.contraction, corrected.first_correction, corrected.error_scale, turn, factor, orientation)
end

"""
Whether `check` refuses the step to the corrected point, `taken` long from the point with step factor `factor`: never
without a check or under its trigger. Else the two half steps (`half_steps!!`) must converge and reach the corrected
point within the check's tolerance, in `Newton`'s `distance`. The corrected point, its tangent in `work.candidate` and
`work.previous` come back as they were.
"""
function split_refused!!(tracer::T, ::Nothing, ::Real, ::Any, ::Bool, ::Real)::Tuple{T, Bool} where {T <: Tracer}
    return tracer, false
end

"""Whether `check` runs a step of step factor `factor` again as two half steps: at its trigger or past it."""
@inline function split_fires(check::HalfStepCheck{E}, factor::E)::Bool where {E <: Real}
    return factor >= check.trigger
end

function split_refused!!(
        tracer::T, check::HalfStepCheck{E}, factor::E, bound::Union{Nothing, Bound{E}}, lands::Bool, taken::E,
    )::Tuple{T, Bool} where {E <: Real, T <: Tracer{E}}
    split_fires(check, factor) || return tracer, false
    @reset tracer.work.split_point = Into(tracer.work.split_point) .= tracer.work.iterate
    @reset tracer.work.split_tangent = Into(tracer.work.split_tangent) .= tracer.work.candidate
    @reset tracer.work.split_previous = Into(tracer.work.split_previous) .= tracer.work.previous
    tracer, halves = half_steps!!(tracer, lands ? bound : nothing, taken)
    # Negated so a NaN distance refuses; a failed half step never reaches the distance.
    refuses = !(halves && correction_length(tracer.algorithm.corrector.distance, tracer, tracer.work.iterate, tracer.work.split_point) <= check.tolerance)
    @reset tracer.work.iterate = Into(tracer.work.iterate) .= tracer.work.split_point
    @reset tracer.work.candidate = Into(tracer.work.candidate) .= tracer.work.split_tangent
    @reset tracer.work.previous = Into(tracer.work.previous) .= tracer.work.split_previous
    return tracer, refuses
end

"""
Two half steps from the point along its tangent, the second ending on the step's own hyperplane (`to_hyperplane!!`):
the first corrects at the half predictor, and the second predicts along that point's tangent to the hyperplane and
corrects there. Whether both converged, the second's point in `work.iterate`. `landing` is the step's `Bound` when it
lands there.
"""
function half_steps!!(tracer::T, landing::Union{Nothing, Bound{E}}, taken::E)::Tuple{T, Bool} where {E <: Real, T <: Tracer{E}}
    @reset tracer.work.predicted = Into(tracer.work.predicted) .= tracer.point .+ (taken / 2) .* tracer.tangent
    @reset tracer.work.row = metric_apply!!(tracer.work.row, tracer.metric, tracer.tangent)
    tracer, corrected = correct!!(tracer)
    corrected.refusal == NoRefusal || return tracer, false
    tracer, finite = evaluate!!(tracer, tracer.work.iterate)
    finite || return tracer, false
    tracer, oriented = tangent_solve!!(tracer, tracer.tangent)
    is_error(oriented) && return tracer, false
    tracer = to_hyperplane!!(tracer, landing, taken)
    tracer, corrected = correct!!(tracer)
    return tracer, corrected.refusal == NoRefusal
end

"""
Predict along `work.candidate` from `work.iterate` to a step's hyperplane, and border with its normal: through the
predictor `taken` long and normal to the tangent, or the coordinate of a `Bound` the step lands on, at its value.
"""
function to_hyperplane!!(tracer::T, ::Nothing, taken::E)::T where {E <: Real, T <: Tracer{E}}
    (; point, tangent, metric, work) = tracer
    offset = Into(work.scratch) .= point .+ taken .* tangent .- work.iterate
    along = metric_inner(metric, tangent, offset) / metric_inner(metric, tangent, work.candidate)
    @reset tracer.work.predicted = Into(work.predicted) .= work.iterate .+ along .* work.candidate
    @reset tracer.work.row = metric_apply!!(work.row, metric, tangent)
    return tracer
end

function to_hyperplane!!(tracer::T, bound::Bound{E}, ::E)::T where {E <: Real, T <: Tracer{E}}
    (; coordinate, value) = bound
    (; work) = tracer
    along = (value - element(work.iterate, coordinate)) / element(work.candidate, coordinate)
    @reset tracer.work.predicted = set_element!!(Into(work.predicted) .= work.iterate .+ along .* work.candidate, coordinate, value)
    @reset tracer.work.row = unit!!(work.row, coordinate)
    return tracer
end

"""Whether `check` refuses a step whose orientation sign went from `before` to `after`."""
flip_refused(::Nothing, ::Int8, ::Int8)::Bool = false
flip_refused(::DeterminantSign, before::Int8, after::Int8)::Bool = after != before

"""
Newton's method from the predictor `x̂` on `[H(x); r ⋅ (x - x̂)] = 0`, `r = work.row`, leaving its last iterate in
`work.iterate`. An `Attempt` with the iterations, the contraction and the first correction once `converged` holds,
else its refusal.
"""
function correct!!(tracer::T)::Tuple{T, Attempt{E}} where {E <: Real, T <: Tracer{E}}
    (; corrector, step_control) = tracer.algorithm
    @reset tracer.work.iterate = Into(tracer.work.iterate) .= tracer.work.predicted
    before, first_correction, contraction = E(NaN32), E(NaN32), E(NaN32)
    for iteration in 1:corrector.max_iterations
        tracer, finite = evaluate!!(tracer, tracer.work.iterate)
        finite || return tracer, refused(E, NotFiniteResidual)
        (; work) = tracer
        bordered = border!!(work.bordered, work.jacobian, work.row)
        rhs = negated_stack!!(work.rhs, work.residual, row_offset(work.row, work.predicted, work.iterate))
        tracer, solved = bordered_solve!!(tracer, bordered, rhs)
        solved || return tracer, refused(E, SingularSystem)
        @reset tracer.work.previous = Into(tracer.work.previous) .= tracer.work.iterate
        @reset tracer.work.iterate = Into(tracer.work.iterate) .= tracer.work.iterate .+ tracer.work.candidate
        correction = correction_length(corrector.distance, tracer, tracer.work.previous, tracer.work.iterate)
        if iteration == 1
            first_correction = correction
        elseif iteration == 2
            contraction = correction / first_correction
        end
        # Negated so a NaN correction counts as diverging.
        if iteration > 1 && !(correction <= corrector.contraction_limit * before)
            return tracer, refused(E, Diverging, step_factor(step_control, contraction, first_correction, E(NaN32)))
        end
        scale = error_scale(corrector.error_estimate, correction, before)
        before = correction
        if corrector.converged(tracer.work.previous, tracer.work.iterate, tracer.work.residual, scale)
            return tracer, Attempt{E}(NoRefusal, iteration, contraction, first_correction, settled_scale(scale), E(NaN32), E(NaN32), Int8(0))
        end
    end
    return tracer, refused(E, NotConverged, step_factor(step_control, contraction, first_correction, E(NaN32)))
end

"""
`scale` at a converged iterate. A `converged` that settles at a scale that is not finite and non-negative throws an
`ArgumentError`: an infinite scale would bound the error by Inf, and the reach would pass any move.
"""
function settled_scale(scale::E)::E where {E <: Real}
    # A throw: the fault lies in `converged`, and a refusal would only shrink the step until `TooManyRefusals`.
    isfinite(scale) && scale >= 0 || throw(ArgumentError(lazy"converged settled at scale $scale; it may settle only at a finite, non-negative scale"))
    return scale
end

"""
The corrector's bound on the error left at `to`, one correction from `from`: `scale` (Newton's, at convergence) times
that correction's length (`correction_length`). Its convergence test holds that product under its tolerance.
"""
function settle_error(tracer::Tracer{E}, scale::E, from::AbstractVector{E}, to::AbstractVector{E})::E where {E <: Real}
    return scale * correction_length(tracer.algorithm.corrector.distance, tracer, from, to)
end

"""
Whether the corrected point lies farther than step control's `reach` times `taken` from its predictor in the step
metric, by more than the corrector's noise. The noise is the move, in Newton's `distance`, within the error bounds of
the start and of the point (`settle_error`, the point's from its convergence `scale`). A NaN move is far.
A move that `distance` measures as zero passes, by intent: in GameTracing such a move leaves every strategy unchanged.
"""
function far_from_predictor(tracer::Tracer{E}, taken::E, scale::E)::Bool where {E <: Real}
    (; algorithm, work) = tracer
    metric_distance(tracer.metric, work.scratch, work.predicted, work.iterate) <= algorithm.step_control.reach * taken && return false
    noise = tracer.settle_error + settle_error(tracer, scale, work.previous, work.iterate)
    return !(correction_length(algorithm.corrector.distance, tracer, work.predicted, work.iterate) <= noise)
end

"""A correction's length from `from` to `to`: `tracer`'s step metric norm, or the corrector's own `distance`."""
function correction_length(::MetricDistance, tracer::Tracer{E}, from::AbstractVector{E}, to::AbstractVector{E})::E where {E <: Real}
    return metric_distance(tracer.metric, tracer.work.scratch, from, to)
end

function correction_length(distance::Any, ::Tracer{E}, from::AbstractVector{E}, to::AbstractVector{E})::E where {E <: Real}
    return E(distance(from, to))
end

"""
The tangent at the point whose Jacobian `work.jacobian` holds, into `work.candidate`: the solution of
`[J; (M previous)ᵀ] t = [0; 1]` with `M` the turn metric, so `⟨t, previous⟩` is one, positive. The orientation sign
at that tangent (`orientation_sign!`), or `none` when the solve fails.
"""
function tangent_solve!!(tracer::T, previous::AbstractVector{E})::Tuple{T, Option{Int8}} where {E <: Real, T <: Tracer{E}}
    @reset tracer.work.row = metric_apply!!(tracer.work.row, tracer.turn_metric, previous)
    (; work) = tracer
    tracer, solved = bordered_solve!!(tracer, border!!(work.bordered, work.jacobian, work.row), unit!!(work.rhs, lastindex(work.rhs)))
    solved || return tracer, none(Int8)
    return tracer, some(orientation_sign!(tracer.linear, tracer.algorithm.orientation_check, tracer.work))
end

"""Solve `bordered \\ rhs` into `work.candidate`; whether the solve succeeded with a finite solution."""
function bordered_solve!!(tracer::T, bordered::AbstractMatrix, rhs::AbstractVector)::Tuple{T, Bool} where {T <: Tracer}
    linear, candidate, solved = linear_solve!!(tracer.linear, bordered, rhs, tracer.work.candidate)
    @reset tracer.linear = linear
    @reset tracer.work.candidate = candidate
    return tracer, solved
end

"""
Whether `predicted` moves no coordinate of `point` by more than `ulps` ulps of its own size: the per-coordinate test
`|Δxᵢ| ≤ ulps · eps · |xᵢ|`, so coordinates of different scales each count, and any move of a zero coordinate
resolves.
"""
function unresolved(point::AbstractVector{E}, predicted::AbstractVector{E}, ulps::Integer)::Bool where {E <: Real}
    tolerance = ulps * eps(E)
    # A NaN move counts as resolved, as the comparison is false for it.
    within = elementwise((from, to) -> abs(to - from) <= tolerance * abs(from), point, predicted)
    return mapreduce(identity, &, within; init = true)
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
Move `tracer` to `attempt`'s corrected point with the new tangent and orientation, freeze both metrics there and size
the next step from the predictor's length `taken`.
"""
function accept_point!!(tracer::T, attempt::Attempt{E}, taken::E)::T where {E <: Real, T <: Tracer{E}}
    (; max_step_growth, max_step) = tracer.algorithm.step_control
    @reset tracer.point = Into(tracer.point) .= tracer.work.iterate
    @reset tracer.orientation = attempt.orientation
    @reset tracer.metric = freeze_metric!!(tracer.metric, tracer.point)
    @reset tracer.turn_metric = freeze_metric!!(tracer.turn_metric, tracer.point)
    # Measured after the freeze, as the next reach check measures. Only `correct!!` writes `previous`: the iterate before.
    @reset tracer.settle_error = settle_error(tracer, attempt.error_scale, tracer.work.previous, tracer.point)
    @reset tracer.tangent = Into(tracer.tangent) .= tracer.work.candidate ./ metric_norm(tracer.metric, tracer.work.candidate)
    @reset tracer.step_length = min(taken / max(attempt.factor, inv(max_step_growth)), max_step)
    @reset tracer.steps = tracer.steps + 1
    return tracer
end
