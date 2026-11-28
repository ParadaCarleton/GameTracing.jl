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
A trace's end short of its caller's goal, a Moshi sum in the step's element type `E`. `last_refusal` is the last
`Refusal` before the end (`NoRefusal` when none came first), and `refusals` counts the failed step's refusals.

- `Unresolved(last_refusal, refusals)`: a predictor below the float type's resolution. The returned tracer holds the
  step it judged: the controller's step, or the retry that did not shrink below the predictor it refuses.
- `TooManyRefusals(last_refusal, step_length, refusals)`: more refusals in one step than a set `max_refusals`. `step_length`
  is the last refused predictor, cut at a `Bound`.
- `NotFiniteStart()`: `H` or its Jacobian is not finite at the start.
- `SingularStart()`: no tangent at the start along `direction`.
- `TooManySteps()`: a `land!!` past a set step budget.

A variant with no field of type `E` takes it explicitly: `StepFailure.TooManySteps{E}()`.
"""
# No `E <: Real` bound: Moshi's `@derive Eq` of a sum that holds a `StepFailure` builds its type at `E = Any`.
@data StepFailure{E} begin
    struct Unresolved
        last_refusal::Refusal
        refusals::Int
    end
    struct TooManyRefusals
        last_refusal::Refusal
        step_length::E
        refusals::Int
    end
    NotFiniteStart()
    SingularStart()
    TooManySteps()
end

@derive StepFailure[Show, Eq, Hash]

# Moshi declares each variant `public`, but not `Type`, the sum's own name.
@eval StepFailure $(Expr(:public, :Type))

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
    Tracer

The state of a trace: the last accepted `point`, its unit `tangent` in the step metric, the next `step_length`, the
accepted `steps`, the `orientation` sign at `point` (zero without an orientation check), the corrector's estimate of
the error left at `point` (`settle_error`; zero at the start), the step and turn metrics frozen at `point`, the work
arrays (`Workspace`) and the linear solver's state. `init` builds one and `step!!` advances it.

Its arrays take the start's array type: `similar` arrays on the start's device for a mutable start, static arrays for a
static one. A step changes mutable arrays in place, so use only the tracer it returns. Read its fields; do not write
them. A start in group memory gives a tracer of `GroupHandle`s: pass its `GroupMemory` to each step beside it
(`step!!(memory, tracer)`), and read its arrays through it, as in `group_view(memory, tracer).point`. Its `memory_tag`
names that memory (`memory_tag`), so a step in another memory throws. The tag is built from addresses: a memory
freed and allocated again at those addresses passes the check, though the tracer's arrays in it are lost.
"""
struct Tracer{E <: Real, V <: StoredVector{E}, P <: CurveProblem, A <: PseudoArclength, M, T, W <: Workspace, L}
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
    memory_tag::UInt
end

"""
    init(problem, algorithm; first_step)

A `Tracer` at `problem.start`, its tangent oriented along `problem.direction`, its first step `first_step` long in the
step metric and at most the step control's `max_step`; or the `StepFailure` (`NotFiniteStart`, `SingularStart`) that
stops it there. `first_step` must be positive and finite. A start in group memory gives a tracer of that memory.
"""
function CommonSolve.init(
        problem::CurveProblem{S, V},
        algorithm::PseudoArclength;
        first_step::Real,
    )::Result{<:Tracer{E}, StepFailure.Type{E}} where {S, E <: Real, V <: AbstractVector{E}}  # concrete in `started`
    # Checked in `E`, where a tiny or huge `first_step` rounds to 0 or Inf.
    valid_first_step(E(first_step)) || throw(ArgumentError(lazy"first_step must be positive and finite in $E; got $first_step"))
    # The step control computes in its own type: a shrink below 1 there can round back to the same step in `E`.
    algorithm.step_control isa StepControl{E} ||
        throw(ArgumentError(lazy"the step control's element type must be the start's $E; got $(typeof(algorithm.step_control))"))
    algorithm.split_check isa Union{Nothing, HalfStepCheck{E}} ||
        throw(ArgumentError(lazy"the split check's element type must be the start's $E; got $(typeof(algorithm.split_check))"))
    (; start, direction) = problem
    length(direction) == length(start) || throw(DimensionMismatch("start and direction differ in length"))
    memory = group_memory(start)
    # Only a direction in group memory names one; the trace keeps only the start's.
    group_memory(direction) isa GroupMemory && group_memory(direction) !== memory &&
        throw(ArgumentError("a direction in group memory lies in its start's memory"))
    work = workspace(start, E)
    tracer = Tracer(
        stored(problem), algorithm, stored(@into!!(zero_point(start, E) .= start)), stored(zero_point(start, E)),
        min(E(first_step), algorithm.step_control.max_step), 0, Int8(0), zero(E), stored(frozen_metric(algorithm.metric, start)),
        stored(frozen_metric(algorithm.turn_metric, start)), stored(work), linear_state(algorithm.linear_solver, work),
        memory_tag(memory),
    )
    return started(memory, tracer)
end

"""Throw unless `memory` is the memory that `tracer` started in, by its tag (`memory_tag`)."""
function check_memory(memory::TraceMemory, tracer::Tracer)::Nothing
    memory_tag(memory) == tracer.memory_tag || throw(ArgumentError("a trace steps only in the group memory it started in"))
    return nothing
end

"""
    group_view(memory, tracer)

`tracer` with each array viewed through `memory`, the `GroupMemory` it started in, so its fields read as arrays: as in
`group_view(memory, tracer).point`. Throws for another memory. Read the view; step the tracer itself.
"""
function group_view(memory::TraceMemory, tracer::Tracer)::Tracer
    check_memory(memory, tracer)
    viewed = Base.Fix1(stored_view, memory)
    return Tracer(
        viewed(tracer.problem), tracer.algorithm, viewed(tracer.point), viewed(tracer.tangent), tracer.step_length, tracer.steps,
        tracer.orientation, tracer.settle_error, viewed(tracer.metric), viewed(tracer.turn_metric), viewed(tracer.work), tracer.linear,
        tracer.memory_tag,
    )
end

"""Whether `step` can open a trace: positive and finite."""
valid_first_step(step::Real)::Bool = 0 < step < Inf

"""`tracer` with its unit tangent at its start, where `init` froze its metrics, or the `StepFailure` there."""
function started(memory::TraceMemory, tracer::T)::Result{T, StepFailure.Type{E}} where {E <: Real, T <: Tracer{E}}
    viewed = Base.Fix1(stored_view, memory)
    tracer, finite = evaluate!!(memory, tracer, viewed(tracer.point))
    finite || return Err(StepFailure.NotFiniteStart{E}())
    tracer, oriented = tangent_solve!!(memory, tracer, viewed(tracer.problem.direction))
    is_error(oriented) && return Err(StepFailure.SingularStart{E}())
    candidate = viewed(tracer.work.candidate)
    @reset tracer.tangent = stored(@into!!(viewed(tracer.tangent) .= candidate ./ metric_norm(viewed(tracer.metric), candidate)))
    @reset tracer.orientation = unwrap(oriented)
    return Ok(tracer)
end

"""
`tracer` with `H` and its Jacobian at `point` in its work arrays (`CurveProblem`'s `system!!`), and whether both are
finite. A system that returns arrays of other types than it got throws an `ArgumentError`.
"""
function evaluate!!(memory::TraceMemory, tracer::T, point::AbstractVector)::Tuple{T, Bool} where {T <: Tracer}
    given_residual, given_jacobian = stored_view(memory, tracer.work.residual), stored_view(memory, tracer.work.jacobian)
    residual, jacobian = tracer.problem.system!!(given_residual, given_jacobian, point)
    # Folded away when the types match, as they must for the tracer to keep its type.
    residual isa typeof(given_residual) && jacobian isa typeof(given_jacobian) || throw(
        ArgumentError(lazy"system!! must return its residual and Jacobian as a $(typeof(given_residual)) and a $(typeof(given_jacobian))")
    )
    @reset tracer.work.residual = stored(residual)
    @reset tracer.work.jacobian = stored(jacobian)
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
    tracer, outcome = step!!([memory, ]tracer[, bound])

Advance `tracer` by one accepted point. Return the advanced tracer and the step's `AcceptedStep`, or the `StepFailure`
that ends the trace. Use only the returned tracer: the step changes the old one's mutable arrays in place. A refused
attempt shrinks the predictor it took, cut or not, and tries again. With a `Bound`, the step stays on the point's side
of the bound or stops on it. A predictor that reaches the bound is cut where its line meets it, and Newton's method
corrects at the bound's value, under every test of a step. A corrected point past the bound refuses as `PastBound`. The
step sees the bound only at its predictor and corrected point: a curve that passes the bound and comes back within one
step goes unseen. The resolution check judges the controller's step before the cut, as a landing always moves the
point. A retry no shorter than the predictor it refuses fails `Unresolved`. The point must lie off the bound.
A tracer in group memory takes its `GroupMemory` first; any other takes none.
"""
function step!!(
        memory::TraceMemory,
        tracer::T,
        bound::Union{Nothing, Bound{E}} = nothing,
    )::Tuple{T, Result{AcceptedStep{E}, StepFailure.Type{E}}} where {E <: Real, T <: Tracer{E}}
    (; algorithm) = tracer
    check_memory(memory, tracer)
    viewed = Base.Fix1(stored_view, memory)
    on_bound(bound, viewed(tracer.point)) && throw(ArgumentError(lazy"the point lies on the bound $bound"))
    refusals = 0
    last_refusal = NoRefusal
    taken = tracer.step_length
    shorter = true
    while shorter
        point, tangent = viewed(tracer.point), viewed(tracer.tangent)
        @reset tracer.work.predicted = stored(@into!!(viewed(tracer.work.predicted) .= point .+ tracer.step_length .* tangent))
        # Checked before the cut: a cut predictor lands on the bound, so its length shows no stagnation.
        if unresolved(point, viewed(tracer.work.predicted), algorithm.min_step_ulps)
            return tracer, Err(StepFailure.Unresolved{E}(last_refusal, refusals))
        end
        lands = reaches(bound, point, viewed(tracer.work.predicted))
        tracer, taken = lands ? cut_predictor!!(memory, tracer, bound) : (tracer, tracer.step_length)
        tracer, attempt = attempt_step!!(memory, tracer, bound, lands, taken)
        if attempt.refusal == NoRefusal
            tracer = accept_point!!(memory, tracer, attempt, taken)
            return tracer, Ok(AcceptedStep(taken, attempt.iterations, refusals, attempt.contraction, attempt.first_correction, attempt.turn))
        end
        last_refusal = attempt.refusal
        # This refusal is one past a budget the earlier refusals reached.
        limit_reached(algorithm.max_refusals, refusals) && return tracer, Err(StepFailure.TooManyRefusals(last_refusal, taken, refusals + 1))
        refusals += 1
        # The retry shrinks the predictor taken, cut or not; an infinite factor shrinks like an unmeasured one.
        shrink = inv(algorithm.step_control.shrink)
        @reset tracer.step_length = taken / ifelse(isfinite(attempt.factor), max(attempt.factor, shrink), shrink)
        # A retry that rounds back to the refused length would repeat it: the step is below resolution.
        shorter = tracer.step_length < taken
    end
    return tracer, Err(StepFailure.Unresolved{E}(last_refusal, refusals))
end

function step!!(tracer::T, bound::Union{Nothing, Bound{E}} = nothing)::Tuple{T, Result{AcceptedStep{E}, StepFailure.Type{E}}} where {E <: Real, T <: Tracer{E}}
    return step!!(nothing, tracer, bound)
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
function cut_predictor!!(memory::TraceMemory, tracer::T, bound::Bound{E})::Tuple{T, E} where {E <: Real, T <: Tracer{E}}
    (; coordinate, value) = bound
    viewed = Base.Fix1(stored_view, memory)
    point, tangent = viewed(tracer.point), viewed(tracer.tangent)
    from = element(point, coordinate)
    fraction = (value - from) / (element(viewed(tracer.work.predicted), coordinate) - from)
    taken = fraction * tracer.step_length
    predicted = @into!! viewed(tracer.work.predicted) .= point .+ taken .* tangent
    @reset tracer.work.predicted = stored(set_element!!(predicted, coordinate, value))
    @reset tracer.work.row = stored(unit!!(viewed(tracer.work.row), coordinate))
    return tracer, taken
end

"""
After a correction against `bound`: a landing (`lands`) keeps the bound's value exactly; else whether the corrected
point went past the bound.
"""
function past_bound!!(::TraceMemory, tracer::T, ::Nothing, ::Bool)::Tuple{T, Bool} where {T <: Tracer}
    return tracer, false
end

function past_bound!!(memory::TraceMemory, tracer::T, bound::Bound{E}, lands::Bool)::Tuple{T, Bool} where {E <: Real, T <: Tracer{E}}
    viewed = Base.Fix1(stored_view, memory)
    if lands
        # The bordered row holds the coordinate at the value; this drops the solves' rounding.
        @reset tracer.work.iterate = stored(set_element!!(viewed(tracer.work.iterate), bound.coordinate, bound.value))
        return tracer, false
    end
    iterate = viewed(tracer.work.iterate)
    return tracer, reaches(bound, viewed(tracer.point), iterate) && !on_bound(bound, iterate)
end

"""
Correct the predictor, `taken` long, then test the corrected point: its distance from the predictor, `bound`,
`accept`, a finite system there, its tangent, step control's factor, its orientation sign and the split check. A
landing (`lands`) holds its row from `cut_predictor!!`; any other attempt borders with the tangent.
"""
function attempt_step!!(
        memory::TraceMemory, tracer::T, bound::Union{Nothing, Bound{E}}, lands::Bool, taken::E,
    )::Tuple{T, Attempt{E}} where {E <: Real, T <: Tracer{E}}
    (; algorithm) = tracer
    viewed = Base.Fix1(stored_view, memory)
    if !lands
        @reset tracer.work.row = stored(metric_apply!!(viewed(tracer.work.row), viewed(tracer.metric), viewed(tracer.tangent)))
    end
    tracer, corrected = correct!!(memory, tracer)
    corrected.refusal == NoRefusal || return tracer, corrected
    far_from_predictor(memory, tracer, taken) && return tracer, refused(E, FarFromPredictor)
    tracer, past = past_bound!!(memory, tracer, bound, lands)
    past && return tracer, refused(E, PastBound)
    algorithm.accept(viewed(tracer.point), viewed(tracer.work.predicted), viewed(tracer.work.iterate)) || return tracer, refused(E, AcceptRefused)
    tracer, finite = evaluate!!(memory, tracer, viewed(tracer.work.iterate))
    finite || return tracer, refused(E, NotFiniteResidual)
    tracer, oriented = tangent_solve!!(memory, tracer, viewed(tracer.tangent))
    is_error(oriented) && return tracer, refused(E, SingularSystem)
    turn = metric_angle(viewed(tracer.turn_metric), viewed(tracer.work.scratch), viewed(tracer.tangent), viewed(tracer.work.candidate))
    factor = step_factor(algorithm.step_control, corrected.contraction, corrected.first_correction, turn)
    factor > algorithm.step_control.refusal_factor && return tracer, refused(E, TooFast, factor)
    orientation = unwrap(oriented)
    flip_refused(algorithm.orientation_check, tracer.orientation, orientation) && return tracer, refused(E, OrientationFlip)
    tracer, split = split_refused!!(memory, tracer, algorithm.split_check, factor, bound, lands, taken)
    split && return tracer, refused(E, SplitMismatch)
    return tracer, Attempt{E}(NoRefusal, corrected.iterations, corrected.contraction, corrected.first_correction, turn, factor, orientation)
end

"""
Whether `check` refuses the step to the corrected point, `taken` long from the point with step factor `factor`: never
without a check or under its trigger. Else the two half steps (`half_steps!!`) must converge and reach the corrected
point within the check's tolerance, in `Newton`'s `distance`. The corrected point, its tangent in `work.candidate` and
`work.previous` come back as they were.
"""
function split_refused!!(::TraceMemory, tracer::T, ::Nothing, ::Real, ::Any, ::Bool, ::Real)::Tuple{T, Bool} where {T <: Tracer}
    return tracer, false
end

"""Whether `check` runs a step of step factor `factor` again as two half steps: at its trigger or past it."""
@inline function split_fires(check::HalfStepCheck{E}, factor::E)::Bool where {E <: Real}
    return factor >= check.trigger
end

function split_refused!!(
        memory::TraceMemory, tracer::T, check::HalfStepCheck{E}, factor::E, bound::Union{Nothing, Bound{E}}, lands::Bool, taken::E,
    )::Tuple{T, Bool} where {E <: Real, T <: Tracer{E}}
    split_fires(check, factor) || return tracer, false
    viewed = Base.Fix1(stored_view, memory)
    @reset tracer.work.split_point = stored(@into!!(viewed(tracer.work.split_point) .= viewed(tracer.work.iterate)))
    @reset tracer.work.split_tangent = stored(@into!!(viewed(tracer.work.split_tangent) .= viewed(tracer.work.candidate)))
    @reset tracer.work.split_previous = stored(@into!!(viewed(tracer.work.split_previous) .= viewed(tracer.work.previous)))
    tracer, halves = half_steps!!(memory, tracer, lands ? bound : nothing, taken)
    # Negated so a NaN distance refuses; a failed half step never reaches the distance.
    distance = tracer.algorithm.corrector.distance
    refuses = !(halves && correction_length(memory, distance, tracer, viewed(tracer.work.iterate), viewed(tracer.work.split_point)) <= check.tolerance)
    @reset tracer.work.iterate = stored(@into!!(viewed(tracer.work.iterate) .= viewed(tracer.work.split_point)))
    @reset tracer.work.candidate = stored(@into!!(viewed(tracer.work.candidate) .= viewed(tracer.work.split_tangent)))
    @reset tracer.work.previous = stored(@into!!(viewed(tracer.work.previous) .= viewed(tracer.work.split_previous)))
    return tracer, refuses
end

"""
Two half steps from the point along its tangent, the second ending on the step's own hyperplane (`to_hyperplane!!`):
the first corrects at the half predictor, and the second predicts along that point's tangent to the hyperplane and
corrects there. Whether both converged, the second's point in `work.iterate`. `landing` is the step's `Bound` when it
lands there.
"""
function half_steps!!(memory::TraceMemory, tracer::T, landing::Union{Nothing, Bound{E}}, taken::E)::Tuple{T, Bool} where {E <: Real, T <: Tracer{E}}
    viewed = Base.Fix1(stored_view, memory)
    point, tangent, metric = viewed(tracer.point), viewed(tracer.tangent), viewed(tracer.metric)
    @reset tracer.work.predicted = stored(@into!!(viewed(tracer.work.predicted) .= point .+ (taken / 2) .* tangent))
    @reset tracer.work.row = stored(metric_apply!!(viewed(tracer.work.row), metric, tangent))
    tracer, corrected = correct!!(memory, tracer)
    corrected.refusal == NoRefusal || return tracer, false
    tracer, finite = evaluate!!(memory, tracer, viewed(tracer.work.iterate))
    finite || return tracer, false
    tracer, oriented = tangent_solve!!(memory, tracer, tangent)
    is_error(oriented) && return tracer, false
    tracer = to_hyperplane!!(memory, tracer, landing, taken)
    tracer, corrected = correct!!(memory, tracer)
    return tracer, corrected.refusal == NoRefusal
end

"""
Predict along `work.candidate` from `work.iterate` to a step's hyperplane, and border with its normal: through the
predictor `taken` long and normal to the tangent, or the coordinate of a `Bound` the step lands on, at its value.
"""
function to_hyperplane!!(memory::TraceMemory, tracer::T, ::Nothing, taken::E)::T where {E <: Real, T <: Tracer{E}}
    viewed = Base.Fix1(stored_view, memory)
    point, tangent, metric = viewed(tracer.point), viewed(tracer.tangent), viewed(tracer.metric)
    iterate, candidate = viewed(tracer.work.iterate), viewed(tracer.work.candidate)
    offset = @into!! viewed(tracer.work.scratch) .= point .+ taken .* tangent .- iterate
    along = metric_inner(metric, tangent, offset) / metric_inner(metric, tangent, candidate)
    @reset tracer.work.predicted = stored(@into!!(viewed(tracer.work.predicted) .= iterate .+ along .* candidate))
    @reset tracer.work.row = stored(metric_apply!!(viewed(tracer.work.row), metric, tangent))
    return tracer
end

function to_hyperplane!!(memory::TraceMemory, tracer::T, bound::Bound{E}, ::E)::T where {E <: Real, T <: Tracer{E}}
    (; coordinate, value) = bound
    viewed = Base.Fix1(stored_view, memory)
    iterate, candidate = viewed(tracer.work.iterate), viewed(tracer.work.candidate)
    along = (value - element(iterate, coordinate)) / element(candidate, coordinate)
    predicted = @into!! viewed(tracer.work.predicted) .= iterate .+ along .* candidate
    @reset tracer.work.predicted = stored(set_element!!(predicted, coordinate, value))
    @reset tracer.work.row = stored(unit!!(viewed(tracer.work.row), coordinate))
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
function correct!!(memory::TraceMemory, tracer::T)::Tuple{T, Attempt{E}} where {E <: Real, T <: Tracer{E}}
    (; corrector, step_control) = tracer.algorithm
    viewed = Base.Fix1(stored_view, memory)
    @reset tracer.work.iterate = stored(@into!!(viewed(tracer.work.iterate) .= viewed(tracer.work.predicted)))
    before, first_correction, contraction = E(NaN32), E(NaN32), E(NaN32)
    for iteration in 1:corrector.max_iterations
        tracer, finite = evaluate!!(memory, tracer, viewed(tracer.work.iterate))
        finite || return tracer, refused(E, NotFiniteResidual)
        work = viewed(tracer.work)
        bordered = border!!(work.bordered, work.jacobian, work.row)
        rhs = negated_stack!!(work.rhs, work.residual, row_offset(work.row, work.predicted, work.iterate))
        tracer, solved = bordered_solve!!(memory, tracer, stored(bordered), stored(rhs))
        solved || return tracer, refused(E, SingularSystem)
        @reset tracer.work.previous = stored(@into!!(viewed(tracer.work.previous) .= viewed(tracer.work.iterate)))
        iterate = viewed(tracer.work.iterate)
        @reset tracer.work.iterate = stored(@into!!(iterate .= iterate .+ viewed(tracer.work.candidate)))
        previous, iterate = viewed(tracer.work.previous), viewed(tracer.work.iterate)
        correction = correction_length(memory, corrector.distance, tracer, previous, iterate)
        if iteration == 1
            first_correction = correction
        elseif iteration == 2
            contraction = correction / first_correction
        end
        # Negated so a NaN correction counts as diverging.
        if iteration > 1 && !(correction <= corrector.contraction_limit * before)
            return tracer, refused(E, Diverging, step_factor(step_control, contraction, first_correction, E(NaN32)))
        end
        if corrector.converged(previous, iterate, viewed(tracer.work.residual))
            return tracer, Attempt{E}(NoRefusal, iteration, contraction, first_correction, E(NaN32), E(NaN32), Int8(0))
        end
        before = correction
    end
    return tracer, refused(E, NotConverged, step_factor(step_control, contraction, first_correction, E(NaN32)))
end

"""
The corrector's estimate of the error left at `to`, one correction from `from`: that correction's length
(`correction_length`), which its convergence test holds under its tolerance (`Newton`, `contraction_limit`).
"""
function settle_error(memory::TraceMemory, tracer::Tracer{E}, from::AbstractVector{E}, to::AbstractVector{E})::E where {E <: Real}
    return correction_length(memory, tracer.algorithm.corrector.distance, tracer, from, to)
end

"""
Whether the corrected point lies farther than step control's `reach` times `taken` from its predictor in the step
metric, by more than the corrector's noise. The noise is the move, in Newton's `distance`, within the error estimates of
the start and of the point (`settle_error`). A NaN move is far.
A move that `distance` measures as zero passes, by intent: in GameTracing such a move leaves every strategy unchanged.
"""
function far_from_predictor(memory::TraceMemory, tracer::Tracer{E}, taken::E)::Bool where {E <: Real}
    (; algorithm) = tracer
    viewed = Base.Fix1(stored_view, memory)
    predicted, iterate = viewed(tracer.work.predicted), viewed(tracer.work.iterate)
    metric_distance(viewed(tracer.metric), viewed(tracer.work.scratch), predicted, iterate) <= algorithm.step_control.reach * taken && return false
    noise = tracer.settle_error + settle_error(memory, tracer, viewed(tracer.work.previous), iterate)
    return !(correction_length(memory, algorithm.corrector.distance, tracer, predicted, iterate) <= noise)
end

"""A correction's length from `from` to `to`: `tracer`'s step metric norm, or the corrector's own `distance`."""
function correction_length(memory::TraceMemory, ::MetricDistance, tracer::Tracer{E}, from::AbstractVector{E}, to::AbstractVector{E})::E where {E <: Real}
    return metric_distance(stored_view(memory, tracer.metric), stored_view(memory, tracer.work.scratch), from, to)
end

function correction_length(::TraceMemory, distance::Any, ::Tracer{E}, from::AbstractVector{E}, to::AbstractVector{E})::E where {E <: Real}
    return E(distance(from, to))
end

"""
The tangent at the point whose Jacobian `work.jacobian` holds, into `work.candidate`: the solution of
`[J; (M previous)ᵀ] t = [0; 1]` with `M` the turn metric, so `⟨t, previous⟩` is one, positive. The orientation sign
at that tangent (`orientation_sign!`), or `none` when the solve fails.
"""
function tangent_solve!!(memory::TraceMemory, tracer::T, previous::AbstractVector{E})::Tuple{T, Option{Int8}} where {E <: Real, T <: Tracer{E}}
    viewed = Base.Fix1(stored_view, memory)
    @reset tracer.work.row = stored(metric_apply!!(viewed(tracer.work.row), viewed(tracer.turn_metric), previous))
    work = viewed(tracer.work)
    bordered, rhs = border!!(work.bordered, work.jacobian, work.row), unit!!(work.rhs, lastindex(work.rhs))
    tracer, solved = bordered_solve!!(memory, tracer, stored(bordered), stored(rhs))
    solved || return tracer, none(Int8)
    return tracer, some(orientation_sign!(tracer.linear, tracer.algorithm.orientation_check, viewed(tracer.work)))
end

"""Solve `bordered \\ rhs`, both as stored (`stored`), into `work.candidate`; whether it succeeded with a finite solution."""
function bordered_solve!!(memory::TraceMemory, tracer::T, bordered::StoredMatrix, rhs::StoredVector)::Tuple{T, Bool} where {T <: Tracer}
    viewed = Base.Fix1(stored_view, memory)
    linear, candidate, solved = linear_solve!!(tracer.linear, viewed(bordered), viewed(rhs), viewed(tracer.work.candidate))
    @reset tracer.linear = linear
    @reset tracer.work.candidate = stored(candidate)
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
function accept_point!!(memory::TraceMemory, tracer::T, attempt::Attempt{E}, taken::E)::T where {E <: Real, T <: Tracer{E}}
    (; max_step_growth, max_step) = tracer.algorithm.step_control
    viewed = Base.Fix1(stored_view, memory)
    @reset tracer.point = stored(@into!!(viewed(tracer.point) .= viewed(tracer.work.iterate)))
    @reset tracer.orientation = attempt.orientation
    @reset tracer.metric = stored(freeze_metric!!(viewed(tracer.metric), viewed(tracer.point)))
    @reset tracer.turn_metric = stored(freeze_metric!!(viewed(tracer.turn_metric), viewed(tracer.point)))
    # Measured after the freeze, as the next reach check measures. Only `correct!!` writes `previous`: the iterate before.
    @reset tracer.settle_error = settle_error(memory, tracer, viewed(tracer.work.previous), viewed(tracer.point))
    candidate = viewed(tracer.work.candidate)
    @reset tracer.tangent = stored(@into!!(viewed(tracer.tangent) .= candidate ./ metric_norm(viewed(tracer.metric), candidate)))
    @reset tracer.step_length = min(taken / max(attempt.factor, inv(max_step_growth)), max_step)
    @reset tracer.steps = tracer.steps + 1
    return tracer
end
