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
a changed sign (`OrientationFlip`), the corrected point went past the step's `Bound` (`PastBound`), or it lies
farther from its predictor than step control's `reach` (`FarFromPredictor`). `NoRefusal` marks an accepted attempt.
"""
@enum Refusal NoRefusal NotConverged Diverging NotFiniteResidual SingularSystem AcceptRefused TooFast OrientationFlip PastBound FarFromPredictor

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
end

"""
    Tracer

The state of a trace: the last accepted `point`, its unit `tangent` in the step metric, the next `step_length`,
the accepted `steps`, the `orientation` sign at `point` (zero without an orientation check), `settle_error`, the
corrector's bound on the error left at `point` (`settle_error`; zero at the start), the step and turn metrics frozen at
`point` (`frozen_metric`), and work arrays; `row` is the border row of the next bordered system. `init` builds one and
`step!` advances it. Read its fields; do not write them.
"""
mutable struct Tracer{E <: Real, P <: CurveProblem, A <: PseudoArclength, C, M, T}
    const problem::P
    const algorithm::A
    const point::Vector{E}
    const tangent::Vector{E}
    step_length::E
    steps::Int
    orientation::Int8
    settle_error::E
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
    # The step control computes in its own type: a shrink below 1 there can round back to the same step in `E`.
    algorithm.step_control isa StepControl{E} ||
        throw(ArgumentError("the step control's element type must be the start's $E; got $(typeof(algorithm.step_control))"))
    size = length(problem.start)
    length(problem.direction) == size || throw(DimensionMismatch("start and direction differ in length"))
    bordered = zeros(E, size, size)
    rhs = zeros(E, size)
    # A failed solve is a refusal (`SingularSystem`), so LinearSolve's own warning would only repeat it.
    linear = LinearSolve.init(LinearSolve.LinearProblem(bordered, rhs), algorithm.linear_solver; verbose = false)
    tracer = Tracer(
        problem, algorithm, collect(problem.start), zeros(E, size), min(E(first_step), algorithm.step_control.max_step), 0, Int8(0), zero(E),
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
    tracer.orientation = unwrap(oriented)
    return Ok(tracer)
end

"""
    Bound(coordinate, value)

A value of one coordinate that a step (`step!`) stops at: `x[coordinate] = value`.
"""
struct Bound{E <: Real}
    coordinate::Int
    value::E
end

"""
    step!(tracer[, bound])

Advance `tracer` by one accepted point and return its `AcceptedStep`, or the `StepFailure` that ends the trace. A
refused attempt shrinks the predictor it took, cut or not, and tries again. With a `Bound`, the step stays on the
point's side of the bound or stops on it. A predictor that reaches the bound is cut where its line meets it, and
Newton's method corrects at the bound's value, under every test of a step. A corrected point past the bound refuses as
`PastBound`. The step sees the bound only at its predictor and corrected point: a curve that passes the bound and
comes back within one step goes unseen. The resolution check judges the controller's step before the cut, as a landing
always moves the point. A retry no shorter than the predictor it refuses fails `Unresolved`. The point must lie off the
bound.
"""
function CommonSolve.step!(
        tracer::Tracer{E},
        bound::Union{Nothing, Bound{E}} = nothing,
    )::Result{AcceptedStep{E}, StepFailure{E}} where {E <: Real}
    (; algorithm) = tracer
    if on_bound(bound, tracer.point)
        throw(ArgumentError("the point lies on the bound $bound"))
    end
    refusals = 0
    last_refusal = NoRefusal
    taken = tracer.step_length
    while true
        tracer.predicted .= tracer.point .+ tracer.step_length .* tracer.tangent
        # Checked before the cut: a cut predictor lands on the bound, so its length shows no stagnation.
        if unresolved(tracer.point, tracer.predicted, algorithm.min_step_ulps)
            return Err(StepFailure(Unresolved, last_refusal, tracer.step_length, refusals))
        end
        lands = reaches(bound, tracer.point, tracer.predicted)
        taken = lands ? cut_predictor!(tracer, bound) : tracer.step_length
        attempt = attempt_step!(tracer, bound, lands, taken)
        if attempt.refusal == NoRefusal
            accept_point!(tracer, attempt, taken)
            return Ok(AcceptedStep(taken, attempt.iterations, refusals, attempt.contraction, attempt.first_correction, attempt.turn))
        end
        refusals += 1
        last_refusal = attempt.refusal
        if refusals > algorithm.max_refusals
            break
        end
        # The retry shrinks the predictor taken, cut or not; an infinite factor shrinks like an unmeasured one.
        shrink = inv(algorithm.step_control.shrink)
        tracer.step_length = taken / ifelse(isfinite(attempt.factor), max(attempt.factor, shrink), shrink)
        # A retry that rounds back to the refused length would repeat it: the step is below resolution.
        if !(tracer.step_length < taken)
            return Err(StepFailure(Unresolved, last_refusal, tracer.step_length, refusals))
        end
    end
    return Err(StepFailure(TooManyRefusals, last_refusal, taken, refusals))
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
    return point[bound.coordinate] == bound.value
end

"""Whether `to` lies on `bound` or on its other side from `from`; never without one."""
reaches(::Nothing, ::AbstractVector, ::AbstractVector)::Bool = false
function reaches(bound::Bound{E}, from::AbstractVector{E}, to::AbstractVector{E})::Bool where {E <: Real}
    return signbit(from[bound.coordinate] - bound.value) != signbit(to[bound.coordinate] - bound.value) || on_bound(bound, to)
end

"""
Cut `tracer.predicted` where its line from `tracer.point` meets `bound`, and border the corrector's system with the
bound's coordinate. The cut predictor's length in the step metric.
"""
function cut_predictor!(tracer::Tracer{E}, bound::Bound{E})::E where {E <: Real}
    (; coordinate, value) = bound
    fraction = (value - tracer.point[coordinate]) / (tracer.predicted[coordinate] - tracer.point[coordinate])
    tracer.predicted .= tracer.point .+ (fraction * tracer.step_length) .* tracer.tangent
    tracer.predicted[coordinate] = value
    fill!(tracer.row, zero(E))
    tracer.row[coordinate] = one(E)
    return fraction * tracer.step_length
end

"""
After a correction against `bound`: a landing (`lands`) keeps the bound's value exactly; else whether the corrected
point went past the bound.
"""
past_bound!(::Tracer, ::Nothing, ::Bool)::Bool = false
function past_bound!(tracer::Tracer{E}, bound::Bound{E}, lands::Bool)::Bool where {E <: Real}
    if lands
        # The bordered row holds the coordinate at the value; this drops the solves' rounding.
        tracer.iterate[bound.coordinate] = bound.value
        return false
    end
    return reaches(bound, tracer.point, tracer.iterate) && !on_bound(bound, tracer.iterate)
end

"""
Correct `tracer.predicted`, a predictor `taken` long, then test the corrected point: its distance from the predictor,
`bound`, `accept`, a finite system there, its tangent, step control's factor and its orientation sign. A landing
(`lands`) holds its row from `cut_predictor!`; any other attempt borders with the tangent.
"""
function attempt_step!(tracer::Tracer{E}, bound::Union{Nothing, Bound{E}}, lands::Bool, taken::E)::Attempt{E} where {E <: Real}
    (; algorithm) = tracer
    if !lands
        metric_apply!(tracer.row, tracer.metric, tracer.tangent)
    end
    corrected = correct!(tracer)
    if corrected.refusal != NoRefusal
        return corrected
    end
    if far_from_predictor(tracer, taken, corrected.error_scale)
        return refused(E, FarFromPredictor)
    end
    if past_bound!(tracer, bound, lands)
        return refused(E, PastBound)
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
    return Attempt{E}(NoRefusal, corrected.iterations, corrected.contraction, corrected.first_correction, corrected.error_scale, turn, factor, orientation)
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
            return Attempt{E}(NoRefusal, iteration, contraction, first_correction, settled_scale(scale), E(NaN32), E(NaN32), Int8(0))
        end
    end
    return refused(E, NotConverged, step_factor(tracer.algorithm.step_control, contraction, first_correction, E(NaN32)))
end

"""
`scale` at a converged iterate. A `converged` that settles at a scale that is not finite and non-negative throws an
`ArgumentError`: an infinite scale would bound the error by Inf, and the reach would pass any move.
"""
function settled_scale(scale::E)::E where {E <: Real}
    # A throw: the fault lies in `converged`, and a refusal would only shrink the step until `TooManyRefusals`.
    isfinite(scale) && scale >= 0 || throw(ArgumentError("converged settled at scale $scale; it may settle only at a finite, non-negative scale"))
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
    (; algorithm) = tracer
    metric_distance(tracer.metric, tracer.scratch, tracer.predicted, tracer.iterate) <= algorithm.step_control.reach * taken && return false
    noise = tracer.settle_error + settle_error(tracer, scale, tracer.previous, tracer.iterate)
    return !(correction_length(algorithm.corrector.distance, tracer, tracer.predicted, tracer.iterate) <= noise)
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
Move `tracer` to `attempt`'s corrected point with the new tangent and orientation, freeze both metrics there and size
the next step from the predictor's length `taken`.
"""
function accept_point!(tracer::Tracer{E}, attempt::Attempt{E}, taken::E)::Tracer{E} where {E <: Real}
    (; algorithm) = tracer
    copyto!(tracer.point, tracer.iterate)
    tracer.orientation = attempt.orientation
    freeze_metric!(tracer.metric, tracer.point)
    freeze_metric!(tracer.turn_metric, tracer.point)
    # Measured after the freeze, as the next reach check measures. Only `correct!` writes `previous`: the iterate before.
    tracer.settle_error = settle_error(tracer, attempt.error_scale, tracer.previous, tracer.point)
    tracer.tangent .= tracer.candidate ./ metric_norm(tracer.metric, tracer.candidate)
    (; max_step_growth, max_step) = algorithm.step_control
    tracer.step_length = min(taken / max(attempt.factor, inv(max_step_growth)), max_step)
    tracer.steps += 1
    return tracer
end
