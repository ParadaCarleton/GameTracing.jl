"""
    CurveProblem(system!, start, direction)

The curve `H(x) = 0` through `start`, traced toward `direction`. `system!(residual, jacobian, x)` writes `H(x)` (length
`n`) and its Jacobian `∂H/∂x` (`n × (n + 1)`) from one evaluation and returns `nothing`. `start` and `direction` have
`n + 1` entries; `direction` needs only a positive product with the tangent at `start` under the turn metric.
"""
struct CurveProblem{S, V <: AbstractVector{<:Real}}
    system!::S
    start::V
    direction::V
end

"""
    MetricDistance()

A correction's length in the trace's step metric (`PseudoArclength`'s `metric`).
"""
struct MetricDistance end

"""
    correction_bound(distance)

The largest length the corrector's `distance` measures, `Inf` for an unbounded one. Each distance that step control
targets defines it: `PseudoArclength` checks that its first-correction refusal can fire.
"""
function correction_bound end

correction_bound(::MetricDistance)::Real = Inf32

"""
    LastCorrection()

`Newton`'s `error_estimate` that settles on the last correction itself: `converged` sees `scale = 1`. That correction
bounds the error of the iterate before it, so the corrector runs one iteration past need.
"""
struct LastCorrection end

"""
    ContractionEstimate(; safety)

`Newton`'s `error_estimate` that settles on the error left in the iterate: the last correction times `θ / (1 - θ)`, `θ`
its ratio to the one before, over `safety` in `(0, 1]` (Hairer and Wanner, Solving Ordinary Differential Equations II,
1996, §IV.8, take 0.01 to 0.1). The first correction has no ratio and gets `scale = 1`; a ratio of one or more gets
`Inf`, which never settles. A slow contraction estimates its error poorly, and `safety = 1` let one such corrector
settle off the curve near a fold, which stopped the trace.
"""
struct ContractionEstimate{E <: Real}
    safety::E
    function ContractionEstimate(; safety::E)::ContractionEstimate{E} where {E <: Real}
        0 < safety <= 1 || throw(ArgumentError("safety must be in (0, 1]; got $safety"))
        return new{E}(safety)
    end
end

"""
    Newton(; max_iterations, converged, contraction_limit, error_estimate, distance = MetricDistance())

The corrector: Newton's method on `[H(x); ⟨t, x - x̂⟩] = 0`, with `x̂` the predictor, `t` the tangent and `⟨⋅,⋅⟩` the
step metric.

- `max_iterations`: corrections before the corrector gives up and the step is refused.
- `converged(previous, iterate, residual, scale)`: whether `iterate`, one correction from `previous`, has converged,
  the correction times `scale` standing for the error left in `iterate`; `residual` is `H(previous)`. A `converged`
  that settles at an infinite or NaN `scale` throws an `ArgumentError`.
- `contraction_limit`: a correction longer than this times the one before it is refused as diverging.
- `error_estimate`: `scale`'s source, `LastCorrection()` or `ContractionEstimate(; safety)`.
- `distance(from, to)`: a correction's length, for the contraction test, the error estimate and step control.
  `MetricDistance()` is the step metric's norm. A distance with a target first correction declares the largest length
  it measures (`correction_bound`).
"""
struct Newton{C, E <: Real, R <: Union{LastCorrection, ContractionEstimate{E}}, D}
    max_iterations::Int
    converged::C
    contraction_limit::E
    error_estimate::R
    distance::D
end

function Newton(;
        max_iterations::Integer, converged::C, contraction_limit::E, error_estimate::R, distance::D = MetricDistance(),
    )::Newton{C, E, R, D} where {C, E <: Real, R <: Union{LastCorrection, ContractionEstimate}, D}
    # The bound ties `R`'s float type to `E` only on the struct: a keyword method's bound cannot name `E`.
    max_iterations >= 1 || throw(ArgumentError("max_iterations must be at least 1; got $max_iterations"))
    contraction_limit > 0 || throw(ArgumentError("contraction_limit must be positive; got $contraction_limit"))
    return Newton{C, E, R, D}(Int(max_iterations), converged, contraction_limit, error_estimate, distance)
end

"""The `scale` `converged` sees after a correction of length `correction` that followed one of length `before`."""
function error_scale(::LastCorrection, ::E, ::E)::E where {E <: Real}
    return one(E)
end

function error_scale(estimate::ContractionEstimate{E}, correction::E, before::E)::E where {E <: Real}
    # The first correction follows a NaN `before`.
    isnan(before) && return one(E)
    ratio = correction / before
    return ratio < 1 ? ratio / ((1 - ratio) * estimate.safety) : E(Inf32)
end

"""
    StepControl(; target_contraction, target_correction, target_turn, max_step_growth, max_step, refusal_factor = 2,
                shrink = 1/2, reach = 3)

Step-length adaptation after Allgower and Georg (Numerical Continuation Methods, 1990, §6.1). A converged corrector
gives `f = max(√(κ / target_contraction), √(δ / target_correction), α / target_turn)`: `κ` the ratio of its second
correction to its first, `δ` its first correction's length (`Newton`'s `distance`) and `α` the turn between the
tangents in radians. Each target is the value its measurement should take: at `f = 1` the step keeps its length.
The turn is measured in the turn metric, so its target depends on that metric's scaling. `nothing` drops a term.

A step with `f > refusal_factor` is refused; any other step is accepted, and the next one is `1 / max(f, 1 /
max_step_growth)` times as long as its predictor (cut at a `Bound`), at most `max_step`. A refused step is retried at `1 / max(f, 1 / shrink)` times its length, with `f` from the measurements it made: `κ`
and `δ` when the corrector diverged or ran out of iterations, none (so `1 / shrink`) when it failed earlier.

A converged point farther than `reach` predictor lengths from its predictor, in the step metric, is refused
(`FarFromPredictor`) and retried at `shrink` times the length. Such a corrector left the curve the predictor followed,
and its point can lie on another branch. A move within the corrector's noise passes: in `Newton`'s `distance`, at most
the error bounds of the start and of the point, each `scale` times its last correction. Each point settles only to its
tolerance, so at short steps the start's own error moves the corrector past the reach. On 300 unfolded pitchforks per
scale (battle of the sexes plus `0.1 randn` payoffs), steps that kept the branch moved at most 1.86 lengths, and the 3
that jumped moved 4.2 to 5.09. `reach = Inf`
switches the refusal off; any other `reach` must be positive and finite in the step control's type.
"""
struct StepControl{E <: Real}
    target_contraction::Union{Nothing, E}
    target_correction::Union{Nothing, E}
    target_turn::Union{Nothing, E}
    max_step_growth::E
    max_step::E
    refusal_factor::E
    shrink::E
    reach::E
end

function StepControl(;
        target_contraction::Union{Nothing, Real},
        target_correction::Union{Nothing, Real},
        target_turn::Union{Nothing, Real},
        max_step_growth::E,
        max_step::Real,
        refusal_factor::Real = 2,
        shrink::Real = 1 // 2,
        reach::Real = 3,
    )::StepControl{E} where {E <: Real}
    max_step_growth >= 1 || throw(ArgumentError("max_step_growth must be at least 1; got $max_step_growth"))
    refusal_factor >= 1 || throw(ArgumentError("refusal_factor must be at least 1; got $refusal_factor"))
    # Checked in `E`, where a shrink near 1 rounds to 1 and a retry would repeat its step.
    zero(E) < E(shrink) < one(E) || throw(ArgumentError("shrink must lie in (0, 1) in $E; got $shrink"))
    max_step > 0 || throw(ArgumentError("max_step must be positive; got $max_step"))
    # A straight stretch measures f = 0, so the growth bound or the cap must stop the step from going infinite.
    isfinite(max_step_growth) || isfinite(max_step) || throw(ArgumentError("max_step_growth and max_step cannot both be infinite"))
    return StepControl{E}(
        optional(E, target_contraction), optional(E, target_correction), optional(E, target_turn), max_step_growth,
        E(max_step), E(refusal_factor), E(shrink), reach_in(E, reach),
    )
end

"""
`reach` in `E`. It must stay positive and finite there: a tiny reach rounds to zero, which refuses every move, and a
huge one rounds to Inf. Only an explicit Inf switches the refusal off.
"""
function reach_in(::Type{E}, reach::Real)::E where {E <: Real}
    converted = E(reach)
    zero(E) < converted && (isfinite(converted) || isinf(reach)) || throw(ArgumentError("reach must be positive and finite in $E, or Inf; got $reach"))
    return converted
end

"""`limit` in `E`, or `nothing`; a set limit must be positive."""
optional(::Type{<:Real}, ::Nothing)::Nothing = nothing

function optional(::Type{E}, limit::Real)::E where {E <: Real}
    limit > 0 || throw(ArgumentError("a target must be positive; got $limit"))
    return E(limit)
end

"""
    DeterminantSign()

`PseudoArclength`'s orientation check: the sign of `det([J; tᵀ])` at each corrected point, with `J` the Jacobian and
`t` the tangent there. The sign is constant along a branch between bifurcation points. A changed sign refuses the step
(`OrientationFlip`). The corrector landed where the tangent, oriented along the one before, runs back along the
curve, or the step crossed a bifurcation point. The check never accepts a changed sign. A jump onto a part of the curve
that runs the other way changes it the same way as a crossing, so no rule tells the two apart. At an exact
bifurcation the trace closes in on the point in shorter steps and fails `Unresolved` below the float resolution.

With `LUFactorization`, the check reads the sign from the tangent solve's own LU in O(n). Other linear solvers pay a
second LU factorization, of `[J; tᵀ]`, at each corrected point.
"""
struct DeterminantSign end

"""
    PseudoArclength(; corrector, step_control, metric, turn_metric = metric, orientation_check, linear_solver,
                    accept = AcceptAll(), min_step_ulps = 4, max_refusals)

Pseudo-arclength continuation with an Euler predictor along the bordered tangent.

- `corrector`: a `Newton`.
- `step_control`: a `StepControl`.
- `metric`: the step `Metric`, such as `FixedWeights`. Step lengths, the corrector's hyperplane and
  the tangent's unit length are in it.
- `turn_metric`: the `Metric` that orients each tangent along the one before and measures their turn.
- `orientation_check`: a `DeterminantSign`, or `nothing` to orient each tangent by the turn metric alone. Without it, a long
  step near a fold can land where the oriented tangent runs back along the curve.
- `linear_solver`: the LinearSolve.jl algorithm for the bordered systems.
- `accept(start, predicted, corrected)`: whether to keep a converged point; `false` refuses the step.
- `min_step_ulps`: a predictor that moves no coordinate by more than this many ulps of its size ends the trace
  as `Unresolved`; smaller steps cannot add a point. The check judges the step before a `Bound` cuts it.
- `max_refusals`: refusals in one step before the trace ends as `TooManyRefusals`.
"""
struct PseudoArclength{C <: Newton, S <: StepControl, M <: Metric, T <: Metric, O <: Union{Nothing, DeterminantSign}, L, A}
    corrector::C
    step_control::S
    metric::M
    turn_metric::T
    orientation_check::O
    linear_solver::L
    accept::A
    min_step_ulps::Int
    max_refusals::Int
end

"""
Throw unless `control`'s first-correction term can refuse a step: `target_correction refusal_factor²`, the first
correction past which it refuses, must lie under `correction_bound(distance)`.
"""
function refusable_correction(control::StepControl, distance::Any)::Nothing
    (; target_correction, refusal_factor) = control
    isnothing(target_correction) && return nothing
    limit, bound = target_correction * refusal_factor^2, correction_bound(distance)
    limit < bound || throw(
        ArgumentError("step control refuses first corrections past $limit, but the corrector's distance measures at most $bound")
    )
    return nothing
end

"""A `PseudoArclength` `accept` that keeps every converged point."""
struct AcceptAll end

(::AcceptAll)(::AbstractVector, ::AbstractVector, ::AbstractVector)::Bool = true

function PseudoArclength(;
        corrector::C, step_control::S, metric::M, turn_metric::T = metric, orientation_check::O, linear_solver::L,
        accept::A = AcceptAll(), min_step_ulps::Integer = 4, max_refusals::Integer,
    )::PseudoArclength{C, S, M, T, O, L, A} where {C <: Newton, S <: StepControl, M <: Metric, T <: Metric, O <: Union{Nothing, DeterminantSign}, L, A}
    min_step_ulps >= 0 || throw(ArgumentError("min_step_ulps must be non-negative; got $min_step_ulps"))
    refusable_correction(step_control, corrector.distance)
    max_refusals >= 0 || throw(ArgumentError("max_refusals must be non-negative; got $max_refusals"))
    return PseudoArclength{C, S, M, T, O, L, A}(
        corrector, step_control, metric, turn_metric, orientation_check, linear_solver, accept, Int(min_step_ulps),
        Int(max_refusals),
    )
end
