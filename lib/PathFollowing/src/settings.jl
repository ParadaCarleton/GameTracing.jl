"""
    CurveProblem(system!!, start, direction)

The curve `H(x) = 0` through `start`, traced toward `direction`. `system!!(residual, jacobian, x)` returns `H(x)`
(length `n`) and its Jacobian `∂H/∂x` (`n × (n + 1)`) from one evaluation, as `(residual, jacobian)` in the types of
the arrays it gets. Mutable arrays come back changed in place, and static arrays come back new (`into!!`). `start` and
`direction` have `n + 1` entries; `direction` needs only a positive product with the tangent at `start` under the turn
metric. A static `start` (`SVector`) needs `StaticLU`, and a start in a workgroup's memory (`GroupVector`) needs `GroupLU`.
"""
struct CurveProblem{S, V <: AbstractVector{<:Real}}
    system!!::S
    start::V
    direction::V
end

"""
    MetricDistance()

A correction's length in the trace's step metric (`PseudoArclength`'s `metric`).
"""
struct MetricDistance end

"""
    correction_bound(distance, E)

The largest length the corrector's `distance` measures, in step control's float type `E`; `Inf` for an unbounded one.
Each distance that step control targets defines it: `PseudoArclength` checks that its first-correction refusal can fire.
"""
function correction_bound end

function correction_bound(::MetricDistance, ::Type{E})::E where {E <: Real}
    return E(Inf32)
end

"""
    Newton(; max_iterations, converged, contraction_limit, distance = MetricDistance())

The corrector: Newton's method on `[H(x); ⟨t, x - x̂⟩] = 0`, with `x̂` the predictor, `t` the tangent and `⟨⋅,⋅⟩` the
step metric. It settles where `converged` passes its last correction, as Gambit's corrector does.

- `max_iterations`: corrections before the corrector gives up and the step is refused.
- `converged(previous, iterate, residual)`: whether `iterate`, one correction from `previous`, has converged;
  `residual` is `H(previous)`.
- `contraction_limit`: a correction longer than this times the one before it is refused as diverging. It must lie in
  `(0, 1/2]` (`contraction_limit_in`). Where later corrections would contract as fast, the last correction then bounds
  the error left in its iterate. Two corrections cannot show that, so the last correction only estimates that error.
- `distance(from, to)`: a correction's length, for the contraction test, the error estimate and step control.
  `MetricDistance()` is the step metric's norm. A distance with a target first correction declares the largest length
  it measures (`correction_bound`).

An estimate of the error left, `θ / (1 - θ)` times the last correction (Hairer and Wanner, Solving Ordinary
Differential Equations II, 1996, §IV.8), settled with 6 to 18% fewer corrections. Over 2502 benchmark traces it failed
16 times against 12. Its median Float64 branch error rose from 5e-14 to 4e-12. A guard on the estimate, a bound on `θ`
or a falling `θ`, settled no more traces and added branch error.
"""
struct Newton{C, E <: Real, D}
    max_iterations::Int
    converged::C
    contraction_limit::E
    distance::D
end

function Newton(;
        max_iterations::Integer, converged::C, contraction_limit::E, distance::D = MetricDistance(),
    )::Newton{C, E, D} where {C, E <: Real, D}
    max_iterations >= 1 || throw(ArgumentError(lazy"max_iterations must be at least 1; got $max_iterations"))
    return Newton{C, E, D}(Int(max_iterations), converged, contraction_limit_in(E, contraction_limit), distance)
end

"""
`limit` in `E`, which must lie in `(0, 1/2]` there. Past 1/2, corrections that keep contracting by `θ` leave
`θ / (1 - θ) > 1` times the last one in its iterate, so even then a settled last correction would not bound that error.
"""
function contraction_limit_in(::Type{E}, limit::Real)::E where {E <: Real}
    converted = E(limit)
    zero(E) < converted <= one(E) / 2 || throw(ArgumentError(lazy"contraction_limit must lie in (0, 1/2] in $E; got $limit"))
    return converted
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
the last corrections of the start and of the point, which estimate their errors. Each point settles only to its
tolerance, so at short steps the start's own error moves the corrector past the reach. On 300 unfolded pitchforks per
scale (battle of the sexes plus `0.1 randn` payoffs), steps that kept the branch moved at most 1.86 lengths, and the 3
that jumped moved 4.2 to 5.09. `reach = Inf`
switches the refusal off; any other `reach` must be positive and finite in the step control's type.
"""
# Each target's type is a parameter, so a step control is isbits and a GPU kernel can take it.
struct StepControl{E <: Real, C <: Union{Nothing, E}, D <: Union{Nothing, E}, T <: Union{Nothing, E}}
    target_contraction::C
    target_correction::D
    target_turn::T
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
    max_step_growth >= 1 || throw(ArgumentError(lazy"max_step_growth must be at least 1; got $max_step_growth"))
    refusal_factor >= 1 || throw(ArgumentError(lazy"refusal_factor must be at least 1; got $refusal_factor"))
    # Checked in `E`, where a shrink near 1 rounds to 1 and a retry would repeat its step.
    zero(E) < E(shrink) < one(E) || throw(ArgumentError(lazy"shrink must lie in (0, 1) in $E; got $shrink"))
    max_step > 0 || throw(ArgumentError(lazy"max_step must be positive; got $max_step"))
    # A straight stretch measures f = 0, so the growth bound or the cap must stop the step from going infinite.
    isfinite(max_step_growth) || isfinite(max_step) || throw(ArgumentError("max_step_growth and max_step cannot both be infinite"))
    contraction, correction, turn = optional(E, target_contraction), optional(E, target_correction), optional(E, target_turn)
    return StepControl{E, typeof(contraction), typeof(correction), typeof(turn)}(
        contraction, correction, turn, max_step_growth, E(max_step), E(refusal_factor), E(shrink), reach_in(E, reach),
    )
end

"""
`reach` in `E`. It must stay positive and finite there: a tiny reach rounds to zero, which refuses every move, and a
huge one rounds to Inf. Only an explicit Inf switches the refusal off.
"""
function reach_in(::Type{E}, reach::Real)::E where {E <: Real}
    converted = E(reach)
    zero(E) < converted && (isfinite(converted) || isinf(reach)) || throw(ArgumentError(lazy"reach must be positive and finite in $E, or Inf; got $reach"))
    return converted
end

"""`limit` in `E`, or `nothing`; a set limit must be positive."""
optional(::Type{<:Real}, ::Nothing)::Nothing = nothing

function optional(::Type{E}, limit::Real)::E where {E <: Real}
    limit > 0 || throw(ArgumentError(lazy"a target must be positive; got $limit"))
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

With `LUFactorization` or `StaticLU`, the check reads the sign from the tangent solve's own LU in O(n). Other linear solvers pay a
second LU factorization, of `[J; tᵀ]`, at each corrected point.
"""
struct DeterminantSign end

"""
    HalfStepCheck{E}(; trigger, tolerance)
    HalfStepCheck(; trigger, tolerance)

`PseudoArclength`'s check that a step kept its branch. A step whose factor `f` (`StepControl`) is at least `trigger`
runs again in two half steps. The first corrects at the half predictor. The second predicts along that point's tangent
to the step's own hyperplane (the bound's at a `Bound`) and corrects there. Near the corrected point that hyperplane
meets the start's branch once, so on that branch the half steps reach the point. A point further away than
`tolerance` in `Newton`'s `distance`, or a half step that fails, refuses the step (`SplitMismatch`). The retry is
`shrink` times as long. A corrector can converge on another branch that passes the predictor closer than its own, with
every step measurement in range. The shorter half steps often keep the start's branch. Half steps that jump to the
step's point pass the check.

A trigger of 1 checks every step that meets or passes a step-control target. A higher trigger fits the smallest jump
seen: on random two-action games at growth 2, more draws found `f` 1.52, then 1.23, then 1.1 to 1.2. A slow growth
holds many steps at a target, where trigger 1 fires on most of them: GameTracing's growth 1.1 picks 1.3.

Both numbers convert to `E` and must stay finite there: `trigger` non-negative and `tolerance` positive. Without `E`,
the check takes the float type of the two numbers promoted. `PseudoArclength` also needs `trigger` at most the step
control's `refusal_factor`, past which a step is refused first.
"""
struct HalfStepCheck{E <: Real}
    trigger::E
    tolerance::E
    function HalfStepCheck{E}(; trigger::Real, tolerance::Real)::HalfStepCheck{E} where {E <: Real}
        # Checked in `E`, where a tiny tolerance rounds to zero and a huge number to Inf.
        converted_trigger, converted_tolerance = E(trigger), E(tolerance)
        zero(E) <= converted_trigger < Inf || throw(ArgumentError(lazy"trigger must be non-negative and finite in $E; got $trigger"))
        zero(E) < converted_tolerance < Inf || throw(ArgumentError(lazy"tolerance must be positive and finite in $E; got $tolerance"))
        return new{E}(converted_trigger, converted_tolerance)
    end
end

"""`HalfStepCheck` in the float type of `trigger` and `tolerance` promoted."""
function HalfStepCheck(; trigger::Real, tolerance::Real)::HalfStepCheck
    return HalfStepCheck{float(promote_type(typeof(trigger), typeof(tolerance)))}(; trigger, tolerance)
end

"""
    StaticLU()

`PseudoArclength`'s `linear_solver` for a static start (`SVector`): StaticArrays.jl's pivoted LU, `lu(A, RowMaximum();
check = false)`, built anew at each solve. It keeps no cache and allocates nothing. A zero pivot fails the solve. StaticArrays factors at most 14 × 14 matrices this way, so a start has at most 14
coordinates. The orientation check reads its sign from this LU in O(n).
"""
struct StaticLU end

"""
    PseudoArclength(; corrector, step_control, metric, turn_metric = metric, orientation_check, split_check,
                    linear_solver, accept = AcceptAll(), min_step_ulps = 4, max_refusals = nothing)

Pseudo-arclength continuation with an Euler predictor along the bordered tangent.

- `corrector`: a `Newton`.
- `step_control`: a `StepControl`.
- `metric`: the step `Metric`, such as `FixedWeights`. Step lengths, the corrector's hyperplane and
  the tangent's unit length are in it.
- `turn_metric`: the `Metric` that orients each tangent along the one before and measures their turn.
- `orientation_check`: a `DeterminantSign`, or `nothing` to orient each tangent by the turn metric alone. Without it, a long
  step near a fold can land where the oriented tangent runs back along the curve.
- `split_check`: a `HalfStepCheck`, or `nothing` to accept a converged point on another branch that passes every other
  test.
- `linear_solver`: the LinearSolve.jl algorithm for the bordered systems of a mutable start, `StaticLU()` for a
  static start, or `GroupLU()` for a start in a workgroup's memory.
- `accept(start, predicted, corrected)`: whether to keep a converged point; `false` refuses the step.
- `min_step_ulps`: a predictor that moves no coordinate by more than this many ulps of its size ends the trace
  as `Unresolved`; smaller steps cannot add a point. The check judges the step before a `Bound` cuts it.
- `max_refusals`: refusals in one step before the trace ends as `TooManyRefusals`, or `nothing` for no limit. Each
  refusal shrinks the step (`StepControl`'s `shrink`), and a retry that rounds back to its length fails `Unresolved`,
  so a step ends at the float type's resolution without a limit.
"""
struct PseudoArclength{C <: Newton, S <: StepControl, M <: Metric, T <: Metric, O <: Union{Nothing, DeterminantSign}, K <: Union{Nothing, HalfStepCheck}, L, A, R <: Union{Nothing, Int}}
    corrector::C
    step_control::S
    metric::M
    turn_metric::T
    orientation_check::O
    split_check::K
    linear_solver::L
    accept::A
    min_step_ulps::Int
    # A type parameter, not a `Union` field, so a batch algorithm stays isbits.
    max_refusals::R
end

"""
Throw unless `control`'s first-correction term can refuse a step: `target_correction refusal_factor²`, the first
correction past which it refuses, must lie under `correction_bound(distance, E)`.
"""
function refusable_correction(control::StepControl{E}, distance::Any)::Nothing where {E <: Real}
    (; target_correction, refusal_factor) = control
    isnothing(target_correction) && return nothing
    limit, bound = target_correction * refusal_factor^2, correction_bound(distance, E)
    limit < bound || throw(
        ArgumentError(lazy"step control refuses first corrections past $limit, but the corrector's distance measures at most $bound")
    )
    return nothing
end

"""A `PseudoArclength` `accept` that keeps every converged point."""
struct AcceptAll end

(::AcceptAll)(::AbstractVector, ::AbstractVector, ::AbstractVector)::Bool = true

function PseudoArclength(;
        corrector::C, step_control::S, metric::M, turn_metric::T = metric, orientation_check::O, split_check::K,
        linear_solver::L, accept::A = AcceptAll(), min_step_ulps::Integer = 4, max_refusals::Union{Nothing, Integer} = nothing,
    )::PseudoArclength{C, S, M, T, O, K, L, A} where {C <: Newton, S <: StepControl, M <: Metric, T <: Metric, O <: Union{Nothing, DeterminantSign}, K <: Union{Nothing, HalfStepCheck}, L, A}
    min_step_ulps >= 0 || throw(ArgumentError(lazy"min_step_ulps must be non-negative; got $min_step_ulps"))
    refusable_correction(step_control, corrector.distance)
    reachable_trigger(step_control, split_check)
    limit = checked_limit(:max_refusals, max_refusals)
    return PseudoArclength{C, S, M, T, O, K, L, A, typeof(limit)}(
        corrector, step_control, metric, turn_metric, orientation_check, split_check, linear_solver, accept,
        Int(min_step_ulps), limit,
    )
end

"""
`limit` as an `Int`, or `nothing` for no limit; a negative limit throws an `ArgumentError` that names the keyword `name`.
"""
checked_limit(::Symbol, ::Nothing)::Nothing = nothing
function checked_limit(name::Symbol, limit::Integer)::Int
    limit >= 0 || throw(ArgumentError(lazy"$name must be non-negative or nothing; got $limit"))
    return Int(limit)
end

"""
Whether `count` has reached `limit`, a step or refusal budget; `nothing` is no limit, which no count reaches. Both
trace engines of GameTracing count their steps with it.
"""
limit_reached(::Nothing, ::Integer)::Bool = false
limit_reached(limit::Integer, count::Integer)::Bool = count >= limit

"""Throw unless `check` can fire: its trigger must not pass `control`'s `refusal_factor`, past which steps are refused."""
reachable_trigger(::StepControl, ::Nothing)::Nothing = nothing
function reachable_trigger(control::StepControl, check::HalfStepCheck)::Nothing
    check.trigger <= control.refusal_factor ||
        throw(ArgumentError(lazy"the split check's trigger $(check.trigger) must not pass the refusal factor $(control.refusal_factor)"))
    return nothing
end
