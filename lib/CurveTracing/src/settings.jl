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
    Newton(; max_iterations, converged, contraction_limit, distance = MetricDistance())

The corrector: Newton's method on `[H(x); ⟨t, x - x̂⟩] = 0`, with `x̂` the predictor, `t` the tangent and `⟨⋅,⋅⟩` the
step metric.

- `max_iterations`: corrections before the corrector gives up and the step is refused.
- `converged(previous, iterate, residual)`: whether `iterate`, one correction from `previous`, has converged;
  `residual` is `H(previous)`.
- `contraction_limit`: a correction longer than this times the one before it is refused as diverging.
- `distance(from, to)`: a correction's length, for the contraction test and step control. `MetricDistance()` is the
  step metric's norm.
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
    max_iterations >= 1 || throw(ArgumentError("max_iterations must be at least 1; got $max_iterations"))
    contraction_limit > 0 || throw(ArgumentError("contraction_limit must be positive; got $contraction_limit"))
    return Newton{C, E, D}(Int(max_iterations), converged, contraction_limit, distance)
end

"""
    StepControl(; nominal_contraction, nominal_distance, nominal_turn, growth_limit, max_step, refusal_factor = 2,
                shrink = 1/2)

Step-length adaptation after Allgower and Georg (Numerical Continuation Methods, 1990, §6.1). A converged corrector
gives `f = max(√(κ / nominal_contraction), √(δ / nominal_distance), α / nominal_turn)`: `κ` the ratio of its second
correction to its first, `δ` its first correction's length (`Newton`'s `distance`) and `α` the turn between the
tangents in radians. `nothing` drops a term. A step with `f > refusal_factor` is refused and retried at `shrink`
times its length; any other step is accepted, and the next one is `1 / max(f, 1 / growth_limit)` times as long, at
most `max_step`. Every other refusal also shrinks the step by `shrink`.
"""
struct StepControl{E <: Real}
    nominal_contraction::Union{Nothing, E}
    nominal_distance::Union{Nothing, E}
    nominal_turn::Union{Nothing, E}
    growth_limit::E
    max_step::E
    refusal_factor::E
    shrink::E
end

function StepControl(;
        nominal_contraction::Union{Nothing, Real},
        nominal_distance::Union{Nothing, Real},
        nominal_turn::Union{Nothing, Real},
        growth_limit::E,
        max_step::Real,
        refusal_factor::Real = 2,
        shrink::Real = 1 // 2,
    )::StepControl{E} where {E <: Real}
    growth_limit >= 1 || throw(ArgumentError("growth_limit must be at least 1; got $growth_limit"))
    refusal_factor >= 1 || throw(ArgumentError("refusal_factor must be at least 1; got $refusal_factor"))
    0 < shrink < 1 || throw(ArgumentError("shrink must lie in (0, 1); got $shrink"))
    max_step > 0 || throw(ArgumentError("max_step must be positive; got $max_step"))
    return StepControl{E}(
        optional(E, nominal_contraction), optional(E, nominal_distance), optional(E, nominal_turn), growth_limit,
        E(max_step), E(refusal_factor), E(shrink),
    )
end

"""`limit` in `E`, or `nothing`; a set limit must be positive."""
optional(::Type{<:Real}, ::Nothing)::Nothing = nothing

function optional(::Type{E}, limit::Real)::E where {E <: Real}
    limit > 0 || throw(ArgumentError("a nominal value must be positive; got $limit"))
    return E(limit)
end

"""
    PseudoArclength(; corrector, step_control, metric, turn_metric = metric, linear_solver, accept = AcceptAll(),
                    resolution_ulps = 4, max_refusals)

Pseudo-arclength continuation with an Euler predictor along the bordered tangent.

- `corrector`: a `Newton`.
- `step_control`: a `StepControl`.
- `metric`: the step metric (`FixedWeights` or `PointWeights`). Step lengths, the corrector's hyperplane and the
  tangent's unit length are in it.
- `turn_metric`: the metric that orients each tangent along the one before and measures their turn.
- `linear_solver`: the LinearSolve.jl algorithm for the bordered systems.
- `accept(start, predicted, corrected)`: whether to keep a converged point; `false` refuses the step.
- `resolution_ulps`: a predictor that moves no coordinate by more than this many ulps of its size ends the trace
  as `Unresolved`; smaller steps cannot add a point.
- `max_refusals`: refusals in one step before the trace ends as `TooManyRefusals`.
"""
struct PseudoArclength{C <: Newton, S <: StepControl, M, T, L, A}
    corrector::C
    step_control::S
    metric::M
    turn_metric::T
    linear_solver::L
    accept::A
    resolution_ulps::Int
    max_refusals::Int
end

"""A `PseudoArclength` `accept` that keeps every converged point."""
struct AcceptAll end

(::AcceptAll)(::AbstractVector, ::AbstractVector, ::AbstractVector)::Bool = true

function PseudoArclength(;
        corrector::C, step_control::S, metric::M, turn_metric::T = metric, linear_solver::L, accept::A = AcceptAll(),
        resolution_ulps::Integer = 4, max_refusals::Integer,
    )::PseudoArclength{C, S, M, T, L, A} where {C <: Newton, S <: StepControl, M, T, L, A}
    resolution_ulps >= 0 || throw(ArgumentError("resolution_ulps must be non-negative; got $resolution_ulps"))
    max_refusals >= 0 || throw(ArgumentError("max_refusals must be non-negative; got $max_refusals"))
    return PseudoArclength{C, S, M, T, L, A}(
        corrector, step_control, metric, turn_metric, linear_solver, accept, Int(resolution_ulps), Int(max_refusals),
    )
end
