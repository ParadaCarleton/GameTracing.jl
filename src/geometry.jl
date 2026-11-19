using BifurcationKit: BifurcationKit
using LinearAlgebra: dot
using PathFollowing: PathFollowing
"""
The trace's step length in a state `x = [w; λ]` of `k` payoff gaps `w` at target precision `P`, at `state_weight` (the
gaps' weight). Gaps and precision enter in the units `P w` and `λ / P`, so each plain distance weighs `Δwᵢ²` by `P²`
and `Δλ²` by `1 / P²`:

- `AveragedDistance`: `state_weight P² ‖Δw‖² / k + (1 - state_weight) (Δλ / P)²`.
- `SummedDistance`: `state_weight P² ‖Δw‖² + (Δλ / P)²`.
- `FisherDistance`: `state_weight Σₚ Δzₚᵀ Gₚ Δzₚ` plus a `PrecisionTerm`: the Fisher information of the strategy
  profile at the logits `z = λ w`, pulled back to the state through `Δz = λ Δw + w Δλ`. Player `p`'s block
  `Gₚ = diag(qₚ) - qₚqₚᵀ` spans its logits (`logit_ranges`); `FisherMetric` applies it.

The Fisher distance freezes `G`, `w` and `λ` at each accepted point. The trace's plain products (`trace_dot`) measure it
as `SummedDistance`. The corrector's reach check measures in logits under every distance (`within_reach`).
"""
@enum StepDistance AveragedDistance SummedDistance FisherDistance

"""
The precision's term in `FisherDistance`: `TargetPrecisionTerm` `(Δλ / P)²`, or `LogPrecisionTerm`
`(Δλ / (1 + λ))²`, the step in `log(1 + λ)` to first order, LogitNash's path variable. The log term does not depend on
`P`, so the target only caps the trace. The plain distances weigh the precision at the target alone.
"""
@enum PrecisionTerm TargetPrecisionTerm LogPrecisionTerm

"""
    TraceGeometry{E}(; distance, state_weight, precision_term)

The norm a trace steps in: its `StepDistance`, `state_weight`, the gaps' weight against the precision's, and the
Fisher distance's `PrecisionTerm`. A plain distance takes `TargetPrecisionTerm` only; another throws an
`ArgumentError`. The default is both engines' (`default_step_metric`): the Fisher distance at `state_weight = 0.1`
with `LogPrecisionTerm`. A plain distance defaults to `TargetPrecisionTerm`, its only term.

`state_weight = 0.1` serves both precision terms (`benchmark/results/tune-4dd74d7-summary.txt`, weights 0.003 to 0.9).
Each term reached every categorical game at 0.1. The target term took its fewest steps there (304); the log term took
408, against 396 at 0.3. Below 0.03 the log term exhausted at the folding 3x3 game and at `UNFOLDED_PITCHFORK_GAME`.
"""
struct TraceGeometry{E <: Real}
    distance::StepDistance
    state_weight::E
    precision_term::PrecisionTerm

    function TraceGeometry{E}(distance::StepDistance, state_weight::E, precision_term::PrecisionTerm)::TraceGeometry{E} where {E <: Real}
        if distance != FisherDistance && precision_term != TargetPrecisionTerm
            throw(ArgumentError("$distance weighs the precision at the target; $precision_term needs FisherDistance"))
        end
        return new{E}(distance, state_weight, precision_term)
    end
end

function TraceGeometry{E}(;
        distance::StepDistance = FisherDistance,
        state_weight::Real = 0.1f0,
        precision_term::PrecisionTerm = distance == FisherDistance ? LogPrecisionTerm : TargetPrecisionTerm,
    )::TraceGeometry{E} where {E <: Real}
    return TraceGeometry{E}(distance, E(state_weight), precision_term)
end

"""
    default_step_metric(E)

The step metric of both engines' defaults (`BifurcationKitSettings`, `gambit_settings`, `PathFollowingSettings`):
`TraceGeometry`'s defaults, defined once so the engines cannot drift. `FisherDistance` with `LogPrecisionTerm` steps
in `log(1 + λ)`, so a trace to a far precision opens at the scale of a near one.
"""
function default_step_metric(::Type{E})::TraceGeometry{E} where {E <: Real}
    return TraceGeometry{E}()
end

"""
Gambit's largest step growth, `max_accel`: the default growth of both engines (`logit_solve`'s `max_accel`,
BifurcationKit's `StepControl`, `PathFollowingSettings`), defined once so they cannot drift.
"""
const GAMBIT_MAX_ACCEL = 1.1f0

"""The weight of each `Δwᵢ²` in `geometry`'s norm over `width` gaps, before the target's `P²` (`StepDistance`)."""
function coordinate_weight(geometry::TraceGeometry{E}, width::Integer)::E where {E <: Real}
    if geometry.distance == AveragedDistance
        return geometry.state_weight / width
    end
    return geometry.state_weight
end

"""The weight of `(Δλ / P)²` in `geometry`'s norm (`StepDistance`)."""
function precision_weight(geometry::TraceGeometry{E})::E where {E <: Real}
    if geometry.distance == AveragedDistance
        return one(E) - geometry.state_weight
    end
    return one(E)
end

"""
The root of the weight of `Δλ²` in `geometry`'s norm at `λ = 0` toward `target_precision` `P`. A unit step there moves
`λ` by at most its inverse; the state's terms only shorten the move. `LogPrecisionTerm` gives one; the target's unit
`λ / P` gives `√precision_weight / P`. The root never overflows, though its inverse may.
"""
function opening_root(geometry::TraceGeometry{E}, target_precision::Real)::E where {E <: Real}
    target = E(target_precision)
    if geometry.distance == FisherDistance
        return precision_term_root(geometry.precision_term, zero(E), target)
    end
    return sqrt(precision_weight(geometry)) / target
end

"""
`(gap, precision)`: the weights of `Σ Δwᵢ²` and `Δλ²` in `geometry`'s plain norm over `width` gaps at target precision
`target_precision` (`coordinate_weight`, `precision_weight`, `StepDistance`). A target whose square leaves `E`'s
range throws `ArgumentError`; `FisherDistance` with `LogPrecisionTerm` does not square it.
"""
@inline function plain_weights(geometry::TraceGeometry{E}, width::Integer, target_precision::Real)::NTuple{2, E} where {E <: Real}
    target = E(target_precision)
    gap, precision = coordinate_weight(geometry, width) * target^2, precision_weight(geometry) / target^2
    if !(gap > 0 && isfinite(gap) && precision > 0 && isfinite(precision))
        squared_out_of_range(target_precision, E)
    end
    return (gap, precision)
end

"""The `ArgumentError` of a target precision whose square leaves `E`'s range; out of line, so inlined `plain_weights` stays small."""
@noinline function squared_out_of_range(target_precision::Real, ::Type{E})::Union{} where {E <: Real}
    throw(ArgumentError("target precision $target_precision squares out of range in $E; use FisherDistance with LogPrecisionTerm or a wider type"))
end

"""The plain inner product of two displacements `[Δw; Δλ]` in `geometry` at `target_precision` (`plain_weights`)."""
function trace_dot(
        left::AbstractVector{E},
        right::AbstractVector{E},
        geometry::TraceGeometry{E},
        target_precision::Real,
    )::E where {E <: Real}
    gap, precision = plain_weights(geometry, length(left) - 1, target_precision)
    return gap * dot(payoff_gaps(left), payoff_gaps(right)) + precision * last(left) * last(right)
end

"""The distance from `from` to `to`, both `[w; λ]`, in `geometry`'s plain norm at `target_precision` (`trace_dot`)."""
function arclength_distance(
        from::AbstractVector{E},
        to::AbstractVector{E},
        geometry::TraceGeometry{E},
        target_precision::Real,
    )::E where {E <: Real}
    gap_square = zero(E)
    for index in firstindex(from):(lastindex(from) - 1)
        gap_square += abs2(to[index] - from[index])
    end
    gap, precision = plain_weights(geometry, length(from) - 1, target_precision)
    return sqrt(gap * gap_square + precision * abs2(last(to) - last(from)))
end

"""
PALC's state weight for a norm whose smallest weight of `Δλ²` over the trace is `smallest`
(`smallest_precision_weight`): one minus half of the smaller of `smallest` and one. PALC's own `Δλ²` term then stays
under the norm's, so the dot's precision weight stays positive.
"""
function palc_state_weight_under(smallest::E)::E where {E <: Real}
    return one(E) - min(smallest, one(E)) / 2
end

"""The smallest weight `geometry`'s norm gives `Δλ²` over the precisions `[0, P]`, over `width` gaps at target precision `P`."""
function smallest_precision_weight(geometry::TraceGeometry{E}, width::Integer, target_precision::E)::E where {E <: Real}
    if geometry.distance == FisherDistance
        return precision_term_root(geometry.precision_term, target_precision, target_precision)^2
    end
    return last(plain_weights(geometry, width, target_precision))
end

"""
The operator of `geometry`'s norm over `width` gaps of `pass!`'s game at target precision `target_precision`, divided by
PALC's state weight `θ = palc_state_weight` past PALC's own `(1 - θ) Δλ²` (one for PathFollowing): a `FisherMetric`
over the game's `logit_ranges`, or for a plain distance PathFollowing's `FixedWeights`, `gap / θ` per gap and
`(precision - (1 - θ)) / θ` for the precision (`plain_weights`).
"""
function trace_metric(
        geometry::TraceGeometry{E},
        pass::Any,
        width::Integer,
        target_precision::E,
        palc_state_weight::E,
    )::PathFollowing.Metric where {E <: Real}
    if geometry.distance == FisherDistance
        ranges = checked_ranges(logit_ranges(pass), width)
        return FisherMetric(ranges, geometry.state_weight, geometry.precision_term, target_precision, palc_state_weight)
    end
    gap, precision = plain_weights(geometry, width, target_precision)
    return PathFollowing.FixedWeights([fill(gap / palc_state_weight, width); (precision - (one(E) - palc_state_weight)) / palc_state_weight])
end

"""
`(; palc_state_weight, dot)`: PALC's `θ` and `dotθ` for `geometry` over a trace of `width` gaps of `pass!`'s game to
precision `target_precision`. PALC's norm of `[Δx; Δλ]`, `x = [w; λ]`, is `θ ⟨Δx, Δx⟩ + (1 - θ) Δλ²`, which `dotθ`
(`trace_metric`) makes the trace's norm up to the target (`FisherMetric` past it). `θ` comes from the norm's smallest
precision weight (`palc_state_weight_under`). Every `state_weight` in (0, 1) is valid, at every width; another throws
an `ArgumentError`. A Fisher distance reads the game's `logit_ranges`.
"""
function palc_geometry(geometry::TraceGeometry{E}, pass::Any, width::Integer, target_precision::Real)::NamedTuple where {E <: Real}
    (; state_weight) = geometry
    if !(zero(E) < state_weight < one(E))
        throw(ArgumentError("the state weight $state_weight lies outside (0, 1), where PALC weights the gaps and the precision"))
    end
    target = E(target_precision)
    palc_state_weight = palc_state_weight_under(smallest_precision_weight(geometry, width, target))
    return (; palc_state_weight, dot = metric_dot(trace_metric(geometry, pass, width, target, palc_state_weight), zeros(E, width + 1)))
end

"""PALC's product `state_weight ⟨x, y⟩ + (1 - state_weight) λₓ λ_y` of two branch vectors `[x; λ]` under the trace's `dot`."""
function palc_product(
        dot::BifurcationKit.DotTheta,
        state_weight::E,
        left::AbstractVector{E},
        right::AbstractVector{E},
    )::E where {E <: Real}
    return dot(left, right, last(left), last(right), state_weight)
end

"""
The row `b` with `b ⋅ τ = palc_product(dot, state_weight, vector, τ)` for every branch vector `τ`. `palc_geometry`'s
`dot` scales by its point weights, `⟨x, y⟩ = xᵀ A y`, and `palc_options` cannot replace it.
"""
function palc_border(dot::BifurcationKit.DotTheta, state_weight::E, vector::AbstractVector{E})::AbstractVector{E} where {E <: Real}
    border = state_weight .* dot.apply!(copyto!(similar(vector), vector))
    border[end] += (one(E) - state_weight) * last(vector)
    return border
end
