using BifurcationKit: BifurcationKit
using LinearAlgebra: dot
using LogExpFunctions: log1pexp, logistic
"""
The trace's step length in a state `x = [w; λ]` of `k` payoff gaps `w` at target precision `P`, at `state_weight` (the
gaps' weight). Gaps and precision enter in the units `P w` and `λ / P`, so each distance weighs `Δwᵢ²` by `P²` and `Δλ²` by `1 / P²`:

- `AveragedDistance`: `state_weight P² ‖Δw‖² / k + (1 - state_weight) (Δλ / P)²`.
- `SummedDistance`: `state_weight P² ‖Δw‖² + (Δλ / P)²`.
- `FisherDistance`: `state_weight P² Σ σ(zᵢ)σ(-zᵢ) Δwᵢ² + (Δλ / P)²`, with the Bernoulli Fisher weights of the logits `z = λ w`.
- `FisherLogPrecisionDistance`: `state_weight Σ σ(zᵢ)σ(-zᵢ) Δwᵢ² + (Δλ / (1 + λ))²`: Fisher's payoff gains `w` beside the
  step in `log(1 + λ)` to first order, LogitNash's path variable. Neither term depends on `P`, so the target only caps the trace.

Both Fisher distances freeze their weights at each accepted point. The corrector's reach check measures in logits under
every distance (`within_reach`).
"""
@enum StepDistance AveragedDistance SummedDistance FisherDistance FisherLogPrecisionDistance

"""
    TraceGeometry{E}(; distance, state_weight)

The norm a trace steps in: its `StepDistance`, and `state_weight`, the gaps' weight against the precision's.
The default is the Fisher distance at `state_weight = 0.01`: of the arms benchmarked, it took the fewest steps,
corrector iterates and bytes, with every endpoint on the reference (`benchmark/results/arms-f3244ba-summary.txt`).
"""
Base.@kwdef struct TraceGeometry{E <: Real}
    distance::StepDistance = FisherDistance
    state_weight::E = 0.01f0
end

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
`(gap, precision)`: the weights of `Σ Δwᵢ²` and `Δλ²` in `geometry`'s plain norm over `width` gaps at target precision
`target_precision` (`coordinate_weight`, `precision_weight`, `StepDistance`). A target whose square leaves `E`'s
range throws `ArgumentError`; `FisherLogPrecisionDistance` does not square it.
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
    throw(ArgumentError("target precision $target_precision squares out of range in $E; use FisherLogPrecisionDistance or a wider type"))
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

"""`KL(Bernoulli(logistic(from)) ‖ Bernoulli(logistic(to)))`, from the logits, exact where a probability rounds to 0 or 1."""
function bernoulli_kl(from::E, to::E)::E where {E <: Real}
    probability = logistic(from)
    return probability * (log1pexp(-to) - log1pexp(-from)) + (one(E) - probability) * (log1pexp(to) - log1pexp(from))
end

"""The largest `bernoulli_kl` over the logits of two points `[w; λ]` (`logits_at`), one logit at a time."""
function largest_kl(from::AbstractVector{E}, to::AbstractVector{E})::E where {E <: Real}
    # Rounding can make a KL slightly negative, so the search starts below every value.
    largest = typemin(E)
    for index in firstindex(from):(lastindex(from) - 1)
        largest = max(largest, bernoulli_kl(last(from) * from[index], last(to) * to[index]))
    end
    return largest
end

"""
`weights!` of a `FisherDistance` trace's `PointWeightedDot`: each gap's Bernoulli Fisher weight `σ(z)σ(-z)` at the
accepted point's logits `z` times `gap_scale`, and the constant `precision_weight` for the precision.
"""
struct FisherWeights{E <: Real}
    gap_scale::E
    precision_weight::E
end

function (weighting::FisherWeights{E})(weights::AbstractVector{E}, point::BifurcationKit.BorderedArray)::AbstractVector{E} where {E <: Real}
    precision, gaps = last(point.u), payoff_gaps(point.u)
    payoff_gaps(weights) .= weighting.gap_scale .* logistic.(precision .* gaps) .* logistic.(.-(precision .* gaps))
    weights[end] = weighting.precision_weight
    return weights
end

"""`(Δλ / (1 + λ))²` per `Δλ²`, `1 / (1 + λ)²`, at precision `precision`."""
function log_precision_scale(precision::E)::E where {E <: Real}
    return inv(one(E) + precision)^2
end

"""
PALC's state weight for a `FisherLogPrecisionDistance` trace to `target_precision`: one minus half of `1 / (1 + P)²`,
the smallest `log_precision_scale` over λ ∈ [0, P]. PALC's own `Δλ²` term then stays under the distance's, so the
dot's precision weight (`FisherLogPrecisionWeights`) stays positive.
"""
function log_precision_palc_state_weight(target_precision::E)::E where {E <: Real}
    return one(E) - log_precision_scale(target_precision) / 2
end

"""
PALC's state weight for a plain-norm trace whose precision weight is `precision` (`plain_weights`): one minus half of
the smaller of `precision` and one. PALC's own `Δλ²` term then stays under the norm's, so the dot's precision weight
stays positive.
"""
function plain_palc_state_weight(precision::E)::E where {E <: Real}
    return one(E) - min(precision, one(E)) / 2
end

"""
`weights!` of a `FisherLogPrecisionDistance` trace's `PointWeightedDot`, under PALC state weight `palc_state_weight`
(`log_precision_palc_state_weight`), both weights at the accepted point:

- each gap: its Fisher weight `σ(z)σ(-z)` times `gap_scale = state_weight / palc_state_weight`;
- the precision: `(log_precision_scale - (1 - palc_state_weight)) / palc_state_weight`.

PALC's `palc_state_weight ⟨Δx, Δx⟩ + (1 - palc_state_weight) Δλ²` is then the distance's norm.
"""
struct FisherLogPrecisionWeights{E <: Real}
    gap_scale::E
    palc_state_weight::E
end

"""The weight of `Δλ` in `weighting`'s dot at precision `precision`."""
function precision_dot_weight(weighting::FisherLogPrecisionWeights{E}, precision::E)::E where {E <: Real}
    scale = log_precision_scale(precision)
    return (scale - (one(E) - weighting.palc_state_weight)) / weighting.palc_state_weight
end

function (weighting::FisherLogPrecisionWeights{E})(weights::AbstractVector{E}, point::BifurcationKit.BorderedArray)::AbstractVector{E} where {E <: Real}
    precision, gaps = last(point.u), payoff_gaps(point.u)
    payoff_gaps(weights) .= weighting.gap_scale .* logistic.(precision .* gaps) .* logistic.(.-(precision .* gaps))
    weights[end] = precision_dot_weight(weighting, precision)
    return weights
end

"""`weights!` of an `AveragedDistance` or `SummedDistance` trace's `PointWeightedDot`: its weights stay those it was built with."""
struct FixedWeights end

function (::FixedWeights)(weights::AbstractVector{E}, ::BifurcationKit.BorderedArray)::AbstractVector{E} where {E <: Real}
    return weights
end

"""
`(; palc_state_weight, dot)`: PALC's `θ` and `dotθ` for `geometry` over a trace of `width` gaps to precision
`target_precision`. PALC's norm of `[Δx; Δλ]`, `x = [w; λ]`, is `palc_state_weight ⟨Δx, Δx⟩ + (1 - palc_state_weight) Δλ²`.
A plain-norm distance (`plain_weights`: `gap`, `precision`) runs PALC at `plain_palc_state_weight(precision)`, and the
inner product weighs each gap by `gap / palc_state_weight` (times its Fisher weight under `FisherDistance`) and the
precision by `(precision - (1 - palc_state_weight)) / palc_state_weight`.

A `FisherLogPrecisionDistance` trace runs PALC at `log_precision_palc_state_weight` instead (`FisherLogPrecisionWeights`).
Every `state_weight` in (0, 1) is valid, at every width; another throws an `ArgumentError`.
"""
function palc_geometry(geometry::TraceGeometry{E}, width::Integer, target_precision::Real)::NamedTuple where {E <: Real}
    (; state_weight) = geometry
    if !(zero(E) < state_weight < one(E))
        throw(ArgumentError("the state weight $state_weight lies outside (0, 1), where PALC weights the gaps and the precision"))
    end
    if geometry.distance == FisherLogPrecisionDistance
        palc_state_weight = log_precision_palc_state_weight(E(target_precision))
        weighting = FisherLogPrecisionWeights(state_weight / palc_state_weight, palc_state_weight)
        # The flat start's logits are zero, where each Fisher weight is 1/4.
        weights = [fill(weighting.gap_scale / 4, width); precision_dot_weight(weighting, zero(E))]
        return (; palc_state_weight, dot = BifurcationKit.DotTheta(BifurcationKit.PointWeightedDot(weighting, weights)))
    end
    gap, precision = plain_weights(geometry, width, target_precision)
    palc_state_weight = plain_palc_state_weight(precision)
    gap_scale = gap / palc_state_weight
    precision_weight = (precision - (one(E) - palc_state_weight)) / palc_state_weight
    if geometry.distance in (AveragedDistance, SummedDistance)
        weights = [fill(gap_scale, width); precision_weight]
        return (; palc_state_weight, dot = BifurcationKit.DotTheta(BifurcationKit.PointWeightedDot(FixedWeights(), weights)))
    end
    weighted = BifurcationKit.PointWeightedDot(FisherWeights(gap_scale, precision_weight), [fill(gap_scale / 4, width); precision_weight])
    return (; palc_state_weight, dot = BifurcationKit.DotTheta(weighted))
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
    border = state_weight .* dot.apply!(copy(vector))
    border[end] += (one(E) - state_weight) * last(vector)
    return border
end
