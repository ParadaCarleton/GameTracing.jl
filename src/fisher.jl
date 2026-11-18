using BifurcationKit: BifurcationKit
using LogExpFunctions: log1pmx
using PathFollowing: PathFollowing
"""The gap of strategy `index` at a point `[w; λ]`: `w[index]`, or zero for the reference strategy, `index = 0`."""
function strategy_gap(point::AbstractVector{E}, index::Integer)::E where {E <: Real}
    return iszero(index) ? zero(E) : point[index]
end

"""The logit of strategy `index` (0 for the reference) at a point `[w; λ]`, less its player's pivot's: `λ (wᵢ - w_pivot)`."""
function centered_logit(point::AbstractVector{E}, index::Integer, pivot::Integer)::E where {E <: Real}
    return last(point) * (strategy_gap(point, index) - strategy_gap(point, pivot))
end

"""
A player's pivot and log normalizer at a point `[w; λ]`, its logits `z = λ w[range]` against its reference. The pivot is
its most probable strategy: the index of its largest positive logit, or 0 for its reference. The normalizer is
`log Σ exp(zᵢ - z_pivot)` over its strategies and its reference (`centered_logit`). Each logit enters as `λ (wᵢ - w_pivot)`,
which keeps its relative accuracy at a large common logit.
"""
function log_normalizer(point::AbstractVector{E}, range::UnitRange{Int})::Tuple{Int, E} where {E <: Real}
    pivot, largest = 0, zero(E)
    for index in range
        if last(point) * point[index] > largest
            pivot, largest = index, last(point) * point[index]
        end
    end
    total = iszero(pivot) ? zero(E) : exp(centered_logit(point, 0, pivot))
    for index in range
        total += index == pivot ? zero(E) : exp(centered_logit(point, index, pivot))
    end
    return (pivot, log(one(E) + total))
end

"""The log probability of strategy `index` (0 for the reference) at a point `[w; λ]`, its player's `log_normalizer` `normalizer`."""
function log_probability(point::AbstractVector{E}, index::Integer, normalizer::Tuple{Int, E})::E where {E <: Real}
    return centered_logit(point, index, first(normalizer)) - last(normalizer)
end

"""
One strategy's share `π_from (eᵈ - 1 - d)` of a player's KL, `d = log π_to - log π_from`, from the log probabilities
`from` and `to`. The shares sum to the KL as both distributions sum to one. Each share is second order in `d`, so the sum
keeps its relative accuracy at a small KL. Past `|d| = 1/2`, where `log1pmx` loses nothing to cancellation, the share is
`π_to - π_from (1 + d)`, which stays finite where `eᵈ` overflows.
"""
function kl_share(from::E, to::E)::E where {E <: Real}
    gap = to - from
    if abs(gap) <= one(E) / 2
        return -exp(from) * log1pmx(expm1(gap))
    end
    return exp(to) - exp(from) * (one(E) + gap)
end

"""
`KL(π_from ‖ π_to)` of one player's mixed strategy, its logits `z = λ w[range]` at two points `[w; λ]`: the sum of each
strategy's `kl_share`, its reference's included, from log probabilities centered on each point's pivot (`log_normalizer`).
It keeps its relative accuracy at large common logits and stays finite where a probability rounds to 0.
"""
function player_kl(from::AbstractVector{E}, to::AbstractVector{E}, range::UnitRange{Int})::E where {E <: Real}
    from_normalizer, to_normalizer = log_normalizer(from, range), log_normalizer(to, range)
    total = kl_share(log_probability(from, 0, from_normalizer), log_probability(to, 0, to_normalizer))
    for index in range
        total += kl_share(log_probability(from, index, from_normalizer), log_probability(to, index, to_normalizer))
    end
    return total
end

"""The largest `player_kl` from `from` to `to`, two points `[w; λ]`, over the players of `ranges` (`logit_ranges`)."""
function largest_kl(ranges::Any, from::AbstractVector{E}, to::AbstractVector{E})::E where {E <: Real}
    # The search starts below every value, so a game of no players passes every limit.
    largest = typemin(E)
    for range in ranges
        largest = max(largest, player_kl(from, to, range))
    end
    return largest
end

"""
    FisherMetric(ranges, state_weight, precision_term, target_precision, palc_state_weight)

`FisherDistance` as an operator `M` on displacements `[Δw; Δλ]` at a point `[w; λ]`, the logits `z = λ w` of each
player in its range of `ranges` (`logit_ranges`), at PALC state weight `θ = palc_state_weight`:

`ΔxᵀMΔx = (state_weight Σₚ Δzₚᵀ Gₚ Δzₚ + (c(λ) - (1 - θ)) Δλ²) / θ`, with `Δz = λ Δw + w Δλ`, up to the target
(`precision_dot_weight` past it).

`Gₚ = diag(qₚ) - qₚqₚᵀ` is the Fisher information of player `p`'s mixed strategy in its logits, `qₚ` the probabilities
of its strategies past the reference: the Hessian of `log(1 + Σ exp zₚ)`. A player with one logit has `σ(z)σ(-z)`.
`c(λ)` is `precision_term`'s weight toward `target_precision` (`precision_term_weight`). PALC adds `(1 - θ) Δλ²` itself,
so its norm is the distance's; PathFollowing passes `θ = 1`. `PathFollowing.frozen_metric` freezes `M` at a point
(`FrozenFisher`).
"""
struct FisherMetric{R, E <: Real} <: PathFollowing.Metric
    ranges::R
    state_weight::E
    precision_term::PrecisionTerm
    target_precision::E
    palc_state_weight::E
end

"""
    FrozenFisher(metric, point, probabilities, references, pivots)

`metric` (`FisherMetric`) frozen at `point = [w; λ]`: each strategy's probability by its logit, each player's reference
probability, and each player's pivot, its most probable strategy's logit index or 0 for its reference. The products
center each player's logit moves on its pivot's move. A move common to every strategy but an extinct reference then
keeps its Fisher length `√p_ref`, which a raw `Σ qᵢΔzᵢ² - (Σ qᵢΔzᵢ)²` rounds away. The buffers follow `point`'s storage.
"""
struct FrozenFisher{M <: FisherMetric, E <: Real, V <: AbstractVector{E}, I <: AbstractVector{Int}}
    metric::M
    point::V
    probabilities::V
    references::V
    pivots::I
end

function PathFollowing.frozen_metric(metric::FisherMetric{R, E}, point::AbstractVector{E})::FrozenFisher{FisherMetric{R, E}, E} where {R, E <: Real}
    players = length(metric.ranges)
    copied, pivots = copyto!(similar(point), point), fill!(similar(point, Int, players), 0)
    probabilities, references = fill!(similar(copied, length(point) - 1), zero(E)), fill!(similar(copied, players), zero(E))
    frozen = FrozenFisher{FisherMetric{R, E}, E, typeof(copied), typeof(pivots)}(metric, copied, probabilities, references, pivots)
    return PathFollowing.freeze_metric!(frozen, point)
end

function PathFollowing.freeze_metric!(frozen::F, point::AbstractVector{E})::F where {E <: Real, F <: FrozenFisher{<:FisherMetric, E}}
    copyto!(frozen.point, point)
    for (player, range) in enumerate(frozen.metric.ranges)
        normalizer = log_normalizer(point, range)
        frozen.pivots[player] = first(normalizer)
        frozen.references[player] = exp(log_probability(point, 0, normalizer))
        for index in range
            frozen.probabilities[index] = exp(log_probability(point, index, normalizer))
        end
    end
    return frozen
end

"""
The weight of `Δλ²` that `precision_term` gives at precision `precision` toward `target_precision`:
`TargetPrecisionTerm` `1 / P²`, `LogPrecisionTerm` `1 / (1 + |λ|)²`. Over `λ ∈ [0, P]` both are smallest at `λ = P`.
"""
function precision_term_weight(precision_term::PrecisionTerm, precision::E, target_precision::E)::E where {E <: Real}
    if precision_term == LogPrecisionTerm
        return inv(one(E) + abs(precision))^2
    end
    return inv(target_precision)^2
end

"""
The weight of `Δλ²` in `frozen`'s dot, before PALC's `θ` divides it: `c(λ) - (1 - θ)` (`FisherMetric`). PALC's `θ`
keeps `1 - θ` within half of `c(P)` (`palc_state_weight_under`), but past `P` the log term's `c(λ)` falls further.
There PALC's share shrinks with it, to `(1 - θ) c(λ) / c(P)`, so the weight stays about `c(λ) / 2`, never negative.
"""
function precision_dot_weight(frozen::FrozenFisher{<:FisherMetric, E})::E where {E <: Real}
    (; precision_term, target_precision, palc_state_weight) = frozen.metric
    weight = precision_term_weight(precision_term, last(frozen.point), target_precision)
    smallest = precision_term_weight(precision_term, target_precision, target_precision)
    return weight - (one(E) - palc_state_weight) * min(one(E), weight / smallest)
end

"""The logit move `Δzᵢ = λ Δwᵢ + wᵢ Δλ` of the displacement `vector = [Δw; Δλ]` at `frozen`'s point."""
function logit_move(frozen::FrozenFisher{<:FisherMetric, E}, vector::AbstractVector{E}, index::Integer)::E where {E <: Real}
    return last(frozen.point) * vector[index] + frozen.point[index] * last(vector)
end

"""`vector`'s logit move of player `player`'s pivot (`FrozenFisher`), zero for its reference."""
function pivot_move(frozen::FrozenFisher{<:FisherMetric, E}, vector::AbstractVector{E}, player::Integer)::E where {E <: Real}
    pivot = frozen.pivots[player]
    return iszero(pivot) ? zero(E) : logit_move(frozen, vector, pivot)
end

"""
Player `player`'s mean logit move under its mixed strategy, less its pivot's move `pivot`: `Σ πᵢ (Δzᵢ - pivot)` over
its strategies in `range` and its reference, whose logit does not move.
"""
function centered_mean(
        frozen::FrozenFisher{<:FisherMetric, E},
        vector::AbstractVector{E},
        player::Integer,
        range::UnitRange{Int},
        pivot::E,
    )::E where {E <: Real}
    total = -frozen.references[player] * pivot
    for index in range
        total += frozen.probabilities[index] * (logit_move(frozen, vector, index) - pivot)
    end
    return total
end

"""
Player `player`'s `Δz_leftᵀ G Δz_right` in `range`: `Σ πᵢ dᵢ dᵢ'` over its strategies and its reference, each `d` a
logit move less the player's mean move, both measured from its pivot's move (`FrozenFisher`).
"""
function player_inner(
        frozen::FrozenFisher{<:FisherMetric, E},
        left::AbstractVector{E},
        right::AbstractVector{E},
        player::Integer,
        range::UnitRange{Int},
    )::E where {E <: Real}
    left_pivot, right_pivot = pivot_move(frozen, left, player), pivot_move(frozen, right, player)
    left_mean, right_mean = centered_mean(frozen, left, player, range, left_pivot), centered_mean(frozen, right, player, range, right_pivot)
    # The reference's move is zero: its deviation is `-(pivot + mean)` on each side.
    total = frozen.references[player] * (left_pivot + left_mean) * (right_pivot + right_mean)
    for index in range
        left_deviation = (logit_move(frozen, left, index) - left_pivot) - left_mean
        total += frozen.probabilities[index] * left_deviation * ((logit_move(frozen, right, index) - right_pivot) - right_mean)
    end
    return total
end

function PathFollowing.metric_inner(frozen::FrozenFisher{<:FisherMetric, E}, left::AbstractVector{E}, right::AbstractVector{E})::E where {E <: Real}
    total = zero(E)
    for (player, range) in enumerate(frozen.metric.ranges)
        total += player_inner(frozen, left, right, player, range)
    end
    (; state_weight, palc_state_weight) = frozen.metric
    return (state_weight * total + precision_dot_weight(frozen) * last(left) * last(right)) / palc_state_weight
end

"""
Write player `player`'s block of `M v`, `gap_scale` times `(G J v)ᵢ = qᵢ dᵢ` (`d` as in `player_inner`), into `out[range]`,
and return its `Σᵢ wᵢ qᵢ dᵢ`. The deviations average to zero, so `Σᵢ qᵢ dᵢ` is the reference's `π_ref (pivot + mean)`. The
pivot's entry closes that sum over the other rounded entries, and the gaps enter less the pivot's gap. A common move
against an extinct reference then stays as small as its Fisher length, which rounded entries summed apart would bury.
"""
function player_apply!(
        out::AbstractVector{E},
        frozen::FrozenFisher{<:FisherMetric, E},
        vector::AbstractVector{E},
        player::Integer,
        range::UnitRange{Int},
        gap_scale::E,
    )::E where {E <: Real}
    pivot_index = frozen.pivots[player]
    pivot = pivot_move(frozen, vector, player)
    mean = centered_mean(frozen, vector, player, range, pivot)
    information, pivot_gap = frozen.references[player] * (pivot + mean), strategy_gap(frozen.point, pivot_index)
    pulled, others = pivot_gap * information, zero(E)
    for index in range
        if index != pivot_index
            share = frozen.probabilities[index] * ((logit_move(frozen, vector, index) - pivot) - mean)
            out[index] = gap_scale * share
            others += out[index]
            pulled += (frozen.point[index] - pivot_gap) * share
        end
    end
    if !iszero(pivot_index)
        out[pivot_index] = gap_scale * information - others
    end
    return pulled
end

# `M v = (state_weight Jᵀ G J v + (c - (1 - θ)) Δλ e_λ) / θ` with `J = [λ I  w]`, one block per player (`player_apply!`).
function PathFollowing.metric_apply!(
        out::AbstractVector{E},
        frozen::FrozenFisher{<:FisherMetric, E},
        vector::AbstractVector{E},
    )::AbstractVector{E} where {E <: Real}
    (; state_weight, palc_state_weight) = frozen.metric
    gap_scale = state_weight * last(frozen.point) / palc_state_weight
    pulled = zero(E)
    for (player, range) in enumerate(frozen.metric.ranges)
        pulled += player_apply!(out, frozen, vector, player, range, gap_scale)
    end
    out[end] = (state_weight * pulled + precision_dot_weight(frozen) * last(vector)) / palc_state_weight
    return out
end

"""
    MetricDot(frozen)

PALC's `dotθ.dot` over a frozen PathFollowing metric `frozen` (`PathFollowing.frozen_metric`): `leftᵀ M right`.
BifurcationKit freezes it again at each accepted point and at the start (`BifurcationKit.reweight!`), so a step's
predictor, corrector and tests use `M` at the point the step starts from.
"""
struct MetricDot{F}
    frozen::F
end

function (dot::MetricDot)(left::AbstractVector{E}, right::AbstractVector{E})::E where {E <: Real}
    return PathFollowing.metric_inner(dot.frozen, left, right)
end

function BifurcationKit.reweight!(dot::MetricDot, point::BifurcationKit.BorderedArray)::Bool
    PathFollowing.freeze_metric!(dot.frozen, point.u)
    return true
end

"""
    MetricApply(frozen, buffer)

PALC's `dotθ.apply!` over the frozen metric `frozen` that `MetricDot` shares: `vector ← M vector` in place, through
`buffer`, a vector of the points' length and storage.
"""
struct MetricApply{F, E <: Real, V <: AbstractVector{E}}
    frozen::F
    buffer::V
end

function (apply::MetricApply{F, E})(vector::AbstractVector{E})::AbstractVector{E} where {F, E <: Real}
    copyto!(apply.buffer, vector)
    return PathFollowing.metric_apply!(vector, apply.frozen, apply.buffer)
end

"""
PALC's `DotTheta` over the trace metric `metric` (`trace_metric`), frozen at `start`, a point `[w; λ]`, until PALC
freezes it at its own start (`MetricDot`, `MetricApply`).
"""
function metric_dot(metric::PathFollowing.Metric, start::AbstractVector{E})::BifurcationKit.DotTheta where {E <: Real}
    frozen = PathFollowing.frozen_metric(metric, start)
    buffer = copyto!(similar(start), start)
    return BifurcationKit.DotTheta(MetricDot(frozen), MetricApply{typeof(frozen), E, typeof(buffer)}(frozen, buffer))
end
