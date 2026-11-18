using BifurcationKit: BifurcationKit
using LogExpFunctions: log1pmx
using PathFollowing: PathFollowing
"""The gap of strategy `index` at a point `[w; λ]`: `w[index]`, or zero for the reference strategy, `index = 0`."""
function strategy_gap(point::AbstractVector{E}, index::Integer)::E where {E <: Real}
    return iszero(index) ? zero(E) : point[index]
end

"""
The logit of strategy `index` (0 for the reference) at a point `[w; λ]`, less its player's pivot's: `λ (wᵢ - w_pivot)`.
Past `floatmax` the gaps' difference becomes `λ wᵢ - λ w_pivot`, a sum of two terms of one sign.
"""
function centered_logit(point::AbstractVector{E}, index::Integer, pivot::Integer)::E where {E <: Real}
    precision, gap, pivot_gap = last(point), strategy_gap(point, index), strategy_gap(point, pivot)
    difference = gap - pivot_gap
    return isfinite(difference) ? precision * difference : precision * gap - precision * pivot_gap
end

"""
A player's pivot and log normalizer at a point `[w; λ]`, its logits `z = λ w[range]` against its reference. The pivot is
its most probable strategy: the index of its largest logit, or 0 for its reference. The pivot comes from the gaps `w`,
largest for `λ > 0` and smallest for `λ < 0`, so an overflowing product `λ wᵢ` cannot choose it. The normalizer is
`log Σ exp(zᵢ - z_pivot)` over its strategies and its reference (`centered_logit`). Each logit enters as `λ (wᵢ - w_pivot)`,
which keeps its relative accuracy at a large common logit. No centered logit is positive, so the normalizer is finite.
`log1p` of the other strategies' sum keeps its relative accuracy where that sum is under `eps`.
"""
function log_normalizer(point::AbstractVector{E}, range::UnitRange{Int})::Tuple{Int, E} where {E <: Real}
    pivot, direction = 0, sign(last(point))
    for index in range
        if direction * point[index] > direction * strategy_gap(point, pivot)
            pivot = index
        end
    end
    total = iszero(pivot) ? zero(E) : exp(centered_logit(point, 0, pivot))
    for index in range
        total += index == pivot ? zero(E) : exp(centered_logit(point, index, pivot))
    end
    return (pivot, log1p(total))
end

"""The log probability of strategy `index` (0 for the reference) at a point `[w; λ]`, its player's `log_normalizer` `normalizer`."""
function log_probability(point::AbstractVector{E}, index::Integer, normalizer::Tuple{Int, E})::E where {E <: Real}
    return centered_logit(point, index, first(normalizer)) - last(normalizer)
end

"""
One strategy's share `π_from (eᵈ - 1 - d)` of a player's KL, `d = log π_to - log π_from`, from the log probabilities
`from` and `to`. The shares sum to the KL as both distributions sum to one. Each share is second order in `d`, so the sum
keeps its relative accuracy at a small KL. Past `|d| = 1/2`, where `log1pmx` loses nothing to cancellation, the share is
`π_to - π_from (1 + d)`, which stays finite where `eᵈ` overflows. Where `π_from` rounds to 0, the share is `π_to`, its
limit as `π_from → 0`, also at an infinite `d`.
"""
function kl_share(from::E, to::E)::E where {E <: Real}
    probability, gap = exp(from), to - from
    if iszero(probability)
        return exp(to)
    elseif abs(gap) <= one(E) / 2
        return -probability * log1pmx(expm1(gap))
    end
    return exp(to) - probability * (one(E) + gap)
end

"""
`KL(π_from ‖ π_to)` of one player's mixed strategy, its logits `z = λ w[range]` at two points `[w; λ]`: the sum of each
strategy's `kl_share`, its reference's included, from log probabilities centered on each point's pivot (`log_normalizer`).
It keeps its relative accuracy at large common logits and stays finite where a probability rounds to 0.
It is finite while every log probability at `to`, a centered logit `λ (wᵢ - w_pivot)` less the normalizer, exceeds `-floatmax`.
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

`ΔxᵀMΔx = (state_weight Σₚ Δzₚᵀ Gₚ Δzₚ + (ω(λ) - (1 - θ)) Δλ²) / θ`, with `Δz = λ Δw + w Δλ`.

`Gₚ = diag(qₚ) - qₚqₚᵀ` is the Fisher information of player `p`'s mixed strategy in its logits, `qₚ` the probabilities
of its strategies past the reference: the Hessian of `log(1 + Σ exp zₚ)`. A player with one logit has `σ(z)σ(-z)`.
PALC adds `(1 - θ) Δλ²` itself, so its norm gives `Δλ²` the weight `ω(λ)`. `c(λ)` is `precision_term`'s weight toward
`target_precision` `P` (`precision_term_root`), and `ω(λ)` is `c(λ)` while `c(λ) ≥ c(P)`, up to `P` for the log term.
Past `P` the log term's `c(λ) - (1 - θ)` can turn negative, so `ω(λ)` is `c(λ) + (1 - θ) (1 - c(λ) / c(P))`
(`precision_dot_root`). The trace ends at `P`, so only its last point is past it. PathFollowing passes `θ = 1`, which
gives `ω = c`. `1 - θ` must lie under `c(P)`, or the constructor throws an `ArgumentError`.
`PathFollowing.frozen_metric` freezes `M` at a point (`FrozenFisher`).
"""
struct FisherMetric{R, E <: Real} <: PathFollowing.Metric
    ranges::R
    state_weight::E
    precision_term::PrecisionTerm
    target_precision::E
    palc_state_weight::E

    function FisherMetric(
            ranges::R,
            state_weight::E,
            precision_term::PrecisionTerm,
            target_precision::E,
            palc_state_weight::E,
        )::FisherMetric{R, E} where {R, E <: Real}
        smallest = precision_term_root(precision_term, target_precision, target_precision)
        if !(zero(E) < palc_state_weight <= one(E) && share_ratio(palc_state_weight, smallest) < one(E))
            throw(ArgumentError("PALC's state weight $palc_state_weight lies outside (1 - c(P), 1] for √c(P) = $smallest"))
        end
        return new{R, E}(ranges, state_weight, precision_term, target_precision, palc_state_weight)
    end
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
The root `√c(λ)` of the weight of `Δλ²` that `precision_term` gives at precision `precision` toward
`target_precision`: `TargetPrecisionTerm` `1 / P`, `LogPrecisionTerm` `1 / (1 + |λ|)`. Over `λ ∈ [0, P]` both are
smallest at `λ = P`. The root is positive for every finite `λ`; `c` itself underflows past `1 / √floatmin`.
"""
function precision_term_root(precision_term::PrecisionTerm, precision::E, target_precision::E)::E where {E <: Real}
    if precision_term == LogPrecisionTerm
        return inv(one(E) + abs(precision))
    end
    return inv(target_precision)
end

"""PALC's share `1 - θ`, `θ = palc_state_weight`, over the weight `c = root²`: divided by `root` twice, as `c` can underflow."""
function share_ratio(palc_state_weight::E, root::E)::E where {E <: Real}
    return ((one(E) - palc_state_weight) / root) / root
end

"""
The root `√(ω(λ) - (1 - θ))` of the weight of `Δλ²` in `frozen`'s dot, before PALC's `θ` divides it (`FisherMetric`).
While `c(λ) ≥ c(P)` the weight is `c(λ) - (1 - θ)`. Past `P` PALC's share shrinks with the log term, to
`(1 - θ) c(λ) / c(P)`. Both are `c(λ) (1 - (1 - θ) / max(c(λ), c(P)))`, whose root `√c(λ) √(1 - …)` is positive for
every finite `λ`, as `1 - θ < c(P)`. The products apply the root to each side, `(root Δλ) (root Δλ')`.
The dot's precision term keeps its relative accuracy while each side's `root Δλ` and their product are normal floats.
"""
function precision_dot_root(frozen::FrozenFisher{<:FisherMetric, E})::E where {E <: Real}
    (; precision_term, target_precision, palc_state_weight) = frozen.metric
    root = precision_term_root(precision_term, last(frozen.point), target_precision)
    smallest = precision_term_root(precision_term, target_precision, target_precision)
    return root * sqrt(one(E) - share_ratio(palc_state_weight, max(root, smallest)))
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
    root = precision_dot_root(frozen)
    return (state_weight * total + (root * last(left)) * (root * last(right))) / palc_state_weight
end

"""
Write player `player`'s block of `M v`, `gap_scale` times `(G J v)ᵢ = qᵢ dᵢ` (`d` as in `player_inner`), into `out[range]`,
and return its `Σᵢ wᵢ qᵢ dᵢ`. The deviations average to zero, so `Σᵢ qᵢ dᵢ` is the reference's `π_ref (pivot + mean)`. The
pivot's entry closes that sum over the other rounded entries, and the gaps enter less the pivot's gap. A common move
against an extinct reference then stays as small as its Fisher length, which rounded entries summed apart would bury.

The block's sum is within `eps ((n + L) Σᵢ |outᵢ| + gap_scale π_ref Σᵢ qᵢ (|λ Δwᵢ| + |wᵢ Δλ|))` of `M`'s, for `n`
strategies and `L` the largest `|log π|` among them. The first term is the entries' rounding and the second the logit
moves'. The bound holds under Higham's standard model `fl(x ∘ y) = (x ∘ y) (1 + δ)`: while every probability, product
and partial sum is a normal float or zero. Where one underflows, the block stays finite but loses the bound. Entries
of size `s` hold no nonzero sum under about `eps s`, so a large differential move can sum to zero.
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

# `M v = (state_weight Jᵀ G J v + (ω - (1 - θ)) Δλ e_λ) / θ` with `J = [λ I  w]`, one block per player (`player_apply!`).
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
    root = precision_dot_root(frozen)
    out[end] = (state_weight * pulled + root * (root * last(vector))) / palc_state_weight
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
