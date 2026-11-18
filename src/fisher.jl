using BifurcationKit: BifurcationKit
using PathFollowing: PathFollowing
"""
The log normalizer `log(1 + Σ exp(z))` of a player's logits `z = λ w[range]` at a point `[w; λ]`, in two parts: the
largest logit or zero, and the log of the sum shifted by it. Their sum rounds away the gaps between large logits.
"""
function log_normalizer(point::AbstractVector{E}, range::UnitRange{Int})::NTuple{2, E} where {E <: Real}
    shift = zero(E)
    for index in range
        shift = max(shift, last(point) * point[index])
    end
    total = exp(-shift)
    for index in range
        total += exp(last(point) * point[index] - shift)
    end
    return (shift, log(total))
end

"""The log probability of a strategy with logit `logit`, its player's `log_normalizer` `normalizer`."""
function log_probability(logit::E, normalizer::NTuple{2, E})::E where {E <: Real}
    return (logit - first(normalizer)) - last(normalizer)
end

"""
`KL(π_from ‖ π_to)` of one player's mixed strategy, its logits `z = λ w[range]` at two points `[w; λ]`:
`Σ π_from (z_from - z_to) + A(z_to) - A(z_from)`, `A` the `log_normalizer`. It stays finite where a probability rounds
to 0, and one logit gives the Bernoulli KL.
"""
function player_kl(from::AbstractVector{E}, to::AbstractVector{E}, range::UnitRange{Int})::E where {E <: Real}
    from_normalizer, to_normalizer = log_normalizer(from, range), log_normalizer(to, range)
    total = (first(to_normalizer) - first(from_normalizer)) + (last(to_normalizer) - last(from_normalizer))
    for index in range
        logit = last(from) * from[index]
        total += exp(log_probability(logit, from_normalizer)) * (logit - last(to) * to[index])
    end
    return total
end

"""The largest `player_kl` from `from` to `to`, two points `[w; λ]`, over the players of `ranges` (`logit_ranges`)."""
function largest_kl(ranges::Any, from::AbstractVector{E}, to::AbstractVector{E})::E where {E <: Real}
    # Rounding can make a KL slightly negative, so the search starts below every value.
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

`ΔxᵀMΔx = (state_weight Σₚ Δzₚᵀ Gₚ Δzₚ + (c(λ) - (1 - θ)) Δλ²) / θ`, with `Δz = λ Δw + w Δλ`.

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
        frozen.references[player] = exp(log_probability(zero(E), normalizer))
        pivot, largest = 0, frozen.references[player]
        for index in range
            frozen.probabilities[index] = exp(log_probability(last(point) * point[index], normalizer))
            if frozen.probabilities[index] > largest
                pivot, largest = index, frozen.probabilities[index]
            end
        end
        frozen.pivots[player] = pivot
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

"""The weight of `Δλ²` in `frozen`'s dot, before PALC's `θ` divides it: `c(λ) - (1 - θ)` (`FisherMetric`)."""
function precision_dot_weight(frozen::FrozenFisher{<:FisherMetric, E})::E where {E <: Real}
    (; precision_term, target_precision, palc_state_weight) = frozen.metric
    return precision_term_weight(precision_term, last(frozen.point), target_precision) - (one(E) - palc_state_weight)
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

# `M v = (state_weight Jᵀ G J v + (c - (1 - θ)) Δλ e_λ) / θ` with `J = [λ I  w]`, so `(G J v)ᵢ = qᵢ dᵢ`, `d` as in `player_inner`.
function PathFollowing.metric_apply!(
        out::AbstractVector{E},
        frozen::FrozenFisher{<:FisherMetric, E},
        vector::AbstractVector{E},
    )::AbstractVector{E} where {E <: Real}
    (; state_weight, palc_state_weight) = frozen.metric
    gap_scale = state_weight * last(frozen.point) / palc_state_weight
    pulled = zero(E)
    for (player, range) in enumerate(frozen.metric.ranges)
        pivot = pivot_move(frozen, vector, player)
        mean = centered_mean(frozen, vector, player, range, pivot)
        for index in range
            information = frozen.probabilities[index] * ((logit_move(frozen, vector, index) - pivot) - mean)
            out[index] = gap_scale * information
            pulled += frozen.point[index] * information
        end
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
