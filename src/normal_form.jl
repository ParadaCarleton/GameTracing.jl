"""
    NormalFormGains(payoffs)

The `pass!` of a normal-form game with any number of strategies per player. `payoffs[p][s]` is player `p`'s payoff at
the pure profile `s`; every array has one type and one size, each player's strategy count along its own axis. Player
`p`'s logits are `z_pk = log(π_pk / π_p1)` for its strategies `k ≥ 2`, against its first strategy, stacked player by
player (`logit_ranges`). Its gains are `G_pk = U_p(k, π) - U_p(1, π)`, its expected payoffs against the rivals' mixed
strategies, which do not depend on the precision. A pass writes into work arrays from `pass_workspace`; called
without them, it builds its own.
"""
struct NormalFormGains{N, A <: AbstractArray{<:Real, N}}
    payoffs::NTuple{N, A}

    function NormalFormGains(payoffs::NTuple{N, A})::NormalFormGains{N, A} where {N, A <: AbstractArray{<:Real, N}}
        if !allequal(size.(payoffs))
            throw(ArgumentError("every player's payoff array must have one size; got $(size.(payoffs))"))
        end
        return new{N, A}(payoffs)
    end
end

function NormalFormGains(payoffs::Tuple)::NormalFormGains
    throw(ArgumentError("payoffs must be arrays of one type with one axis per player; got $(typeof.(payoffs))"))
end

"""The element type of the payoffs of `gains`."""
function Base.eltype(::Type{NormalFormGains{N, A}})::Type{eltype(A)} where {N, A}
    return eltype(A)
end

"""Each player's range in the stacked logits of `gains`: its strategies after the first."""
function logit_ranges(gains::NormalFormGains{N})::NTuple{N, UnitRange{Int}} where {N}
    counts = size(first(gains.payoffs)) .- 1
    ends = cumsum(counts)
    return UnitRange.(ends .- counts .+ 1, ends)
end

"""The number of logits of `gains`: every player's strategy count less one, summed."""
function logit_count(gains::NormalFormGains)::Integer
    return last(last(logit_ranges(gains)))
end

"""
The work arrays of a `gains` pass at logits of element type `E`: each player's mixed strategy, and room for one player's
expected payoffs and its payoff sensitivities to each rival's strategies.
"""
function pass_workspace(gains::NormalFormGains, ::AbstractVector{E})::NamedTuple where {E <: Real}
    counts = size(first(gains.payoffs))
    widest = maximum(counts)
    return (; strategies = zeros.(E, counts), expected = zeros(E, widest), sensitivities = zeros.(E, widest, counts))
end

"""Write into `strategy` the mixed strategy `softmax([0; logits])` of a player whose logits are against its first strategy."""
function reference_softmax!(strategy::AbstractVector{E}, logits::AbstractVector{E})::AbstractVector{E} where {E <: Real}
    shift = max(zero(E), maximum(logits; init = zero(E)))
    strategy[begin] = exp(-shift)
    view(strategy, (firstindex(strategy) + 1):lastindex(strategy)) .= exp.(logits .- shift)
    strategy ./= sum(strategy)
    return strategy
end

"""Each player's mixed strategy at the stacked `logits` of `gains`, in fresh vectors."""
function mixed_strategies(gains::NormalFormGains, logits::AbstractVector{E})::Tuple where {E <: Real}
    return reference_softmax!.(zeros.(E, size(first(gains.payoffs))), view.((logits,), logit_ranges(gains)))
end

"""
The weight of player `other` in a column of a payoff array: its probability `strategy[index]` of its strategy there,
or one for the first player, whose strategy runs down the column, and for `player`, whose payoffs the column holds.
"""
function column_weight(strategy::AbstractVector{E}, index::Integer, other::Integer, player::Integer)::E where {E <: Real}
    return other == 1 || other == player ? one(E) : strategy[index]
end

"""The product of `weights` but the one of player `skipped`."""
function weight_without(weights::NTuple{N, E}, skipped::Integer)::E where {N, E <: Real}
    return prod(ifelse.(ntuple(identity, Val(N)) .== skipped, one(E), weights))
end

"""
Throw a `DimensionMismatch` unless the work arrays `work` (`pass_workspace`) fit `payoff`: each player's strategy on
its axis, and room for `player`'s expected payoffs and sensitivities. `walk_player!` indexes them unchecked after it.
"""
function require_fitting_workspace(work::NamedTuple, payoff::AbstractArray, player::Integer)::Nothing
    (; strategies, sensitivities, expected) = work
    fitting = length(strategies) == length(sensitivities) == ndims(payoff) && length(expected) >= size(payoff, player)
    for (other, strategy) in pairs(strategies)
        fitting &= axes(strategy, 1) == axes(payoff, other) == axes(sensitivities[other], 2)
        fitting &= axes(sensitivities[other], 1) == axes(expected, 1)
    end
    if !fitting
        throw(DimensionMismatch("work arrays of sizes $(size.(strategies)) do not fit payoffs of size $(size(payoff))"))
    end
    return nothing
end

"""
    walk_player!(work, payoff, player; slopes)

`player`'s expected payoffs `U_p(k, π)` into `work.expected`, at the strategies `π` in `work.strategies`, and with
`slopes` its payoff sensitivities `∂U_p(k) / ∂π_qj` to each rival `q` into `work.sensitivities[q]`. One walk over
`payoff`, a column at a time: the first player's strategy runs down a column, the other players' weights
(`column_weight`) stay fixed along it.
"""
function walk_player!(work::NamedTuple, payoff::AbstractArray{<:Real, N}, player::Integer; slopes::Bool)::AbstractVector where {N}
    require_fitting_workspace(work, payoff, player)
    (; strategies, sensitivities) = work
    expected = view(work.expected, axes(payoff, player))
    zeroed = zero(eltype(expected))
    fill!(expected, zeroed)
    if slopes
        for rival in eachindex(sensitivities)
            fill!(@view(sensitivities[rival][axes(payoff, player), axes(payoff, rival)]), zeroed)
        end
    end
    first_strategy = first(strategies)
    players = ntuple(identity, Val(N))
    @inbounds for column in CartesianIndices(Base.tail(axes(payoff)))
        values = @view payoff[:, column]
        indices = (firstindex(first_strategy), Tuple(column)...)
        weights = column_weight.(strategies, indices, players, player)
        everything = prod(weights)
        if player == 1
            for strategy in axes(payoff, 1)
                expected[strategy] += everything * values[strategy]
            end
            if slopes
                for rival in 2:N
                    sensitivity, others = sensitivities[rival], weight_without(weights, rival)
                    for strategy in axes(payoff, 1)
                        sensitivity[strategy, indices[rival]] += others * values[strategy]
                    end
                end
            end
        else
            own, mixed = indices[player], zero(promote_type(eltype(expected), eltype(payoff)))
            for strategy in axes(payoff, 1)
                mixed += first_strategy[strategy] * values[strategy]
            end
            expected[own] += everything * mixed
            if slopes
                sensitivity = first(sensitivities)
                for strategy in axes(payoff, 1)
                    sensitivity[own, strategy] += everything * values[strategy]
                end
                for rival in 2:N
                    if rival != player
                        sensitivities[rival][own, indices[rival]] += weight_without(weights, rival) * mixed
                    end
                end
            end
        end
    end
    return expected
end

"""
    write_gains!(work, gain, gains, logits)

Each player's gains `G_pk = U_p(k, π) - U_p(1, π)` of `gains` at the stacked `logits`, into `gain`. The mixed strategies
`π` stay in `work.strategies` (`pass_workspace`).
"""
function write_gains!(
        work::NamedTuple,
        gain::AbstractVector,
        gains::NormalFormGains,
        logits::AbstractVector{E},
    )::Nothing where {E <: Real}
    ranges = logit_ranges(gains)
    reference_softmax!.(work.strategies, view.((logits,), ranges))
    for (player, payoff) in pairs(gains.payoffs)
        write_player_gain!(view(gain, ranges[player]), walk_player!(work, payoff, player; slopes = false))
    end
    return nothing
end

"""Write into `gain` a player's gains: its `expected` payoffs after the first strategy's, less the first's."""
function write_player_gain!(gain::AbstractVector, expected::AbstractVector)::Nothing
    gain .= view(expected, (firstindex(expected) + 1):lastindex(expected)) .- first(expected)
    return nothing
end

function (gains::NormalFormGains)(
        work::NamedTuple,
        gain::AbstractVector,
        gain_slope::AbstractMatrix,
        gain_precision_slope::AbstractVector,
        logits::AbstractVector{E},
        ::Real,
    )::Nothing where {E <: Real}
    ranges = logit_ranges(gains)
    reference_softmax!.(work.strategies, view.((logits,), ranges))
    fill!(gain_slope, zero(E))
    fill!(gain_precision_slope, zero(E))
    for (player, payoff) in pairs(gains.payoffs)
        expected = walk_player!(work, payoff, player; slopes = true)
        write_player_gain!(view(gain, ranges[player]), expected)
        for (rival, rival_range) in pairs(ranges)
            if rival != player
                write_gain_slope!(@view(gain_slope[ranges[player], rival_range]), payoff, work, player, rival)
            end
        end
    end
    return nothing
end

function (gains::NormalFormGains)(
        gain::AbstractVector,
        gain_slope::AbstractMatrix,
        gain_precision_slope::AbstractVector,
        logits::AbstractVector,
        precision::Real,
    )::Nothing
    return gains(pass_workspace(gains, logits), gain, gain_slope, gain_precision_slope, logits, precision)
end

"""
    write_gain_slope!(block, payoff, work, player, rival)

`∂G_pk / ∂z_qj` of `player` `p` in the logits of `rival` `q`, into `block`, from the sensitivities `D_ki = ∂U_p(k) /
∂π_qi` and expected payoffs in `work` (`walk_player!`). With `∂π_qi / ∂z_qj = π_qi (δ_ij - π_qj)` and
`Σ_i D_ki π_qi = U_p(k)`, `∂U_p(k) / ∂z_qj = π_qj (D_kj - U_p(k))`; a gain takes its strategy's row less the first's.
"""
function write_gain_slope!(
        block::AbstractMatrix{E},
        payoff::AbstractArray,
        work::NamedTuple,
        player::Integer,
        rival::Integer,
    )::Nothing where {E <: Real}
    own, other = axes(payoff, player), axes(payoff, rival)
    if size(block) != (length(own) - 1, length(other) - 1)
        throw(DimensionMismatch("a block of size $(size(block)) does not fit $(length(own)) by $(length(other)) strategies"))
    end
    # Loops: a broadcast over small views spends more on alias checks and axes than on its sums. One bounds check per
    # array lets the inner loop vectorize.
    rival_strategy, sensitivity, expected = work.strategies[rival], work.sensitivities[rival], work.expected
    checkbounds(rival_strategy, other)
    checkbounds(sensitivity, own, other)
    checkbounds(expected, own)
    # Block row and column `i` hold strategies `i + shift` of `player` and `rival`: the first strategy has none.
    row_shift, column_shift = first(own) + 1 - firstindex(block, 1), first(other) + 1 - firstindex(block, 2)
    first_expected = expected[first(own)]
    @inbounds for column in axes(block, 2)
        later = column + column_shift
        weight, first_sensitivity = rival_strategy[later], sensitivity[first(own), later]
        # The broadcast this loop replaced summed in this order. The Float32 noise bound in `opening_tests.jl` is tight.
        @simd for row in axes(block, 1)
            block[row, column] = weight * (((sensitivity[row + row_shift, later] - expected[row + row_shift]) - first_sensitivity) + first_expected)
        end
    end
    return nothing
end
