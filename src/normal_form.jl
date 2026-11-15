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
The work arrays of a `gains` pass at logits of element type `E`: each player's mixed strategy, room for one player's
expected payoffs and its payoff sensitivities to each rival's strategies, and the tensors and joint weights of
`contract_payoff!`.
"""
function pass_workspace(gains::NormalFormGains, ::AbstractVector{E})::NamedTuple where {E <: Real}
    counts = size(first(gains.payoffs))
    widest, entries = maximum(counts), prod(counts)
    tensors = [zeros(E, entries) for _ in 1:tensor_count(length(counts) - 1)]
    sensitivities = zeros.(E, widest, counts)
    spare, joint = zeros(E, entries), zeros(E, entries)
    return (; strategies = zeros.(E, counts), expected = zeros(E, widest), sensitivities, tensors, spare, joint)
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

"""The axis of the rival at `position` among the axes of a payoff array but `player`'s."""
function rival_axis(position::Integer, player::Integer)::Integer
    return position < player ? position : position + 1
end

"""
The tensors `contract_payoff!` holds at once for `rivals` rivals: one per halving of the rivals down to one
(`write_first_marginal!`), and at least one.
"""
function tensor_count(rivals::Integer)::Integer
    return rivals <= 2 ? 1 : 1 + tensor_count(cld(rivals, 2))
end

"""
Write into the front of `buffer` the weights of the adjacent axes `group` taken as one axis: at each joint index, the
product of each axis' strategy weight, the first axis fastest. Return that front.
"""
@inline function group_weights!(
        buffer::AbstractVector{E},
        strategies::NTuple{N, AbstractVector{E}},
        group::AbstractUnitRange{<:Integer},
    )::AbstractVector{E} where {N, E <: Real}
    filled = 1
    buffer[1] = one(E)
    for axis in group
        weights = strategies[axis]
        # The last slice first: each product reads the front before the first slice overwrites it.
        for slice in reverse(eachindex(weights)), entry in filled:-1:1
            buffer[(slice - 1) * filled + entry] = buffer[entry] * weights[slice]
        end
        filled *= length(weights)
    end
    return view(buffer, 1:filled)
end

"""
    contract_axes!(target, source, dims, group, weights)

Contract the adjacent axes `group` of the tensor `source` of size `dims`, read in linear order, into `target`: the
slices along the group, each times its joint weight in `weights` (`group_weights!`), summed. Return the new size, one
along `group`. A block past the group holds one `stride`-long run per slice. The inner loop runs along a run when it
is at least as long as the slices are many, else across the blocks; each entry adds its slices in order either way.
"""
@inline function contract_axes!(
        target::AbstractVector{E},
        source::AbstractArray{<:Real},
        dims::NTuple{N, Integer},
        group::AbstractUnitRange{<:Integer},
        weights::AbstractVector{E},
    )::NTuple{N, Integer} where {N, E <: Real}
    stride = prod(ntuple(axis -> axis < first(group) ? dims[axis] : 1, Val(N)))
    width = prod(ntuple(axis -> axis in group ? dims[axis] : 1, Val(N)))
    blocks = prod(ntuple(axis -> axis > last(group) ? dims[axis] : 1, Val(N)))
    fitting = length(source) >= stride * width * blocks && length(target) >= stride * blocks && length(weights) == width
    if Base.has_offset_axes(target, source, weights) || !fitting
        throw(DimensionMismatch("cannot contract axes $group of size $dims from $(length(source)) into $(length(target))"))
    end
    if stride >= width
        first_weight = first(weights)
        @inbounds for block in 0:(blocks - 1)
            into, from = block * stride, block * stride * width
            @simd for inner in 1:stride
                target[into + inner] = first_weight * source[from + inner]
            end
            for slice in 2:width
                weight, offset = weights[slice], from + (slice - 1) * stride
                @simd for inner in 1:stride
                    target[into + inner] += weight * source[offset + inner]
                end
            end
        end
    else
        first_weight, step = first(weights), stride * width
        @inbounds for inner in 1:stride
            @simd for block in 0:(blocks - 1)
                target[block * stride + inner] = first_weight * source[block * step + inner]
            end
            for slice in 2:width
                weight, offset = weights[slice], (slice - 1) * stride + inner
                @simd for block in 0:(blocks - 1)
                    target[block * stride + inner] += weight * source[block * step + offset]
                end
            end
        end
    end
    return ntuple(axis -> axis in group ? 1 : dims[axis], Val(N))
end

"""
    contract_rivals!(target, spare, joint, source, dims, positions, player, strategies)

Contract the axes of the rivals at `positions` (`rival_axis`) of the tensor `source` of size `dims` with their mixed
`strategies` into `target`. The axes form one group (`contract_axes!`), or two around `player`'s own: the upper group
goes first, into `spare`. `joint` holds each group's weights (`group_weights!`). Return the new size.
"""
@inline function contract_rivals!(
        target::AbstractVector{E},
        spare::AbstractVector{E},
        joint::AbstractVector{E},
        source::AbstractArray{<:Real},
        dims::NTuple{N, Integer},
        positions::AbstractUnitRange{<:Integer},
        player::Integer,
        strategies::NTuple{N, AbstractVector{E}},
    )::NTuple{N, Integer} where {N, E <: Real}
    lowest, highest = rival_axis(first(positions), player), rival_axis(last(positions), player)
    if lowest < player < highest
        upper, lower = (player + 1):highest, lowest:(player - 1)
        reduced = contract_axes!(spare, source, dims, upper, group_weights!(joint, strategies, upper))
        return contract_axes!(target, spare, reduced, lower, group_weights!(joint, strategies, lower))
    end
    return contract_axes!(target, source, dims, lowest:highest, group_weights!(joint, strategies, lowest:highest))
end

"""
Write into `marginal` the tensor `tensor` of size `dims`, all of whose axes but `player`'s and `rival`'s have size one,
as a matrix: `player`'s strategies down, `rival`'s across.
"""
function write_marginal!(
        marginal::AbstractMatrix,
        tensor::AbstractArray{<:Real},
        dims::NTuple{N, Integer},
        player::Integer,
        rival::Integer,
    )::Nothing where {N}
    own_count, rival_count = dims[player], dims[rival]
    if prod(dims) != own_count * rival_count
        throw(DimensionMismatch("a tensor of size $dims has axes besides those of players $player and $rival"))
    end
    checkbounds(marginal, 1:own_count, 1:rival_count)
    checkbounds(tensor, 1:(own_count * rival_count))
    own_stride, rival_stride = player < rival ? (1, own_count) : (rival_count, 1)
    @inbounds for column in 1:rival_count, row in 1:own_count
        marginal[row, column] = tensor[1 + (row - 1) * own_stride + (column - 1) * rival_stride]
    end
    return nothing
end

"""
Write into `work.sensitivities` the marginals of the tensor `tensor` of size `dims` for rivals `first_rival` and
`second_rival` (`write_marginal!`): all of its axes but `player`'s and theirs have size one. One pass: each entry, times
one rival's weight, adds to the other rival's marginal, the other rival's strategies in order.
"""
function write_pair_marginals!(
        work::NamedTuple,
        tensor::AbstractArray{<:Real},
        dims::NTuple{N, Integer},
        player::Integer,
        first_rival::Integer,
        second_rival::Integer,
    )::Nothing where {N}
    own_count, first_count, second_count = dims[player], dims[first_rival], dims[second_rival]
    if prod(dims) != own_count * first_count * second_count
        throw(DimensionMismatch("a tensor of size $dims has axes besides those of players $player, $first_rival and $second_rival"))
    end
    strides = accumulate(*, (1, Base.front(dims)...))
    own_stride, first_stride, second_stride = strides[player], strides[first_rival], strides[second_rival]
    first_marginal = @view work.sensitivities[first_rival][1:own_count, 1:first_count]
    second_marginal = @view work.sensitivities[second_rival][1:own_count, 1:second_count]
    first_weights, second_weights = work.strategies[first_rival], work.strategies[second_rival]
    checkbounds(tensor, 1:prod(dims))
    checkbounds(first_weights, 1:first_count)
    checkbounds(second_weights, 1:second_count)
    fill!(first_marginal, zero(eltype(first_marginal)))
    fill!(second_marginal, zero(eltype(second_marginal)))
    @inbounds for second_strategy in 1:second_count, first_strategy in 1:first_count
        base = 1 + (first_strategy - 1) * first_stride + (second_strategy - 1) * second_stride
        first_weight, second_weight = first_weights[first_strategy], second_weights[second_strategy]
        for own in 1:own_count
            entry = tensor[base + (own - 1) * own_stride]
            first_marginal[own, first_strategy] += second_weight * entry
            second_marginal[own, second_strategy] += first_weight * entry
        end
    end
    return nothing
end

"""
    write_marginals!(work, tensor, dims, positions, player, depth)

For each rival `q` at `positions` (`rival_axis`), the tensor `tensor` of size `dims` with every axis but `player`'s and
`q`'s contracted, into `work.sensitivities[q]`. The other rivals' axes are already contracted. Contract each half of
the rivals in turn and recurse into the other half, in `work.tensors[depth]`, down to two rivals
(`write_pair_marginals!`): a few passes over `tensor` in all.
"""
function write_marginals!(
        work::NamedTuple,
        tensor::AbstractArray{<:Real},
        dims::NTuple{N, Integer},
        positions::AbstractUnitRange{<:Integer},
        player::Integer,
        depth::Integer,
    )::Nothing where {N}
    if length(positions) <= 2
        first_axis, last_axis = rival_axis(first(positions), player), rival_axis(last(positions), player)
        if first_axis == last_axis
            write_marginal!(work.sensitivities[first_axis], tensor, dims, player, first_axis)
        else
            write_pair_marginals!(work, tensor, dims, player, first_axis, last_axis)
        end
        return nothing
    end
    lower, upper = halves(positions)
    target = work.tensors[depth]
    for (kept, contracted) in ((lower, upper), (upper, lower))
        reduced = contract_rivals!(target, work.spare, work.joint, tensor, dims, contracted, player, work.strategies)
        write_marginals!(work, target, reduced, kept, player, depth + 1)
    end
    return nothing
end

"""The lower and upper halves of `positions`, the upper one the larger when their count is odd."""
function halves(positions::AbstractUnitRange{<:Integer})::NTuple{2, AbstractUnitRange{<:Integer}}
    middle = first(positions) + length(positions) ÷ 2 - 1
    return first(positions):middle, (middle + 1):last(positions)
end

"""
The first rival's marginal of the tensor `tensor` of size `dims` (`write_marginals!`), into
`work.sensitivities[rival_axis(first(positions), player)]`, by the steps `write_marginals!` takes to it: contract the
upper half of `positions`, then recurse into the lower half. Its rounding is that of `write_marginals!`.
"""
function write_first_marginal!(
        work::NamedTuple,
        tensor::AbstractArray{<:Real},
        dims::NTuple{N, Integer},
        positions::AbstractUnitRange{<:Integer},
        player::Integer,
        depth::Integer,
    )::Nothing where {N}
    rival = rival_axis(first(positions), player)
    if length(positions) == 1
        write_marginal!(work.sensitivities[rival], tensor, dims, player, rival)
        return nothing
    end
    lower, upper = halves(positions)
    target = work.tensors[depth]
    reduced = contract_rivals!(target, work.spare, work.joint, tensor, dims, upper, player, work.strategies)
    write_first_marginal!(work, target, reduced, lower, player, depth + 1)
    return nothing
end

"""
Whether the work arrays `work` (`pass_workspace`) fit `payoff`: each player's strategy on its axis, room for
`player`'s expected payoffs and sensitivities, and tensors for `contract_payoff!`.
"""
function fits_workspace(work::NamedTuple, payoff::AbstractArray, player::Integer)::Bool
    (; strategies, sensitivities, expected, tensors, spare, joint) = work
    fitting = length(strategies) == length(sensitivities) == ndims(payoff) && length(expected) >= size(payoff, player)
    for (other, strategy) in pairs(strategies)
        fitting &= axes(strategy, 1) == axes(payoff, other) == axes(sensitivities[other], 2)
        fitting &= axes(sensitivities[other], 1) == axes(expected, 1)
    end
    entries = length(payoff)
    fitting &= length(tensors) >= tensor_count(ndims(payoff) - 1) && length(spare) >= entries && length(joint) >= entries
    return fitting && minimum(length, tensors) >= entries
end

"""
    contract_payoff!(work, payoff, player; slopes)

`player`'s expected payoffs `U_p(k, π)` into `work.expected`, at the strategies `π` in `work.strategies`: `payoff`
with each rival's axis contracted with its mixed strategy. With `slopes`, also its payoff sensitivities
`∂U_p(k) / ∂π_qj` to each rival `q` into `work.sensitivities[q]`: `payoff` with every axis but `p`'s and `q`'s
contracted (`write_marginals!`).
"""
function contract_payoff!(work::NamedTuple, payoff::AbstractArray{<:Real, N}, player::Integer; slopes::Bool)::AbstractVector where {N}
    if !fits_workspace(work, payoff, player)
        throw(DimensionMismatch("work arrays of sizes $(size.(work.strategies)) do not fit payoffs of size $(size(payoff))"))
    end
    expected, rivals = view(work.expected, axes(payoff, player)), 1:(N - 1)
    if N == 1
        expected .= payoff
    else
        if slopes
            write_marginals!(work, payoff, size(payoff), rivals, player, 1)
        else
            # The same steps to the first rival's marginal: the gains round alike with slopes and without.
            write_first_marginal!(work, payoff, size(payoff), rivals, player, 1)
        end
        rival = rival_axis(first(rivals), player)
        weights, marginal = work.strategies[rival], work.sensitivities[rival]
        checkbounds(marginal, eachindex(expected), eachindex(weights))
        first_weight = first(weights)
        # Down the marginal's columns: each entry adds the rival's strategies in order.
        @inbounds for own in eachindex(expected)
            expected[own] = first_weight * marginal[own, 1]
        end
        @inbounds for other in 2:length(weights)
            weight = weights[other]
            @simd for own in eachindex(expected)
                expected[own] += weight * marginal[own, other]
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
        write_player_gain!(view(gain, ranges[player]), contract_payoff!(work, payoff, player; slopes = false))
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
        expected = contract_payoff!(work, payoff, player; slopes = true)
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
∂π_qi` and expected payoffs in `work` (`contract_payoff!`). With `∂π_qi / ∂z_qj = π_qi (δ_ij - π_qj)` and
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
