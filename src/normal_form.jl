using LazyBroadcast: lazy_broadcast
using PathFollowing: into!!
using StaticArrays: SArray, SMatrix, SVector, Size, StaticVector

"""
    NormalFormGains(payoffs)

The `pass!!` of a normal-form game with any number of strategies per player. `payoffs[p][s]` is player `p`'s payoff at
the pure profile `s`; every array has one type and one size, each player's strategy count along its own axis. Player
`p`'s logits are `z_pk = log(π_pk / π_p1)` for its strategies `k ≥ 2`, against its first strategy, stacked player by
player (`logit_ranges`). Its gains are `G_pk = U_p(k, π) - U_p(1, π)`, its expected payoffs against the rivals' mixed
strategies, which do not depend on the precision. A pass writes into work arrays from `pass_workspace`; called
without them, it builds its own. Static payoffs (`SArray`) and logits (`SVector`) take static work arrays, so a pass
is isbits and runs in a GPU kernel; the arithmetic is the mutable pass's, step for step.
"""
struct NormalFormGains{N, A <: AbstractArray{<:Real, N}}
    payoffs::NTuple{N, A}

    function NormalFormGains(payoffs::NTuple{N, A})::NormalFormGains{N, A} where {N, A <: AbstractArray{<:Real, N}}
        if !allequal(size.(payoffs))
            throw(ArgumentError(lazy"every player's payoff array must have one size; got $(size.(payoffs))"))
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
`contract_payoff!!`.
"""
function pass_workspace(gains::NormalFormGains, ::AbstractVector{E})::NamedTuple where {E <: Real}
    counts = size(first(gains.payoffs))
    widest, entries = maximum(counts), prod(counts)
    tensors = zeros.(E, fill(entries, tensor_count(length(counts) - 1)))
    sensitivities = zeros.(E, widest, counts)
    spare, joint = zeros(E, entries), zeros(E, entries)
    return (; strategies = zeros.(E, counts), expected = zeros(E, widest), sensitivities, tensors, spare, joint)
end

"""
`pass_workspace` for static payoffs at static logits: the same arrays, static, so the pass is isbits. Every player
must have the same strategy count (`ArgumentError` otherwise): the pass indexes its players' arrays at run time, and a
tuple of static arrays of one type keeps that index type-stable.
"""
function pass_workspace(::NormalFormGains{N, A}, ::StaticVector{<:Any, E})::NamedTuple where {N, A <: SArray, E <: Real}
    counts = Tuple(Size(A))
    allequal(counts) || throw(ArgumentError(lazy"a static pass needs the same strategy count for every player; got $counts"))
    count, entries = first(counts), prod(counts)
    strategy, entry_vector = zeros(SVector{count, E}), zeros(SVector{entries, E})
    return (;
        strategies = ntuple(Returns(strategy), Val(N)),
        expected = strategy,
        sensitivities = ntuple(Returns(zeros(SMatrix{count, count, E})), Val(N)),
        tensors = ntuple(Returns(entry_vector), Val(tensor_count(N - 1))),
        spare = entry_vector,
        joint = entry_vector,
    )
end

"""
`strategy` set to the mixed strategy `softmax([0; logits[range]])` of a player whose logits in `range` are against its
first strategy (`into!!`). Use the returned strategy.
"""
@inline function reference_softmax!!(strategy::V, logits::AbstractVector{E}, range::UnitRange{Int})::V where {E <: Real, V <: AbstractVector{E}}
    shift = zero(E)
    for index in range
        shift = max(shift, logits[index])
    end
    strategy = set_entry!!(strategy, exp(-shift), firstindex(strategy))
    for (offset, index) in enumerate(range)
        strategy = set_entry!!(strategy, exp(logits[index] - shift), firstindex(strategy) + offset)
    end
    total = sum(strategy)
    return into!!(strategy, lazy_broadcast.(strategy ./ total))
end

"""Each player's strategy of `strategies` set to its mixed strategy at the stacked `logits` (`reference_softmax!!`)."""
@inline function softmax_strategies(strategies::S, logits::AbstractVector{<:Real}, ranges::Tuple)::S where {S <: Tuple}
    return reference_softmax!!.(strategies, (logits,), ranges)
end

"""Each player's mixed strategy at the stacked `logits` of `gains`, in fresh vectors."""
function mixed_strategies(gains::NormalFormGains, logits::AbstractVector{E})::Tuple where {E <: Real}
    return softmax_strategies(zeros.(E, size(first(gains.payoffs))), logits, logit_ranges(gains))
end

"""The axis of the rival at `position` among the axes of a payoff array but `player`'s."""
function rival_axis(position::Integer, player::Integer)::Integer
    return position < player ? position : position + 1
end

"""
The tensors `contract_payoff!!` holds at once for `rivals` rivals: one per halving of the rivals down to one
(`write_marginals!!`), and at least one. Foldable, so a static workspace sizes its tuple at compile time.
"""
Base.@assume_effects :foldable function tensor_count(rivals::Integer)::Integer
    return rivals <= 2 ? 1 : 1 + tensor_count(cld(rivals, 2))
end

"""
`buffer` with the weights of the adjacent axes `group` taken as one axis in its front, and their count: at each joint
index, the product of each axis' strategy weight, the first axis fastest. Use the returned buffer.
"""
@inline function group_weights!!(
        buffer::B,
        strategies::NTuple{N, AbstractVector{E}},
        group::AbstractUnitRange{<:Integer},
    )::Tuple{B, Int} where {N, E <: Real, B <: AbstractVector{E}}
    filled = 1
    buffer = set_entry!!(buffer, one(E), 1)
    for axis in group
        weights = strategies[axis]
        # The last slice first: each product reads the front before the first slice overwrites it.
        for slice in reverse(eachindex(weights)), entry in filled:-1:1
            buffer = set_entry!!(buffer, buffer[entry] * weights[slice], (slice - 1) * filled + entry)
        end
        filled *= length(weights)
    end
    return buffer, filled
end

"""
    contract_axes!!(target, source, dims, group, weights, weight_count)

Contract the adjacent axes `group` of the tensor `source` of size `dims`, read in linear order, into `target`: the
slices along the group, each times its joint weight among the first `weight_count` of `weights` (`group_weights!!`),
summed. Return `target` and the new size, one along `group`. A block past the group holds one `stride`-long run per
slice. The inner loop runs along a run when it is at least as long as the slices are many, else across the blocks;
each entry adds its slices in order either way.
"""
@inline function contract_axes!!(
        target::T,
        source::AbstractArray{<:Real},
        dims::NTuple{N, Integer},
        group::AbstractUnitRange{<:Integer},
        weights::AbstractVector{E},
        weight_count::Integer,
    )::Tuple{T, NTuple{N, Integer}} where {N, E <: Real, T <: AbstractVector{E}}
    numbers = ntuple(identity, Val(N))
    inside = first(group) .<= numbers .<= last(group)
    stride, width = prod(ifelse.(numbers .< first(group), dims, 1)), prod(ifelse.(inside, dims, 1))
    blocks = prod(ifelse.(numbers .> last(group), dims, 1))
    fitting = length(source) >= stride * width * blocks && length(target) >= stride * blocks
    fitting &= weight_count == width && length(weights) >= width
    if Base.has_offset_axes(target, source, weights) || !fitting
        throw(DimensionMismatch(lazy"cannot contract axes $group of size $dims from $(length(source)) into $(length(target))"))
    end
    if stride >= width
        first_weight = weights[1]
        @inbounds for block in 0:(blocks - 1)
            into, from = block * stride, block * stride * width
            @simd for inner in 1:stride
                target = set_entry!!(target, first_weight * source[from + inner], into + inner)
            end
            for slice in 2:width
                weight, offset = weights[slice], from + (slice - 1) * stride
                @simd for inner in 1:stride
                    target = set_entry!!(target, target[into + inner] + weight * source[offset + inner], into + inner)
                end
            end
        end
    else
        first_weight, step = weights[1], stride * width
        @inbounds for inner in 1:stride
            @simd for block in 0:(blocks - 1)
                target = set_entry!!(target, first_weight * source[block * step + inner], block * stride + inner)
            end
            for slice in 2:width
                weight, offset = weights[slice], (slice - 1) * stride + inner
                @simd for block in 0:(blocks - 1)
                    into = block * stride + inner
                    target = set_entry!!(target, target[into] + weight * source[block * step + offset], into)
                end
            end
        end
    end
    return target, ifelse.(inside, 1, dims)
end

"""
    contract_rivals!!(target, spare, joint, source, dims, positions, player, strategies)

Contract the axes of the rivals at `positions` (`rival_axis`) of the tensor `source` of size `dims` with their mixed
`strategies` into `target`. The axes form one group (`contract_axes!!`), or two around `player`'s own: the upper group
goes first, into `spare`. `joint` holds each group's weights (`group_weights!!`). Return the three buffers and the new
size; use the returned buffers.
"""
@inline function contract_rivals!!(
        target::T,
        spare::S,
        joint::J,
        source::AbstractArray{<:Real},
        dims::NTuple{N, Integer},
        positions::AbstractUnitRange{<:Integer},
        player::Integer,
        strategies::NTuple{N, AbstractVector{E}},
    )::Tuple{T, S, J, NTuple{N, Integer}} where {N, E <: Real, T <: AbstractVector{E}, S <: AbstractVector{E}, J <: AbstractVector{E}}
    lowest, highest = rival_axis(first(positions), player), rival_axis(last(positions), player)
    if lowest < player < highest
        upper, lower = (player + 1):highest, lowest:(player - 1)
        joint, filled = group_weights!!(joint, strategies, upper)
        spare, reduced = contract_axes!!(spare, source, dims, upper, joint, filled)
        joint, filled = group_weights!!(joint, strategies, lower)
        target, reduced = contract_axes!!(target, spare, reduced, lower, joint, filled)
        return target, spare, joint, reduced
    end
    joint, filled = group_weights!!(joint, strategies, lowest:highest)
    target, reduced = contract_axes!!(target, source, dims, lowest:highest, joint, filled)
    return target, spare, joint, reduced
end

"""
`marginal` set to the tensor `tensor` of size `dims`, all of whose axes but `player`'s and `rival`'s have size one,
as a matrix: `player`'s strategies down, `rival`'s across. Use the returned marginal.
"""
function write_marginal!!(
        marginal::M,
        tensor::AbstractArray{<:Real},
        dims::NTuple{N, Integer},
        player::Integer,
        rival::Integer,
    )::M where {N, M <: AbstractMatrix}
    own_count, rival_count = dims[player], dims[rival]
    if prod(dims) != own_count * rival_count
        throw(DimensionMismatch(lazy"a tensor of size $dims has axes besides those of players $player and $rival"))
    end
    checkbounds(marginal, 1:own_count, 1:rival_count)
    checkbounds(tensor, 1:(own_count * rival_count))
    own_stride, rival_stride = player < rival ? (1, own_count) : (rival_count, 1)
    @inbounds for column in 1:rival_count, row in 1:own_count
        marginal = set_entry!!(marginal, tensor[1 + (row - 1) * own_stride + (column - 1) * rival_stride], row, column)
    end
    return marginal
end

"""
`work` with the marginals of the tensor `tensor` of size `dims` for rivals `first_rival` and `second_rival` in its
sensitivities (`write_marginal!!`): all of its axes but `player`'s and theirs have size one. One pass: each entry, times
one rival's weight, adds to the other rival's marginal, the other rival's strategies in order. Use the returned work.
"""
function write_pair_marginals!!(
        work::W,
        tensor::AbstractArray{<:Real},
        dims::NTuple{N, Integer},
        player::Integer,
        first_rival::Integer,
        second_rival::Integer,
    )::W where {N, W <: NamedTuple}
    own_count, first_count, second_count = dims[player], dims[first_rival], dims[second_rival]
    if prod(dims) != own_count * first_count * second_count
        throw(DimensionMismatch(lazy"a tensor of size $dims has axes besides those of players $player, $first_rival and $second_rival"))
    end
    strides = accumulate(*, (1, Base.front(dims)...))
    own_stride, first_stride, second_stride = strides[player], strides[first_rival], strides[second_rival]
    first_marginal, second_marginal = work.sensitivities[first_rival], work.sensitivities[second_rival]
    first_weights, second_weights = work.strategies[first_rival], work.strategies[second_rival]
    checkbounds(tensor, 1:prod(dims))
    checkbounds(first_weights, 1:first_count)
    checkbounds(second_weights, 1:second_count)
    checkbounds(first_marginal, 1:own_count, 1:first_count)
    checkbounds(second_marginal, 1:own_count, 1:second_count)
    blank = zero(eltype(first_marginal))
    @inbounds for column in 1:first_count, own in 1:own_count
        first_marginal = set_entry!!(first_marginal, blank, own, column)
    end
    @inbounds for column in 1:second_count, own in 1:own_count
        second_marginal = set_entry!!(second_marginal, blank, own, column)
    end
    @inbounds for second_strategy in 1:second_count, first_strategy in 1:first_count
        base = 1 + (first_strategy - 1) * first_stride + (second_strategy - 1) * second_stride
        first_weight, second_weight = first_weights[first_strategy], second_weights[second_strategy]
        for own in 1:own_count
            entry = tensor[base + (own - 1) * own_stride]
            first_marginal = set_entry!!(first_marginal, first_marginal[own, first_strategy] + second_weight * entry, own, first_strategy)
            second_marginal = set_entry!!(second_marginal, second_marginal[own, second_strategy] + first_weight * entry, own, second_strategy)
        end
    end
    sensitivities = with_item!!(with_item!!(work.sensitivities, first_rival, first_marginal), second_rival, second_marginal)
    return merge(work, (; sensitivities))
end

"""
    write_marginals!!(work, tensor, dims, positions, player, depth, every_rival)

`work` with, for each rival `q` at `positions` (`rival_axis`), the tensor `tensor` of size `dims` with every axis but
`player`'s and `q`'s contracted, in its sensitivities `q`. The other rivals' axes are already contracted. Contract each
half of the rivals in turn and recurse into the other half, in its tensors `depth`, down to two rivals
(`write_pair_marginals!!`): a few passes over `tensor` in all. Without `every_rival`, write only the first rival's
marginal, by the first branch alone: it rounds as with `every_rival`. Use the returned work.
"""
function write_marginals!!(
        work::W,
        tensor::AbstractArray{<:Real},
        dims::NTuple{N, Integer},
        positions::AbstractUnitRange{<:Integer},
        player::Integer,
        depth::Integer,
        every_rival::Bool,
    )::W where {N, W <: NamedTuple}
    first_axis, last_axis = rival_axis(first(positions), player), rival_axis(last(positions), player)
    if first_axis == last_axis
        marginal = write_marginal!!(work.sensitivities[first_axis], tensor, dims, player, first_axis)
        return merge(work, (; sensitivities = with_item!!(work.sensitivities, first_axis, marginal)))
    end
    if every_rival && length(positions) == 2
        return write_pair_marginals!!(work, tensor, dims, player, first_axis, last_axis)
    end
    lower, upper = halves(positions)
    work, target, reduced = contracted_rivals!!(work, tensor, dims, upper, player, depth)
    work = write_marginals!!(work, target, reduced, lower, player, depth + 1, every_rival)
    if every_rival
        work, target, reduced = contracted_rivals!!(work, tensor, dims, lower, player, depth)
        work = write_marginals!!(work, target, reduced, upper, player, depth + 1, every_rival)
    end
    return work
end

"""
`contract_rivals!!` of the rivals at `positions` of `tensor` into `work`'s tensors `depth`: the work, that tensor and
its size. Use the returned work.
"""
@inline function contracted_rivals!!(
        work::W,
        tensor::AbstractArray{<:Real},
        dims::NTuple{N, Integer},
        positions::AbstractUnitRange{<:Integer},
        player::Integer,
        depth::Integer,
    )::Tuple{W, AbstractVector, NTuple{N, Integer}} where {N, W <: NamedTuple}
    target, spare, joint, reduced = contract_rivals!!(work.tensors[depth], work.spare, work.joint, tensor, dims, positions, player, work.strategies)
    return merge(work, (; tensors = with_item!!(work.tensors, depth, target), spare, joint)), target, reduced
end

"""The lower and upper halves of `positions`, the upper one the larger when their count is odd."""
function halves(positions::AbstractUnitRange{<:Integer})::NTuple{2, AbstractUnitRange{<:Integer}}
    middle = first(positions) + length(positions) ÷ 2 - 1
    return first(positions):middle, (middle + 1):last(positions)
end

"""
Whether the work arrays `work` (`pass_workspace`) fit `payoff`: each player's strategy on its axis, room for
`player`'s expected payoffs and sensitivities, and tensors for `contract_payoff!!`.
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
    contract_payoff!!(work, payoff, player; slopes)

`work` with `player`'s expected payoffs `U_p(k, π)` in the front of its `expected`, at the strategies `π` in its
`strategies`: `payoff` with each rival's axis contracted with its mixed strategy. With `slopes`, also its payoff
sensitivities `∂U_p(k) / ∂π_qj` to each rival `q` in its `sensitivities[q]`: `payoff` with every axis but `p`'s and
`q`'s contracted (`write_marginals!!`). Use the returned work.
"""
function contract_payoff!!(work::W, payoff::AbstractArray{<:Real, N}, player::Integer; slopes::Bool)::W where {W <: NamedTuple, N}
    if !fits_workspace(work, payoff, player)
        throw(DimensionMismatch(lazy"work arrays of sizes $(size.(work.strategies)) do not fit payoffs of size $(size(payoff))"))
    end
    own = axes(payoff, player)
    if N == 1
        expected = work.expected
        for index in own
            expected = set_entry!!(expected, payoff[index], index)
        end
        return merge(work, (; expected))
    end
    # Without slopes, only the first rival's marginal, by the same steps: the gains round alike either way.
    work = write_marginals!!(work, payoff, size(payoff), 1:(N - 1), player, 1, slopes)
    rival = rival_axis(1, player)
    weights, marginal, expected = work.strategies[rival], work.sensitivities[rival], work.expected
    checkbounds(marginal, own, eachindex(weights))
    checkbounds(expected, own)
    first_weight = first(weights)
    # Down the marginal's columns: each entry adds the rival's strategies in order.
    @inbounds for index in own
        expected = set_entry!!(expected, first_weight * marginal[index, 1], index)
    end
    @inbounds for other in 2:length(weights)
        weight = weights[other]
        @simd for index in own
            expected = set_entry!!(expected, expected[index] + weight * marginal[index, other], index)
        end
    end
    return merge(work, (; expected))
end

"""
    write_gains!!(work, gain, gains, logits)

Each player's gains `G_pk = U_p(k, π) - U_p(1, π)` of `gains` at the stacked `logits`, in `gain`. The mixed
strategies `π` stay in the work's `strategies` (`pass_workspace`). Return the work and `gain`; use the returned ones.
"""
function write_gains!!(work::W, gain::G, gains::NormalFormGains, logits::AbstractVector{E})::Tuple{W, G} where {E <: Real, W <: NamedTuple, G <: AbstractVector}
    ranges = logit_ranges(gains)
    work = merge(work, (; strategies = softmax_strategies(work.strategies, logits, ranges)))
    for (player, payoff) in pairs(gains.payoffs)
        work = contract_payoff!!(work, payoff, player; slopes = false)
        gain = write_player_gain!!(gain, ranges[player], work.expected, size(payoff, player))
    end
    return work, gain
end

"""
`gain` with a player's gains in its `range`: the player's `count` expected payoffs in `expected`, each after the
first's less the first's. Use the returned gain.
"""
@inline function write_player_gain!!(gain::G, range::UnitRange{Int}, expected::AbstractVector, count::Integer)::G where {G <: AbstractVector}
    if length(range) != count - 1
        throw(DimensionMismatch(lazy"a range of $(length(range)) gains does not fit $count expected payoffs"))
    end
    reference = expected[begin]
    for (offset, index) in enumerate(range)
        gain = set_entry!!(gain, expected[begin + offset] - reference, index)
    end
    return gain
end

function (gains::NormalFormGains)(
        work::W,
        gain::G,
        gain_slope::S,
        gain_precision_slope::P,
        logits::AbstractVector{<:Real},
        ::Real,
    )::Tuple{W, G, S, P} where {W <: NamedTuple, G <: AbstractVector, S <: AbstractMatrix, P <: AbstractVector}
    ranges = logit_ranges(gains)
    work = merge(work, (; strategies = softmax_strategies(work.strategies, logits, ranges)))
    gain_slope, gain_precision_slope = zero!!(gain_slope), zero!!(gain_precision_slope)
    for (player, payoff) in pairs(gains.payoffs)
        work = contract_payoff!!(work, payoff, player; slopes = true)
        gain = write_player_gain!!(gain, ranges[player], work.expected, size(payoff, player))
        for (rival, rival_range) in pairs(ranges)
            if rival != player
                gain_slope = write_gain_slope!!(gain_slope, ranges[player], rival_range, payoff, work, player, rival)
            end
        end
    end
    return work, gain, gain_slope, gain_precision_slope
end

function (gains::NormalFormGains)(
        gain::G,
        gain_slope::S,
        gain_precision_slope::P,
        logits::AbstractVector,
        precision::Real,
    )::Tuple{G, S, P} where {G <: AbstractVector, S <: AbstractMatrix, P <: AbstractVector}
    _, gain, gain_slope, gain_precision_slope = gains(pass_workspace(gains, logits), gain, gain_slope, gain_precision_slope, logits, precision)
    return gain, gain_slope, gain_precision_slope
end

"""
    write_gain_slope!!(gain_slope, rows, columns, payoff, work, player, rival)

`gain_slope` with `∂G_pk / ∂z_qj` of `player` `p` in the logits of `rival` `q` in its block `rows` by `columns`, from
the sensitivities `D_ki = ∂U_p(k) / ∂π_qi` and expected payoffs in `work` (`contract_payoff!!`). With
`∂π_qi / ∂z_qj = π_qi (δ_ij - π_qj)` and `Σ_i D_ki π_qi = U_p(k)`, `∂U_p(k) / ∂z_qj = π_qj (D_kj - U_p(k))`; a gain
takes its strategy's row less the first's. Use the returned gain slope.
"""
function write_gain_slope!!(
        gain_slope::S,
        rows::UnitRange{Int},
        columns::UnitRange{Int},
        payoff::AbstractArray,
        work::NamedTuple,
        player::Integer,
        rival::Integer,
    )::S where {S <: AbstractMatrix}
    own, other = axes(payoff, player), axes(payoff, rival)
    if (length(rows), length(columns)) != (length(own) - 1, length(other) - 1)
        throw(DimensionMismatch(lazy"a block of size $((length(rows), length(columns))) does not fit $(length(own)) by $(length(other)) strategies"))
    end
    # Loops: a broadcast over small views spends more on alias checks and axes than on its sums. One bounds check per
    # array lets the inner loop vectorize.
    rival_strategy, sensitivity, expected = work.strategies[rival], work.sensitivities[rival], work.expected
    checkbounds(rival_strategy, other)
    checkbounds(sensitivity, own, other)
    checkbounds(expected, own)
    checkbounds(gain_slope, rows, columns)
    # Row and column `i` of the block hold strategies `i + shift` of `player` and `rival`: the first strategy has none.
    row_shift, column_shift = first(own) + 1 - first(rows), first(other) + 1 - first(columns)
    first_expected = expected[first(own)]
    @inbounds for column in columns
        later = column + column_shift
        weight, first_sensitivity = rival_strategy[later], sensitivity[first(own), later]
        # The broadcast this loop replaced summed in this order. The Float32 noise bound in `opening_tests.jl` is tight.
        @simd for row in rows
            slope = weight * (((sensitivity[row + row_shift, later] - expected[row + row_shift]) - first_sensitivity) + first_expected)
            gain_slope = set_entry!!(gain_slope, slope, row, column)
        end
    end
    return gain_slope
end
