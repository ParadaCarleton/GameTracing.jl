using ErrorTypes: is_error, unwrap, unwrap_error
using LogitNash: LogitNash

"""LogitNash's mixed strategies of `payoffs` past `precision`, and the precision it stopped at, which may pass it."""
function logitnash_profile(payoffs::Tuple, precision::Real)::NamedTuple
    profile, status = LogitNash.solve(map(float, payoffs); stop_lambda = float(precision), stop_eps = zero(float(precision)))
    return (; strategies = map(strategy -> Float32.(strategy), profile), precision = Float32(status.lambda))
end

"""The largest difference in any probability between two profiles of mixed strategies."""
function profile_gap(left::Tuple, right::Tuple)::eltype(first(left))
    return maximum(maximum.(abs, left .- right))
end

"""Each player's log odds against its first strategy, stacked: the logits of a profile of mixed strategies."""
function stacked_log_odds(strategies::Tuple)::AbstractVector
    return reduce(vcat, map(strategy -> log.(view(strategy, 2:lastindex(strategy)) ./ first(strategy)), strategies))
end

"""
    strategies_cross_checked!(payoffs, precision)

Trace the normal-form game `payoffs` to LogitNash's stopping precision past `precision` and test the endpoint against
`branch_reference`, with LogitNash as a cross-check (`cross_checked!`). A branch that folds back in precision has no
reference; every endpoint must also solve the map as the corrector settles it (`settled_logits`).
"""
function strategies_cross_checked!(payoffs::Tuple, precision::Real)::Nothing
    cross = logitnash_profile(payoffs, precision)
    gains = NormalFormGains(map(table -> Float32.(table), payoffs))
    settings = BifurcationKitSettings{Float32}()
    result = trace_endpoint(LogitGame(gains, cross.precision), logit_count(gains), settings)
    @test !is_error(result)
    if is_error(result)
        @info "trace failed" payoffs unwrap_error(result).cause
        return nothing
    end
    traced = mixed_strategies(gains, unwrap(result).logits)
    reference_gains = NormalFormGains(map(float, payoffs))
    reference = branch_reference(reference_gains, logit_count(reference_gains), cross.precision)
    @test isempty(reference.crossings)
    if reference.reached
        @test profile_gap(traced, mixed_strategies(reference_gains, reference.logits)) < 1.0f-3
    end
    @test settled_logits(gains, unwrap(result).logits, cross.precision, settings)
    if profile_gap(traced, cross.strategies) >= 1.0f-3 && reference.reached
        @test profile_gap(mixed_strategies(reference_gains, reference.logits), cross.strategies) >= 1.0f-3
    end
    return nothing
end

@testset "NormalFormGains with two strategies each is NormalFormPass with its logits negated" begin
    rng = fresh_rng()
    for payoffs in values(LOGITNASH_GAMES)
        floats = map(table -> float.(table), payoffs)
        logits = randn(rng, length(payoffs))
        level = float(3 // 2)
        against_first = evaluated(NormalFormGains(floats), -logits, level)
        against_second = evaluated(NormalFormPass(floats), logits, level)
        @test against_first.response ≈ -against_second.response
        @test against_first.logit_slope ≈ against_second.logit_slope
    end
    # NormalFormPass reads one logit per player, so a third action would be silently dropped.
    @test_throws ArgumentError NormalFormPass((zeros(Float32, 3, 3), zeros(Float32, 3, 3)))
end

"""Player `player`'s expected payoff of each of its strategies, summed over every pure profile."""
function summed_expected_payoffs(payoff::AbstractArray, strategies::Tuple, player::Integer)::AbstractVector
    rivals = filter(!=(player), eachindex(strategies))
    return [
        sum(
            payoff[profile] * prod(strategies[rival][profile[rival]] for rival in rivals)
                for profile in CartesianIndices(payoff) if profile[player] == strategy
        )
            for strategy in axes(payoff, player)
    ]
end

"""The bytes one call of `pass` on its `pass_workspace` allocates at `logits` and `precision`, after a first call compiles it."""
function pass_bytes(pass::Any, logits::AbstractVector{E}, precision::Real)::Integer where {E <: Real}
    width = length(logits)
    gain, gain_slope, gain_precision_slope = zeros(E, width), zeros(E, width, width), zeros(E, width)
    work = pass_workspace(pass, logits)
    pass(work, gain, gain_slope, gain_precision_slope, logits, precision)
    return @allocated pass(work, gain, gain_slope, gain_precision_slope, logits, precision)
end

@testset "NormalFormGains: each player's gains, stacked in player order, are its expected payoffs less its first strategy's" begin
    rng = fresh_rng()
    for counts in ((3, 3, 3), (2, 3, 4), (4, 2, 3), (3, 2), (2, 3, 2, 2), (2, 2, 2, 2, 2, 2))
        # 32-bit payoffs under 64-bit logits: the work arrays take the logits' element type.
        payoffs = ntuple(_ -> 3 .* randn(rng, Float32, counts), length(counts))
        gains = NormalFormGains(payoffs)
        logits = randn(rng, logit_count(gains))
        strategies = mixed_strategies(gains, logits)
        expected = summed_expected_payoffs.(payoffs, (strategies,), eachindex(payoffs))
        stacked = reduce(vcat, [view(payoff, 2:lastindex(payoff)) .- first(payoff) for payoff in expected])
        @test evaluated(gains, logits, 1).response ≈ stacked
    end
end

@testset "a NormalFormGains pass on its workspace allocates nothing" begin
    rng = fresh_rng()
    for counts in ((3, 2, 4), (2, 2, 3, 2, 2))
        gains = NormalFormGains(ntuple(_ -> randn(rng, Float32, counts), length(counts)))
        @test pass_bytes(gains, randn(rng, Float32, logit_count(gains)), 2.0f0) == 0
    end
end

@testset "a one-player game's gains are its payoffs less the first, with no slope" begin
    rng = fresh_rng()
    payoff = randn(rng, 4)
    gains = NormalFormGains((payoff,))
    pass = evaluated(gains, randn(rng, logit_count(gains)), 1)
    @test pass.response == payoff[2:end] .- first(payoff)
    @test iszero(pass.logit_slope)
end

@testset "a payoff span past the type throws where it is read, even after the payoffs change ($element)" for element in (Float32, widen(Float32))
    large = floatmax(element) * (3 // 4)
    # Whether the extremes belong to one player or to two, the range overflows.
    across = NormalFormGains((element[large 0; 0 0], element[-large 0; 0 0]))
    @test_throws ArgumentError GameTracing.payoff_range(across)
    @test_throws ArgumentError GameTracing.gap_bound(across)
    @test_throws ArgumentError GameTracing.gap_bound(NormalFormGains((element[large -large; 0 0], element[0 0; 0 0])))
    # Payoffs mutated after construction are read afresh.
    payoffs = (element[1 0; 0 0], element[0 0; 0 1])
    mutated = NormalFormGains(payoffs)
    payoffs[1][1, 1], payoffs[1][2, 2] = large, -large
    @test_throws ArgumentError GameTracing.gap_bound(mutated)
    # A non-finite payoff passes through: its trace fails NotFinite as a value.
    @test isnan(GameTracing.payoff_range(NormalFormGains((element[NaN32 0; 0 0], element[-large 0; 0 0]))))
end

@testset "group_weights! takes adjacent axes as one, the first axis fastest ($element)" for element in (Float32, widen(Float32))
    strategies = (element.([1, 2]), element.([3, 5, 7]), element.([11, 13]))
    buffer = zeros(element, 12)
    @test GameTracing.group_weights!(buffer, strategies, 1:2) == element.([3, 6, 5, 10, 7, 14])
    @test GameTracing.group_weights!(buffer, strategies, 2:3) == element.([33, 55, 77, 39, 65, 91])
    @test GameTracing.group_weights!(buffer, strategies, 2:2) == element.([3, 5, 7])
end

"""Each axis of `source` in `group` weighted by its entry of `strategies`, multiplied out: an array of `source`'s rank."""
function joint_weights(source::AbstractArray, strategies::Tuple, group::AbstractUnitRange)::AbstractArray
    shaped = [reshape(strategies[axis], ntuple(other -> other == axis ? size(source, axis) : 1, ndims(source))) for axis in group]
    return reduce((left, right) -> left .* right, shaped)
end

@testset "contract_axes! sums the slices along adjacent axes, each times its joint weight ($element)" for element in (Float32, widen(Float32))
    # Runs of 1 to 54 entries against 2 to 54 slices take both loop orders. Every sum is an exact integer.
    for source in (element.(reshape(1:24, 2, 3, 4)), element.(reshape(1:216, 18, 3, 4))), group in (1:1, 1:2, 2:3, 3:3)
        strategies = ntuple(axis -> element.(collect(axes(source, axis))), ndims(source))
        reference = sum(source .* joint_weights(source, strategies, group); dims = Tuple(group))
        weights = GameTracing.group_weights!(zeros(element, length(source)), strategies, group)
        target = zeros(element, length(reference))
        @test GameTracing.contract_axes!(target, source, size(source), group, weights) == size(reference)
        @test target == vec(reference)
    end
    source = element.(reshape(1:24, 2, 3, 4))
    @test_throws DimensionMismatch GameTracing.contract_axes!(zeros(element, 6), source, size(source), 3:3, element.(1:3))
end

@testset "write_marginal! and write_pair_marginals! refuse a tensor with an axis besides their players'" begin
    @test_throws DimensionMismatch GameTracing.write_marginal!(zeros(2, 2), zeros(8), (2, 2, 2), 1, 2)
    gains = NormalFormGains(ntuple(_ -> zeros(2, 2, 2, 2), 4))
    work = pass_workspace(gains, zeros(logit_count(gains)))
    @test_throws DimensionMismatch GameTracing.write_pair_marginals!(work, zeros(16), (2, 2, 2, 2), 1, 2, 3)
end

@testset "a NormalFormGains contraction refuses work arrays of another game" begin
    rng = fresh_rng()
    gains = NormalFormGains(ntuple(_ -> randn(rng, 3, 2), 2))
    logits = randn(rng, logit_count(gains))
    for other in (NormalFormGains(ntuple(_ -> randn(rng, 2, 3), 2)), NormalFormGains(ntuple(_ -> randn(rng, 3, 2, 2), 3)))
        work = pass_workspace(other, logits)
        @test_throws DimensionMismatch GameTracing.contract_payoff!(work, first(gains.payoffs), 1; slopes = true)
    end
end

"""Fill every float array in pass work `work` with NaN, so a read before a write shows in the pass's outputs."""
function poison!(work::AbstractArray{<:AbstractFloat})::Nothing
    fill!(work, NaN)
    return nothing
end

poison!(work::Union{Tuple, NamedTuple, AbstractArray})::Nothing = foreach(poison!, work)

@testset "a NormalFormGains pass reads no work array before it writes it, so a trace can share them ($element)" for element in (Float32, Float64)
    rng = fresh_rng()
    for counts in ((2, 3), (2, 2, 2), (3, 2, 2, 2))
        gains = NormalFormGains(ntuple(_ -> randn(rng, element, counts...), length(counts)))
        width = logit_count(gains)
        logits = randn(rng, element, width)
        outputs()::Tuple = (zeros(element, width), zeros(element, width, width), zeros(element, width))
        clean, poisoned = outputs(), outputs()
        gains(pass_workspace(gains, logits), clean..., logits, element(2))
        work = pass_workspace(gains, logits)
        poison!(work)
        gains(work, poisoned..., logits, element(2))
        @test all(output -> all(isfinite, output), poisoned)
        @test all(splat(isapprox), zip(clean, poisoned))
    end
end

@testset "write_gain_slope! refuses a block that does not fit the two players' strategies" begin
    rng = fresh_rng()
    gains = NormalFormGains(ntuple(_ -> randn(rng, 3, 2), 2))
    logits = randn(rng, logit_count(gains))
    work = pass_workspace(gains, logits)
    width = logit_count(gains)
    # The pass leaves the strategies at `logits`; the contraction then refills player 1's sensitivities.
    gains(work, zeros(width), zeros(width, width), zeros(width), logits, 1)
    payoff = first(gains.payoffs)
    GameTracing.contract_payoff!(work, payoff, 1; slopes = true)
    @test_throws DimensionMismatch GameTracing.write_gain_slope!(zeros(2, 2), payoff, work, 1, 2)
    fitting = zeros(2, 1)
    GameTracing.write_gain_slope!(fitting, payoff, work, 1, 2)
    @test fitting ≈ evaluated(gains, logits, 1).logit_slope[1:2, 3:3]
end

@testset "NormalFormGains: the gain slope is the derivative of the gains, at any strategy counts" begin
    rng = fresh_rng()
    # Four rivals and more take the contraction two halvings deep.
    for counts in ((3, 3), (2, 3, 4), (4, 2), (1, 3), (3, 2, 2), (2, 2, 3, 2, 2), (2, 2, 2, 2, 2, 2))
        payoffs = ntuple(_ -> 3 .* randn(rng, counts), length(counts))
        gains = NormalFormGains(payoffs)
        logits = randn(rng, logit_count(gains))
        step = inv(float(10^6))
        level = float(3 // 2)
        slope = evaluated(gains, logits, level).logit_slope
        for (rival, _) in pairs(logits)
            nudge = step .* (eachindex(logits) .== rival)
            upper, lower = evaluated(gains, logits .+ nudge, level), evaluated(gains, logits .- nudge, level)
            @test maximum(abs, (upper.response .- lower.response) ./ (2 * step) .- slope[:, rival]) < inv(float(10^6))
        end
        @test evaluated(gains, logits, level).precision_slope ≈ evaluated(gains, logits, 1).response
    end
end

@testset "mixed_strategies are probability vectors whose log odds against the first strategy are the logits" begin
    rng = fresh_rng()
    gains = NormalFormGains(ntuple(_ -> randn(rng, 3, 4, 2), 3))
    @test logit_count(gains) == 2 + 3 + 1
    logits = 5 .* randn(rng, logit_count(gains))
    strategies = mixed_strategies(gains, logits)
    @test length.(strategies) == (3, 4, 2)
    @test all(strategy -> sum(strategy) ≈ 1 && all(>=(0), strategy), strategies)
    @test stacked_log_odds(strategies) ≈ logits
    # A player with one strategy has no logits.
    @test logit_count(NormalFormGains((randn(rng, 1, 3), randn(rng, 1, 3)))) == 2
end

@testset "NormalFormGains refuses payoff arrays of different sizes" begin
    @test_throws ArgumentError NormalFormGains((zeros(2, 3), zeros(3, 2)))
    @test_throws ArgumentError NormalFormGains((zeros(2, 3), zeros(Int, 2, 3)))
    @test_throws ArgumentError NormalFormGains((zeros(2, 3), zeros(2, 3), zeros(2, 3)))
end

@testset "rock-paper-scissors traces to the uniform mixture at every precision" begin
    beats = [0 -1 1; 1 0 -1; -1 1 0]
    game = LogitGame(NormalFormGains((Float32.(beats), Float32.(-beats))), 5.0f0)
    traced = unwrap(trace_endpoint(game, logit_count(game.pass!), BifurcationKitSettings{Float32}()))
    @test profile_gap(mixed_strategies(game.pass!, traced.logits), (fill(1.0f0 / 3, 3), fill(1.0f0 / 3, 3))) < 1.0f-4
end

@testset "games with more than two strategies: the trace follows the principal branch" begin
    rng = fresh_rng()
    for counts in ((3, 3), (2, 3), (3, 2, 2), (4, 3)), _ in 1:4
        strategies_cross_checked!(ntuple(_ -> 3 .* randn(rng, counts), length(counts)), 3)
    end
end
