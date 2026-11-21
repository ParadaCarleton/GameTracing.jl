using ErrorTypes: is_error
using GameTracing: FisherMetric, checked_ranges, largest_kl, player_kl
using LinearAlgebra: dot
using Mooncake: Mooncake
using PathFollowing: PathFollowing
using StaticArrays: SVector

"""`log(1 + Σ exp z)`: the log normalizer of a player's mixed strategy at its logits `z` against its reference."""
function log_partition(logits::AbstractVector{E})::E where {E <: Real}
    return log(one(E) + sum(exp, logits))
end

"""`KL(π_from ‖ π_to)` of the mixed strategies `softmax([0; z])` at the logits `from` and `to`, summed directly."""
function direct_kl(from::AbstractVector{E}, to::AbstractVector{E})::E where {E <: Real}
    probabilities, others = exp.([zero(E); from]), exp.([zero(E); to])
    probabilities, others = probabilities ./ sum(probabilities), others ./ sum(others)
    return sum(probabilities .* log.(probabilities ./ others))
end

"""
PALC's weight `ω(λ)` of `Δλ²` as `FisherMetric` documents it, at `precision` toward `target` under PALC state weight
`palc_state_weight`: `c(λ)` while `c(λ) ≥ c(P)`, and `c(λ) + (1 - θ) (1 - c(λ) / c(P))` past it.
"""
function documented_precision_weight(precision_term::PrecisionTerm, precision::E, target::E, palc_state_weight::E)::E where {E <: Real}
    current, smallest = precision_term == LogPrecisionTerm ? (inv(1 + abs(precision))^2, inv(1 + target)^2) : (inv(target)^2, inv(target)^2)
    return current >= smallest ? current : current + (1 - palc_state_weight) * (1 - current / smallest)
end

"""
`player_apply!`'s bound on the rounding of player `player`'s block sum of `applied = M vector`, in `applied`'s element
type, from the exact frozen metric `exact`: `eps ((n + L) Σᵢ |outᵢ| + |gap_scale| π_ref Σᵢ qᵢ (|λ Δwᵢ| + |wᵢ Δλ|))`.
"""
function block_sum_bound(
        exact::GameTracing.FrozenFisher{<:FisherMetric, BigFloat},
        applied::AbstractVector{E},
        vector::AbstractVector{E},
        player::Integer,
        range::UnitRange{Int},
    )::BigFloat where {E <: Real}
    (; metric, point, probabilities, references) = exact
    largest_log = maximum(probability -> abs(log(probability)), [references[player]; probabilities[range]])
    gap_scale = metric.state_weight * last(point) / metric.palc_state_weight
    moves = sum(index -> probabilities[index] * (abs(last(point) * vector[index]) + abs(point[index] * last(vector))), range)
    return eps(E) * ((length(range) + 1 + largest_log) * sum(entry -> abs(big(entry)), applied[range]) + abs(gap_scale) * references[player] * moves)
end

"""A one-logit pass with zero gain and no `logit_ranges`."""
struct RangelessPass end

# A zero bound: the opening falls back to one (`opening_step`).
GameTracing.gap_bound(::RangelessPass)::Real = 0

function (::RangelessPass)(
        gain::G,
        gain_slope::S,
        gain_precision_slope::P,
        ::AbstractVector,
        ::Real,
    )::Tuple{G, S, P} where {G <: AbstractVector, S <: AbstractMatrix, P <: AbstractVector}
    fill!(gain, zero(eltype(gain)))
    fill!(gain_slope, zero(eltype(gain_slope)))
    fill!(gain_precision_slope, zero(eltype(gain_precision_slope)))
    return gain, gain_slope, gain_precision_slope
end

@testset "a one-logit Fisher block is σ(z)σ(-z) ($element)" for element in (Float32, widen(Float32))
    metric = FisherMetric((1:1,), one(element), TargetPrecisionTerm, one(element), one(element))
    for logit in element[-60, -8, -3 // 10, 0, 17 // 10, 9, 60]
        frozen = PathFollowing.frozen_metric(metric, element[logit, 1])
        exact = inv(1 + exp(-big(logit))) * inv(1 + exp(big(logit)))
        @test PathFollowing.metric_inner(frozen, element[1, 0], element[1, 0]) ≈ exact rtol = 8 * eps(element)
    end
end

@testset "a categorical Fisher block is the Hessian of the log normalizer, by Mooncake ($element)" for element in (Float32, widen(Float32))
    rng = fresh_rng()
    # Players of 3, 2 and 4 strategies, each with one logit fewer.
    ranges = (1:2, 3:3, 4:6)
    for _ in 1:10
        logits = 3 .* randn(rng, element, 6)
        # At precision one, `Δz = Δw`: the metric's gap block is the logits' Fisher information.
        frozen = PathFollowing.frozen_metric(FisherMetric(ranges, one(element), TargetPrecisionTerm, one(element), one(element)), [logits; one(element)])
        hessian = zeros(element, 6, 6)
        for range in ranges
            cache = Mooncake.prepare_hessian_cache(log_partition, logits[range])
            hessian[range, range] .= last(Mooncake.value_gradient_and_hessian!!(cache, log_partition, logits[range]))
        end
        @test inner_matrix(frozen, element, 7)[1:6, 1:6] ≈ hessian rtol = sqrt(eps(element))
    end
end

@testset "PALC's norm is the information pulled back through z = λw plus the documented ω(λ) Δλ² ($element)" for element in (Float32, widen(Float32))
    rng = fresh_rng()
    ranges = (1:2, 3:3, 4:6)
    for precision_term in instances(PrecisionTerm), _ in 1:10
        state_weight, target = rand(rng, element), 1 + 3 * rand(rng, element)
        # About half the points lie past the target.
        point = [3 .* randn(rng, element, 6); 2 * target * rand(rng, element)]
        smallest = precision_term == LogPrecisionTerm ? inv(1 + target)^2 : inv(target)^2
        palc_state_weight = 1 - 9 * rand(rng, element) * smallest / 10
        frozen = PathFollowing.frozen_metric(FisherMetric(ranges, state_weight, precision_term, target, palc_state_weight), point)
        weight = documented_precision_weight(precision_term, last(point), target, palc_state_weight)
        operator = fisher_operator(ranges, point, state_weight, weight)
        # PALC adds `(1 - θ) Δλ²` to `θ ⟨Δx, M Δx⟩`.
        for matrix in (inner_matrix(frozen, element, 7), applied_matrix(frozen, element, 7))
            palc_norm = palc_state_weight .* matrix
            palc_norm[end, end] += 1 - palc_state_weight
            @test palc_norm ≈ operator rtol = sqrt(eps(element))
            @test palc_norm[end, end] ≈ operator[end, end] rtol = sqrt(eps(element))
        end
    end
end

@testset "a common move against an extinct reference keeps its Fisher length √(p(1 - p)) ($element)" for element in (Float32, widen(Float32))
    metric = FisherMetric((1:2,), one(element), TargetPrecisionTerm, one(element), one(element))
    for shift in element[10, 30, 60]
        point = element[shift, shift + 1 // 2, 1]
        frozen = PathFollowing.frozen_metric(metric, point)
        reference = inv(1 + exp(big(shift)) + exp(big(shift) + 1 // 2))
        # Every strategy but the reference moves by one: the move's variance is the reference's Bernoulli variance,
        # which the raw `Σ qᵢΔzᵢ² - (Σ qᵢΔzᵢ)²` rounds away.
        common = element[1, 1, 0]
        # `exp` rounds a log probability near `-shift` at its own size, so the reference's rounding grows with `shift`.
        tolerance = 2 * (shift + 1) * eps(element)
        @test PathFollowing.metric_inner(frozen, common, common) ≈ reference * (1 - reference) rtol = tolerance
        @test dot(PathFollowing.metric_apply!!(similar(common), frozen, common), common) ≈ reference * (1 - reference) rtol = tolerance
    end
end

@testset "each block of M v sums to M's own sum within its rounding bound, at 2, 3 and 4 strategies, against BigFloat ($element)" for element in (Float32, widen(Float32))
    rng = fresh_rng()
    # Players of 2, 3 and 4 strategies.
    ranges = (1:1, 2:3, 4:6)
    for shift in element[0, 10, 30, 60], _ in 1:10
        point = [shift .+ randn(rng, element, 6); (1 + rand(rng, element)) / 2]
        # `1 - θ` lies under the target term's `c(P) = 1 / 100`.
        state_weight, palc_state_weight = rand(rng, element), 1 - rand(rng, element) / 200
        metric = FisherMetric(ranges, state_weight, TargetPrecisionTerm, element(10), palc_state_weight)
        exact_metric = FisherMetric(ranges, big(state_weight), TargetPrecisionTerm, big(element(10)), big(palc_state_weight))
        frozen, exact_frozen = PathFollowing.frozen_metric(metric, point), PathFollowing.frozen_metric(exact_metric, big.(point))
        # The bound's domain: every probability is a normal float.
        @test all(>=(floatmin(element)), [frozen.probabilities; frozen.references])
        for (player, range) in enumerate(ranges)
            common = [index in range ? one(element) : zero(element) for index in 1:7]
            # A common move against an extinct reference has entries near its own sum, and a differential move's sum
            # outweighs its logit moves' rounding, so the bound rejects a zero block for both.
            for vector in (common, randn(rng, element, 7))
                applied = PathFollowing.metric_apply!!(similar(vector), frozen, vector)
                exact = sum(PathFollowing.metric_apply!!(similar(big.(vector)), exact_frozen, big.(vector))[range])
                @test abs(sum(big.(applied[range])) - exact) <= block_sum_bound(exact_frozen, applied, vector, player, range)
                @test abs(exact) > block_sum_bound(exact_frozen, zero(applied), vector, player, range)
            end
            other = randn(rng, element, 7)
            exact = PathFollowing.metric_inner(exact_frozen, big.(common), big.(other))
            reversed = PathFollowing.metric_apply!!(similar(common), frozen, common)
            @test abs(dot(other, reversed) - exact) <= 64 * eps(element) * (abs(exact) + dot(abs.(other), abs.(reversed)))
            # Random moves can be near orthogonal, so the error scales with both Fisher lengths.
            lengths = sqrt(PathFollowing.metric_inner(exact_frozen, big.(common), big.(common)) * PathFollowing.metric_inner(exact_frozen, big.(other), big.(other)))
            @test abs(PathFollowing.metric_inner(frozen, common, other) - exact) <= 64 * eps(element) * (abs(exact) + lengths)
        end
    end
    # Four strategies, a differential move: entries near 1/4 cannot hold M's sum -1.97e-28, so the block may sum to 0.
    point, other, common = element[60, 121 // 2, 61, 1], element[1, -1, 0, 0], element[1, 1, 1, 0]
    metric = FisherMetric((1:3,), one(element), TargetPrecisionTerm, one(element), one(element))
    frozen = PathFollowing.frozen_metric(metric, point)
    exact_frozen = PathFollowing.frozen_metric(FisherMetric((1:3,), one(BigFloat), TargetPrecisionTerm, one(BigFloat), one(BigFloat)), big.(point))
    applied = PathFollowing.metric_apply!!(similar(other), frozen, other)
    exact = PathFollowing.metric_inner(exact_frozen, big.(common), big.(other))
    @test abs(sum(big.(applied[1:3])) - exact) <= block_sum_bound(exact_frozen, applied, other, 1, 1:3)
    @test abs(PathFollowing.metric_inner(frozen, common, other) - exact) <= 64 * eps(element) * abs(exact)
    # Three strategies at logits 30 and 30.5: the product -2.2e-15 lies under the random loop's bound in Float32, but
    # the pivot keeps its relative accuracy, so a zero product fails.
    point, other, common = element[60, 61, 1 // 2], element[1, -1, 0], element[1, 1, 0]
    metric = FisherMetric((1:2,), one(element), TargetPrecisionTerm, one(element), one(element))
    exact_frozen = PathFollowing.frozen_metric(FisherMetric((1:2,), one(BigFloat), TargetPrecisionTerm, one(BigFloat), one(BigFloat)), big.(point))
    exact = PathFollowing.metric_inner(exact_frozen, big.(common), big.(other))
    @test abs(PathFollowing.metric_inner(PathFollowing.frozen_metric(metric, point), common, other) - exact) <= 64 * eps(element) * abs(exact)
    # At precision -1 `gap_scale` is negative. The bound without its absolute value was negative for the zero block of
    # [1, 1]; [1, 0] moves its logit by -1, so a zero block or a flipped `gap_scale` misses by its sum or twice it.
    point = element[1, -1]
    metric = FisherMetric((1:1,), one(element), TargetPrecisionTerm, one(element), one(element))
    exact_frozen = PathFollowing.frozen_metric(FisherMetric((1:1,), one(BigFloat), TargetPrecisionTerm, one(BigFloat), one(BigFloat)), big.(point))
    for (vector, moved) in ((element[1, 1], false), (element[1, 0], true))
        applied = PathFollowing.metric_apply!!(similar(vector), PathFollowing.frozen_metric(metric, point), vector)
        exact = only(PathFollowing.metric_apply!!(similar(big.(vector)), exact_frozen, big.(vector))[1:1])
        @test abs(big(only(applied[1:1])) - exact) <= block_sum_bound(exact_frozen, applied, vector, 1, 1:1)
        @test !moved || abs(exact) > block_sum_bound(exact_frozen, zero(applied), vector, 1, 1:1)
    end
end

@testset "player_kl is each player's categorical KL, zero at equal logits, finite where probabilities round ($element)" for element in (Float32, widen(Float32))
    rng = fresh_rng()
    ranges = (1:1, 2:3, 4:6)
    for _ in 1:20
        from, to = [4 .* randn(rng, element, 6); 2 * rand(rng, element)], [4 .* randn(rng, element, 6); 2 * rand(rng, element)]
        for range in ranges
            # One logit gives the Bernoulli KL.
            exact = direct_kl(last(from) .* big.(from[range]), last(to) .* big.(to[range]))
            @test player_kl(from, to, range) ≈ exact rtol = 1.0f-3 atol = 16 * eps(element)
            @test player_kl(from, to, range) >= -16 * eps(element)
        end
        @test largest_kl(ranges, from, to) == maximum(range -> player_kl(from, to, range), ranges)
    end
    point = element[1, -2, 3, 0, 1 // 2, 4, 1]
    @test all(range -> iszero(player_kl(point, point, range)), ranges)
    saturated, flipped = element[40, -40, 40, 40, -40, 40, 1], element[-40, 40, -40, -40, 40, -40, 1]
    @test all(range -> isfinite(player_kl(saturated, flipped, range)), ranges)
end

@testset "player_kl keeps its accuracy at large common logits, against BigFloat ($element)" for element in (Float32, widen(Float32))
    rng = fresh_rng()
    ranges = (1:1, 2:3, 4:6)
    for shift in element[0, 30, 100], step in element[1 // 1000, 1 // 100, 1 // 10], _ in 1:5
        precision, gaps = (1 + rand(rng, element)) / 2, 4 .* randn(rng, element, 6)
        from, to = [gaps .+ shift; precision], [gaps .+ step .* randn(rng, element, 6) .+ shift; precision]
        for range in ranges
            exact = direct_kl(last(from) .* big.(from[range]), last(to) .* big.(to[range]))
            # A log probability rounds by `eps`, which moves each second-order share by `eps |d|`: `eps √KL` in all.
            @test abs(player_kl(from, to, range) - exact) <= 8 * eps(element) * (sqrt(exact) + exact)
        end
    end
    # Logits 30 and 30.5 against 130 and 130.501: differencing the raw logits left `-3.8e-6` in Float32.
    from, to = element[30, 61 // 2, 1], element[130, 130501 // 1000, 1]
    exact = direct_kl(big.(from[1:2]), big.(to[1:2]))
    @test abs(player_kl(from, to, 1:2) - exact) <= 8 * eps(element) * (sqrt(exact) + exact)
end

@testset "a trace refuses a pass without logit ranges, as its corrector measures every move per player" begin
    @test_throws MethodError GameTracing.palc_geometry(TraceGeometry{Float32}(), RangelessPass(), 1, 2.0f0)
    plain = TraceGeometry{Float32}(; distance = SummedDistance, state_weight = 1 // 2)
    @test_throws MethodError trace_endpoint(LogitGame(RangelessPass(), 2.0f0), 1, BifurcationKitSettings{Float32}(; step_metric = plain))
    for (ranges, width) in (((1:1, 3:3), 3), ((1:2,), 3), ((1:0, 2:3), 3), ((2:3, 1:1), 3), ((1:2, 2:3), 3))
        @test_throws ArgumentError checked_ranges(ranges, width)
    end
    @test checked_ranges((1:1, 2:3), 3) == (1:1, 2:3)
    # A player of one strategy has no logits: its empty range starts where the next one does.
    @test checked_ranges((1:0, 1:2), 2) == (1:0, 1:2)
    @test checked_ranges((1:1, 2:1, 2:2), 2) == (1:1, 2:1, 2:2)
    for distance in (AveragedDistance, SummedDistance)
        @test_throws ArgumentError TraceGeometry{Float32}(; distance, precision_term = LogPrecisionTerm)
    end
end

@testset "a frozen Fisher metric and PALC's dot copy a range point into mutable buffers ($element)" for element in (Float32, widen(Float32))
    metric = FisherMetric((1:2,), one(element), TargetPrecisionTerm, element(10), one(element))
    point = range(one(element), element(3); length = 3)
    collected = PathFollowing.frozen_metric(metric, collect(point))
    @test inner_matrix(PathFollowing.frozen_metric(metric, point), element, 3) == inner_matrix(collected, element, 3)
    palc_dot = GameTracing.metric_dot(metric, point)
    vector = element[1, -1, 2]
    @test palc_dot.apply!(copy(vector)) == PathFollowing.metric_apply!!(similar(vector), collected, vector)
    @test GameTracing.palc_border(palc_dot, one(element) / 2, point) == GameTracing.palc_border(palc_dot, one(element) / 2, collect(point))
end

@testset "a Fisher metric frozen again holds what a fresh freeze at the new point holds ($element)" for element in (Float32, widen(Float32))
    metric = FisherMetric((1:2,), element(2), LogPrecisionTerm, element(10), element(127 // 128))
    # The pivot moves from the first strategy to the reference.
    refrozen = PathFollowing.freeze_metric!!(PathFollowing.frozen_metric(metric, element[1, -1, 1]), element[-1, -2, 3])
    fresh = PathFollowing.frozen_metric(metric, element[-1, -2, 3])
    for field in fieldnames(typeof(fresh))
        @test getfield(refrozen, field) == getfield(fresh, field)
    end
    @test fresh.root == [GameTracing.precision_dot_root(metric, element(3))]
    @test (fresh.state_weight, fresh.palc_state_weight) == (metric.state_weight, metric.palc_state_weight)
end

@testset "a Fisher metric frozen at a static point is isbits and gives the mutable one's products ($element)" for element in (Float32, widen(Float32))
    rng = fresh_rng()
    metric = FisherMetric((1:2, 3:3, 4:6), element(2), LogPrecisionTerm, element(10), element(127 // 128))
    point, left, right, moved = (randn(rng, element, 7) for _ in 1:4)
    static = PathFollowing.frozen_metric(metric, SVector{7}(point))
    @test isbits(static)
    buffered = PathFollowing.frozen_metric(metric, point)
    @test PathFollowing.metric_inner(static, SVector{7}(left), SVector{7}(right)) == PathFollowing.metric_inner(buffered, left, right)
    @test PathFollowing.metric_apply!!(zero(SVector{7, element}), static, SVector{7}(left)) == PathFollowing.metric_apply!!(zeros(element, 7), buffered, left)
    refrozen, fresh = PathFollowing.freeze_metric!!(static, SVector{7}(moved)), PathFollowing.frozen_metric(metric, moved)
    for field in fieldnames(typeof(fresh))
        @test getfield(refrozen, field) == getfield(fresh, field)
    end
end

@testset "PALC weighs Δλ² by the documented ω(λ), positive at every finite λ; PathFollowing by the log term ($element)" for element in (Float32, widen(Float32))
    unit = element[0, 1]
    for target in element[1 // 2, 1, 10, 1000]
        palc_state_weight = GameTracing.palc_state_weight_under(GameTracing.precision_term_root(LogPrecisionTerm, target, target)^2)
        palc, path = (FisherMetric((1:1,), one(element), LogPrecisionTerm, target, weight) for weight in (palc_state_weight, one(element)))
        # Past `√2 (1 + P) - 1`, `c(λ) - (1 - θ)` would turn negative and PALC's dot indefinite.
        for precision in element[0, target / 2, target, sqrt(element(2)) * (1 + target), 10 * target, 10^6 * target]
            @test only(PathFollowing.frozen_metric(palc, element[1, precision]).root) > 0
            # At a zero gap `Δz = 0`, so PALC's norm of a unit `Δλ` is its weight of `Δλ²`.
            palc_norm = GameTracing.palc_product(GameTracing.metric_dot(palc, element[0, precision]), palc_state_weight, unit, unit)
            @test palc_norm ≈ documented_precision_weight(LogPrecisionTerm, precision, target, palc_state_weight) rtol = 8 * eps(element)
            root = GameTracing.precision_term_root(LogPrecisionTerm, precision, target)
            @test PathFollowing.frozen_metric(path, element[1, precision]).root == [root]
        end
    end
    # P = 1, θ = 7/8, λ = 3: ω = 1/16 + (1/8) (1 - (1/16) / (1/4)) = 5/32.
    palc = FisherMetric((1:1,), one(element), LogPrecisionTerm, one(element), element(7 // 8))
    @test GameTracing.palc_product(GameTracing.metric_dot(palc, element[0, 3]), element(7 // 8), unit, unit) ≈ 5 // 32 rtol = 8 * eps(element)
    # Past `1 / √nextfloat(0)`, `1 / (1 + λ)²` underflows to 0: 1.1e23 in Float32, 1.8e162 in Float64. A weight of 0
    # gave a step `Δλ = λ` there length 0. Past `P = 1` the dot's weight is `c(λ) (1 - (1 - θ) / c(P))`, `c(P) = 1/4`.
    for precision in (4 / sqrt(nextfloat(zero(element))), floatmax(element)), weight in (one(element), element(7 // 8))
        frozen = PathFollowing.frozen_metric(FisherMetric((1:1,), one(element), LogPrecisionTerm, one(element), weight), element[1, precision])
        @test only(frozen.root) > 0
        exact_weight = inv(1 + big(precision))^2 * (1 - 4 * (1 - big(weight)))
        step = element[0, precision]
        @test PathFollowing.metric_inner(frozen, step, step) ≈ exact_weight * big(precision)^2 / big(weight) rtol = 8 * eps(element)
    end
end

@testset "PALC's weight of Δλ² keeps its relative accuracy near a share of one, against BigFloat ($element)" for element in (Float32, widen(Float32))
    # Float64 at the first target and lowest weight is where `1 - share` cancelled: a root of 7.5e-9 for 3.7e-10.
    for term in (TargetPrecisionTerm, LogPrecisionTerm), target in element[1402003200320032 // 10^15, 10, 1000]
        smallest = term == LogPrecisionTerm ? inv(1 + big(target))^2 : inv(big(target))^2
        lowest = element(1 - smallest, RoundUp)
        for weight in filter(<=(1), nextfloat.(lowest, 0:20)), precision in (zero(element), target / 3, target, 2 * target)
            denominator = term == LogPrecisionTerm ? 1 + min(abs(big(precision)), big(target)) : big(target)
            complement = 1 - (1 - big(weight)) * denominator^2
            computed = GameTracing.share_complement(weight, GameTracing.weight_denominator(term, precision, target))
            @test abs(computed - complement) <= eps(element) * complement + 8 * eps(element)^2
            frozen = PathFollowing.frozen_metric(FisherMetric((1:1,), one(element), term, target, weight), element[1, precision])
            exact = sqrt(complement) / (term == LogPrecisionTerm ? 1 + abs(big(precision)) : big(target))
            @test abs(only(frozen.root) - exact) <= eps(element) * (3 + 4 * eps(element) / complement) * exact
        end
    end
end

@testset "a Fisher metric refuses a PALC state weight outside (1 - c(P), 1] ($element)" for element in (Float32, widen(Float32))
    for (term, target, weight) in ((LogPrecisionTerm, 1, 3 // 4), (TargetPrecisionTerm, 10, 49 // 50), (LogPrecisionTerm, 1, 0), (TargetPrecisionTerm, 1, 2))
        @test_throws ArgumentError FisherMetric((1:1,), one(element), term, element(target), element(weight))
    end
    # `θ = 1` serves every target, also where `c(P)` underflows.
    @test FisherMetric((1:1,), one(element), LogPrecisionTerm, floatmax(element), one(element)).palc_state_weight == 1
end

@testset "the pivot comes from the gaps, and centered logits stay exact where gaps or products overflow ($element)" for element in (Float32, widen(Float32))
    huge = floatmax(element) / 2
    wide = huge + huge / 8
    # `λ wᵢ` overflows for both strategies: a pivot chosen from those products was strategy 1, and the normalizer infinite.
    @test GameTracing.log_normalizer(element[huge, wide, 4], 1:2) == (2, zero(element))
    # `w₁ - w₂` overflows, and `λ w₁ - λ w₂` does not. The KL of a point against itself was NaN.
    spread = element[-wide, wide, 1 // 2]
    @test GameTracing.centered_logit(spread, 1, 2) == -wide
    @test iszero(player_kl(spread, spread, 1:2))
    # At precision `1 / huge` the logits are near -9/8 and 9/8: an overflowed difference gave strategy 1 probability 0.
    moderate, doubled = element[-wide, wide, inv(huge)], element[-wide, wide, 2 / huge]
    exact = direct_kl(last(moderate) .* big.(moderate[1:2]), last(doubled) .* big.(doubled[1:2]))
    @test abs(player_kl(moderate, doubled, 1:2) - exact) <= 8 * eps(element) * (sqrt(exact) + exact)
    # At precision 4 strategy 1's and the reference's logits lie past -floatmax: probability 0.
    extreme = element[-wide, wide, 4]
    @test player_kl(moderate, extreme, 1:2) == typemax(element)
    exact = -log(inv(1 + sum(exp, big(last(moderate)) .* [-big(wide); -2 * big(wide)])))
    @test player_kl(extreme, moderate, 1:2) ≈ exact rtol = 8 * eps(element)
    # A negative precision flips the logits to 1 and 3: the pivot is the smallest gap.
    flipped = element[-1, -3, -1]
    @test GameTracing.log_normalizer(flipped, 1:2)[1] == 2
    @test GameTracing.log_normalizer(flipped, 1:2)[2] ≈ log(1 + exp(-big(3)) + exp(-big(2))) rtol = 4 * eps(element)
end

@testset "the log normalizer keeps its relative accuracy where the other strategies' sum is under eps ($element)" for element in (Float32, widen(Float32))
    # Logit -20 against the reference: `log(1 + e⁻²⁰)` rounded the normalizer to 0 in Float32; it is 2.06e-9.
    for logit in element[-20, -40]
        normalizer = GameTracing.log_normalizer(element[logit, 1], 1:1)
        @test first(normalizer) == 0
        @test last(normalizer) ≈ log1p(exp(big(logit))) rtol = 4 * eps(element)
    end
end
