using ErrorTypes: is_error
using GameTracing: FisherMetric, checked_ranges, largest_kl, player_kl
using LinearAlgebra: dot
using Mooncake: Mooncake
using PathFollowing: PathFollowing

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

"""A one-logit pass with zero gain and no `logit_ranges`."""
struct RangelessPass end

function (::RangelessPass)(
        gain::AbstractVector,
        gain_slope::AbstractMatrix,
        gain_precision_slope::AbstractVector,
        ::AbstractVector,
        ::Real,
    )::Nothing
    fill!(gain, zero(eltype(gain)))
    fill!(gain_slope, zero(eltype(gain_slope)))
    fill!(gain_precision_slope, zero(eltype(gain_precision_slope)))
    return nothing
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

@testset "the Fisher metric is the information pulled back through z = λw, past PALC's own term ($element)" for element in (Float32, widen(Float32))
    rng = fresh_rng()
    ranges = (1:2, 3:3, 4:6)
    for precision_term in instances(PrecisionTerm), _ in 1:10
        point = [3 .* randn(rng, element, 6); 2 * rand(rng, element)]
        state_weight, target, palc_state_weight = rand(rng, element), 1 + 3 * rand(rng, element), (1 + rand(rng, element)) / 2
        metric = FisherMetric(ranges, state_weight, precision_term, target, palc_state_weight)
        frozen = PathFollowing.frozen_metric(metric, point)
        precision_weight = precision_term == LogPrecisionTerm ? inv(1 + last(point))^2 : inv(target)^2
        # PALC adds `(1 - θ) Δλ²` to `θ ⟨Δx, Δx⟩`, so `θ M + (1 - θ) e eᵀ` is the distance's operator.
        operator = fisher_operator(ranges, point, state_weight, precision_weight)
        operator[end, end] -= 1 - palc_state_weight
        operator ./= palc_state_weight
        @test inner_matrix(frozen, element, 7) ≈ operator rtol = sqrt(eps(element))
        @test applied_matrix(frozen, element, 7) ≈ operator rtol = sqrt(eps(element))
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
        @test dot(PathFollowing.metric_apply!(similar(common), frozen, common), common) ≈ reference * (1 - reference) rtol = tolerance
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

@testset "a Fisher trace or KL limit refuses a pass without logit ranges; a plain trace needs none" begin
    @test_throws MethodError GameTracing.palc_geometry(TraceGeometry{Float32}(), RangelessPass(), 1, 2.0f0)
    @test_throws MethodError GameTracing.kl_ranges(1.0f-3, RangelessPass(), 1)
    plain = TraceGeometry{Float32}(; distance = SummedDistance, state_weight = 1 // 2)
    @test !is_error(trace_endpoint(LogitGame(RangelessPass(), 2.0f0), 1, BifurcationKitSettings{Float32}(; step_metric = plain)))
    for (ranges, width) in (((1:1, 3:3), 3), ((1:2,), 3), ((1:0, 1:2), 2), ((2:3, 1:1), 3), ((1:2, 2:3), 3))
        @test_throws ArgumentError checked_ranges(ranges, width)
    end
    @test checked_ranges((1:1, 2:3), 3) == (1:1, 2:3)
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
    @test palc_dot.apply!(copy(vector)) == PathFollowing.metric_apply!(similar(vector), collected, vector)
    @test GameTracing.palc_border(palc_dot, one(element) / 2, point) == GameTracing.palc_border(palc_dot, one(element) / 2, collect(point))
end
