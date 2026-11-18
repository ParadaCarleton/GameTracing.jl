using ErrorTypes: is_error, unwrap, unwrap_error
using LinearAlgebra: dot

@testset "each distance weighs coordinates and precision as documented, the Fisher distances measuring in the summed norm" begin
    # Each coordinate is the old unit `[P w; λ / P]` displacement `[1, 2, 3]` in `(w, λ)`, at target precision 3.
    target = 3.0f0
    displacement = Float32[1, 2, 3] .* [inv(target), inv(target), target]
    for (distance, precision_term, expected) in (
            (AveragedDistance, TargetPrecisionTerm, 0.4f0 * 5 / 2 + 0.6f0 * 9), (SummedDistance, TargetPrecisionTerm, 0.4f0 * 5 + 9),
            (FisherDistance, TargetPrecisionTerm, 0.4f0 * 5 + 9), (FisherDistance, LogPrecisionTerm, 0.4f0 * 5 + 9),
        )
        geometry = TraceGeometry{Float32}(; distance, state_weight = 0.4f0, precision_term)
        @test GameTracing.arclength_distance(zeros(Float32, 3), displacement, geometry, target)^2 ≈ expected
    end
end

@testset "PALC's norm in each distance is the trace's own, on two-strategy and categorical games ($element)" for element in (Float32, widen(Float32))
    rng = fresh_rng()
    categorical = NormalFormGains((randn(rng, element, 3, 4), randn(rng, element, 3, 4)))
    for (pass, width) in ((GAME, WIDTH), (categorical, logit_count(categorical))), geometry in every_geometry(element(3 // 10)), _ in 1:5
        target = element(3)
        palc = GameTracing.palc_geometry(geometry, pass, width, target)
        point, displacement = [randn(rng, element, width); 3 * rand(rng, element)], randn(rng, element, width + 1)
        GameTracing.BifurcationKit.reweight!(palc.dot, GameTracing.BifurcationKit.BorderedArray(point, last(point)))
        # PALC's pin moves with the precision, so it repeats the displacement's last coordinate.
        palc_square = palc.dot(displacement, displacement, last(displacement), last(displacement), palc.palc_state_weight)
        if geometry.distance == FisherDistance
            precision_weight = geometry.precision_term == LogPrecisionTerm ? inv(1 + last(point))^2 : inv(target)^2
            operator = fisher_operator(GameTracing.logit_ranges(pass), point, geometry.state_weight, precision_weight)
            @test palc_square ≈ dot(displacement, operator, displacement) rtol = sqrt(eps(element))
        else
            @test palc_square ≈ GameTracing.arclength_distance(zeros(element, width + 1), displacement, geometry, target)^2 rtol = sqrt(eps(element))
        end
    end
end

@testset "PALC's dot weighs the precision positively over the trace's range, at every target" begin
    for target_precision in (0.01f0, 0.5f0, 1.0f0, 4.0f0, 100.0f0), geometry in every_geometry(0.01f0)
        palc = GameTracing.palc_geometry(geometry, GAME, WIDTH, target_precision)
        @test 0 < palc.palc_state_weight < 1
        unit = Float32[0, 0, 1]
        for precision in range(0.0f0, target_precision; length = 101)
            # At zero gaps the precision moves no logit, so the dot holds the precision term alone.
            GameTracing.BifurcationKit.reweight!(palc.dot, GameTracing.BifurcationKit.BorderedArray(Float32[0, 0, precision], precision))
            @test palc.dot.dot(unit, unit) > 0
        end
    end
end

@testset "a converged corrector past the KL or reach limit is refused, in the default geometry" begin
    geometry = TraceGeometry{Float32}()
    start, predicted = Float32[0, 0, 0.5f0], Float32[1, 0, 0.5f0]
    # The predictor moves the first logit by 0.5; this moves it 0.5 more, twice the reach of 0.5 times that move.
    beyond_reach = predicted .+ Float32[1, 0, 0]
    @test GameTracing.logit_change(predicted, beyond_reach) ≈ 2 * 0.5f0 * GameTracing.logit_change(start, predicted)
    for (corrector, corrected) in (
            (CorrectorSettings{Float32}(reach = nothing, max_kl = 1.0f-3), Float32[3, 0, 0.5f0]),
            (CorrectorSettings{Float32}(reach = 0.5f0), beyond_reach),
        )
        check = GameTracing.CorrectorReach(
            GameTracing.AcceptAll(),
            corrector,
            geometry,
            TARGET_PRECISION,
            start,
            predicted,
            1.0f0,
            GameTracing.kl_ranges(corrector.max_kl, GAME, WIDTH),
        )
        GameTracing.restate!(check; predictor_move = GameTracing.logit_change(start, predicted))
        @test GameTracing.closing_accepted(check, predicted)
        @test !GameTracing.closing_accepted(check, corrected)
        # An exact root that never settled ends BifurcationKit's solve converged, so it meets the same tests.
        @test check((; x = corrected, res_f = zero(corrected), p = 0.5f0, step = 2, residual = 0.0f0, closing = true); fromNewton = false) === false
        # The closing call reads the solve's own verdict: it settled.
        GameTracing.restate!(check; settled_here = true)
        @test check((; x = corrected, res_f = zero(corrected), p = 0.5f0, step = 2, residual = 0.0f0, closing = true); fromNewton = false) === false
    end
end

@testset "a corrector check over an immutable start and predictor writes its own buffers ($element)" for element in (Float32, widen(Float32))
    quarter, half = element(0.25f0), element(0.5f0)
    start, predicted = zero(element):quarter:half, one(element):(-quarter):half
    check = GameTracing.CorrectorReach(
        GameTracing.AcceptAll(), CorrectorSettings{element}(), TraceGeometry{element}(), element(TARGET_PRECISION), start, predicted, one(element), nothing,
    )
    @test check.predicted == predicted && check.iterate == start
    first_iterate = (; x = collect(predicted), res_f = zeros(element, 3), p = half, step = 0, residual = zero(element), z0 = (; u = collect(start), p = half))
    @test check(first_iterate; fromNewton = false) !== false
    @test check.predicted == predicted
end

@testset "a point closed at an exact root records a correction of 0, a settled one its last correction" begin
    previous, predicted = Float32[0, 0, 0.5f0], Float32[1, 0, 0.5f0]
    # BifurcationKit gives the closing call the step's start `z0` too; its `closing` marker sets it apart.
    closing = (; x = predicted, res_f = zero(predicted), p = 0.5f0, step = 2, residual = 0.0f0, z0 = (; u = previous, p = 0.5f0), closing = true)
    for (settled, recorded) in ((false, 0.0f0), (true, 0.25f0))
        check = GameTracing.CorrectorReach(
            GameTracing.AcceptAll(), CorrectorSettings{Float32}(), TraceGeometry{Float32}(), TARGET_PRECISION, previous, predicted, 1.0f0, nothing,
        )
        GameTracing.restate!(check; predictor_move = GameTracing.logit_change(previous, predicted))
        GameTracing.restate!(check; correction = 0.25f0, settled_here = settled)
        @test check(closing; fromNewton = false) !== false
        @test check.state[].noise.accuracy == recorded
    end
end

@testset "a projection in the summed norm lands on the branch" begin
    game = LogitGame(GAME, TARGET_PRECISION)
    traced = unwrap(trace_endpoint(game, WIDTH, BifurcationKitSettings{Float32}()))
    settings = BifurcationKitSettings{Float32}(step_metric = TraceGeometry{Float32}(distance = SummedDistance))
    projected = branch_projection(game, traced.logits .+ 1.0f-2, TARGET_PRECISION, settings)
    @test residual_norm(GAME, projected.logits, projected.precision) < 1.0f-3
end

"""Whether `settings` traces `payoffs` to the fine Float64 reference at `precision`, or to an equilibrium where the branch folds back."""
function reaches_reference(payoffs::Tuple, precision::Real, settings::BifurcationKitSettings{Float32})::Bool
    pass = NormalFormPass(map(table -> Float32.(table), payoffs))
    result = trace_endpoint(LogitGame(pass, Float32(precision)), length(payoffs), settings)
    if is_error(result)
        return false
    end
    logits = unwrap(result).logits
    reference = branch_reference(NormalFormPass(map(float, payoffs)), length(payoffs), precision)
    if reference.reached
        # First-action probabilities to 1e-3, as the LogitNash cross-check compares them.
        return maximum(abs, inv.(1 .+ exp.(.-logits)) .- reference.first_choices) < 1.0f-3
    end
    return residual_norm(pass, logits, Float32(precision)) < 1.0f-3
end

@testset "every arm reaches the reference equilibrium on LogitNash's games, the overshot game and random games" begin
    rng = fresh_rng()
    random_games = [ntuple(_ -> 3 .* randn(rng, ntuple(_ -> 2, players)), players) for players in rand(rng, 2:4, 6)]
    for (label, settings) in arm_variants(ARM_WEIGHTS)
        for (name, payoffs) in pairs(LOGITNASH_GAMES)
            if !(name in PITCHFORK_GAMES)
                @test reaches_reference(payoffs, 4, settings)
            end
        end
        for precision in (3, 31 // 10)
            @test reaches_reference(OVERSHOT_GAME, precision, settings)
        end
        for payoffs in random_games
            reached = reaches_reference(payoffs, 3, settings)
            @test reached
            if !reached
                @info "arm missed the reference" label payoffs
            end
        end
    end
end

@testset "every arm crosses the fold game's fold" begin
    for (_, settings) in arm_variants(ARM_WEIGHTS)
        result = trace_endpoint(LogitGame(FOLD_GAME, 2.0f0), 1, settings)
        @test !is_error(result)
        if !is_error(result)
            @test residual_norm(FOLD_GAME, unwrap(result).logits, 2.0f0) < 1.0f-3
            @test only(unwrap(result).logits) > 3
        end
    end
end

@testset "every arm stops short of a non-finite game as a stuck trace" begin
    for (label, settings) in arm_variants(ARM_WEIGHTS)
        result = trace_endpoint(LogitGame(PoisonedPass(GAME, 1.0f0), TARGET_PRECISION), WIDTH, settings)
        @test is_error(result)
        @test unwrap_error(result).cause == GameTracing.TraceStuck
        if !is_error(result) || unwrap_error(result).cause != GameTracing.TraceStuck
            @info "arm did not stop as stuck" label
        end
    end
end

@testset "the redraw restart's border row gives PALC's product with the old tangent, in every distance" begin
    rng = fresh_rng()
    for geometry in every_geometry(0.3f0), _ in 1:5
        palc = GameTracing.palc_geometry(geometry, GAME, WIDTH, 3.0f0)
        point = randn(rng, Float32, 3)
        GameTracing.BifurcationKit.reweight!(palc.dot, GameTracing.BifurcationKit.BorderedArray(point, last(point)))
        old_tangent, tangent = randn(rng, Float32, 3), randn(rng, Float32, 3)
        border = GameTracing.palc_border(palc.dot, palc.palc_state_weight, old_tangent)
        @test border' * tangent ≈ GameTracing.palc_product(palc.dot, palc.palc_state_weight, old_tangent, tangent) rtol = 1.0f-4
    end
end
