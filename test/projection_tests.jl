using GameTracing: BranchWorkspace, branch_projection, branch_system!, logits_at, projection_step
using ErrorTypes: unwrap
using LinearAlgebra: Diagonal, dot, norm, normalize, nullspace
using Random: RandomDevice, Xoshiro

"""The averaged norm's weights `[P² state_weight / k … P² state_weight / k, (1 - state_weight) / P²]` over a branch point of `width` logits at target precision `P`."""
function palc_weights(width::Integer, state_weight::E, target_precision::Real)::Diagonal{E} where {E <: Real}
    return Diagonal([fill(target_precision^2 * state_weight / width, width); (1 - state_weight) / target_precision^2])
end

"""`game`'s residual `F` and its Jacobian `J` in payoff gaps `[logits / λ; λ]` at `point`."""
function branch_equations(game::LogitGame, point::AbstractVector{E})::NamedTuple where {E <: Real}
    width = length(point) - 1
    system = branch_system!(BranchWorkspace(game, point), game, point, last(point))
    return (; residual = system.residual[1:width], jacobian = system.jacobian[1:width, :])
end

@testset "projection_step is the minimum-norm Newton step in PALC's norm" begin
    rng = Xoshiro(rand(RandomDevice(), UInt))
    games = (
        LogitGame(LinearGainPass(float.(GAME.interaction), float.(GAME.stand_alone)), float(TARGET_PRECISION)),
        LogitGame(ThresholdPass(float(4), float(2)), float(1)),
        LogitGame(LinearGainPass(randn(rng, 3, 3), randn(rng, 3)), float(2)),
    )
    for (game, width) in zip(games, (2, 1, 3)), state_weight in (float(1 // 5), float(1 // 2), float(9 // 10))
        for _ in 1:5
            logits = randn(rng, width) ./ 10
            precision = game.target_precision * (1 // 2 + 2 * rand(rng) / 5)
            step = projection_step(game, logits, precision, TraceGeometry{typeof(state_weight)}(; distance = AveragedDistance, state_weight))
            start = branch_equations(game, [logits ./ precision; precision])
            # Newton's linearized equation holds, and no part of the step runs along the branch's tangent.
            @test norm(start.jacobian * step .+ start.residual) <= inv(float(10^10)) * (1 + norm(start.residual))
            tangent = vec(nullspace(start.jacobian))
            @test abs(dot(palc_weights(width, state_weight, game.target_precision) * step, tangent)) <= inv(float(10^10)) * (1 + norm(step))
        end
    end
end

"""The residual `branch_projection` leaves from `on_branch`'s logits moved `offset` along `direction`."""
function projected_residual(
        game::LogitGame,
        on_branch::NamedTuple,
        offset::Real,
        direction::AbstractVector,
        settings::BifurcationKitSettings,
    )::typeof(float(offset))
    projected = branch_projection(game, on_branch.logits .+ offset .* direction, on_branch.precision, settings)
    return norm(branch_equations(game, [projected.logits ./ projected.precision; projected.precision]).residual)
end

@testset "branch_projection converges quadratically onto the branch" begin
    rng = Xoshiro(rand(RandomDevice(), UInt))
    game = LogitGame(LinearGainPass(randn(rng, 3, 3), randn(rng, 3)), float(2))
    settings = BifurcationKitSettings{typeof(float(1))}()
    on_branch = (; logits = zeros(3), precision = float(1))
    for _ in 1:10
        on_branch = branch_projection(game, on_branch.logits, on_branch.precision, settings)
    end
    @test norm(branch_equations(game, [on_branch.logits ./ on_branch.precision; on_branch.precision]).residual) < inv(float(10^12))
    direction = normalize(randn(rng, 3))
    # Quadratic convergence drops the residual left a hundredfold per decade of offset; linear, tenfold.
    coarse = projected_residual(game, on_branch, inv(float(100)), direction, settings)
    fine = projected_residual(game, on_branch, inv(float(1000)), direction, settings)
    @test fine <= coarse / 30
end

@testset "branch_projection applies one projection_step in the trace's norm" begin
    game = LogitGame(GAME, TARGET_PRECISION)
    settings = BifurcationKitSettings{Float32}(step_metric = TraceGeometry{Float32}(state_weight = 3 // 10))
    logits, precision = Float32[1 // 10, -1 // 5], 3.0f0 / 4 * TARGET_PRECISION
    step = projection_step(game, logits, precision, settings.step_metric)
    projected = branch_projection(game, logits, precision, settings)
    @test projected.logits == logits_at([logits ./ precision; precision] .+ step)
    @test projected.precision == precision + last(step)
    @test projected.logits isa Vector{Float32}
end

@testset "projection_step rejects a state weight outside (0, 1)" begin
    for state_weight in (0.0f0, 1.0f0, -0.5f0, 1.5f0, NaN32)
        @test_throws ArgumentError projection_step(LogitGame(GAME, TARGET_PRECISION), zeros(Float32, WIDTH), TARGET_PRECISION / 2, TraceGeometry{Float32}(; state_weight))
    end
end

@testset "a projection that would leave the positive precisions throws" begin
    # From far off the branch of G = λ the Gauss–Newton step overshoots λ = 0, to λ = -4.95.
    game = LogitGame(PrecisionGainPass(), 1.0f0)
    settings = BifurcationKitSettings{Float32}()
    @test_throws DomainError branch_projection(game, Float32[-1], 0.1f0, settings)
    # Near the branch z = λ² the same game projects to a positive precision.
    @test branch_projection(game, Float32[0.26f0], 0.5f0, settings).precision > 0
end

@testset "projection_step refuses a point with no payoff gaps, at a precision of zero or not finite" begin
    for precision in (0.0f0, -0.0f0, NaN32, Inf32)
        @test_throws ArgumentError projection_step(LogitGame(GAME, TARGET_PRECISION), zeros(Float32, WIDTH), precision, TraceGeometry{Float32}())
    end
end

@testset "branch_projection leaves a point on the branch where it is" begin
    traced = unwrap(trace_endpoint(LogitGame(GAME, TARGET_PRECISION), WIDTH, BifurcationKitSettings{Float32}()))
    projected = branch_projection(LogitGame(GAME, TARGET_PRECISION), traced.logits, TARGET_PRECISION, BifurcationKitSettings{Float32}())
    @test maximum(abs, projected.logits .- traced.logits) < 1.0f-4
    @test abs(projected.precision - TARGET_PRECISION) < 1.0f-4 * TARGET_PRECISION
end

@testset "near a fold, where a correction at fixed precision is singular, the projection converges quadratically" begin
    # On ThresholdPass(a, t) the branch is precision = z / (a σ(z - t)); it folds where z (1 - σ(z - t)) = 1.
    amplitude, threshold, target = float(6), float(3), float(2)
    sigmoid(logit::Real) = inv(1 + exp(-logit))
    low, high = float(1), float(2)
    for _ in 1:60
        middle = (low + high) / 2
        if middle * (1 - sigmoid(middle - threshold)) < 1
            low = middle
        else
            high = middle
        end
    end
    fold = (low + high) / 2
    fold_precision = fold / (amplitude * sigmoid(fold - threshold))
    game = LogitGame(ThresholdPass(amplitude, threshold), target)
    # At the fold the derivative in the logit vanishes, so Newton at fixed precision has no step.
    at_fold = branch_equations(game, [fold / fold_precision; fold_precision])
    @test abs(only(at_fold.residual)) <= inv(float(10^10))
    @test abs(first(at_fold.jacobian)) <= inv(float(10^8)) * norm(at_fold.jacobian)
    # Newton at fixed precision meets a double root here and converges only linearly; the projection stays quadratic.
    settings = BifurcationKitSettings{typeof(float(1))}(step_metric = TraceGeometry{typeof(float(1))}(state_weight = 2 // 5))
    on_branch = (; logits = [fold], precision = fold_precision)
    for direction in ([-one(fold)], [one(fold)])
        coarse = projected_residual(game, on_branch, inv(float(100)), direction, settings)
        fine = projected_residual(game, on_branch, inv(float(1000)), direction, settings)
        @test fine <= coarse / 30
    end
end
