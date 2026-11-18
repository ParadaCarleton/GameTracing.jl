using BifurcationKit: BifurcationKit
using ErrorTypes: Err, Ok, Result, is_error, unwrap, unwrap_error
using LogitNash: LogitNash

"""
LogitNash's QRE of `payoffs` past `precision`: `(; first_choices, precision)`, each player's probability of its first
action and the precision LogitNash stopped at, which may pass the one asked for.
"""
function logitnash_equilibrium(payoffs::Tuple, precision::Real)::NamedTuple
    profile, status = LogitNash.solve(map(float, payoffs); stop_lambda = float(precision), stop_eps = zero(float(precision)))
    return (; first_choices = Float32[first(strategy) for strategy in profile], precision = Float32(status.lambda))
end

# A trace's endpoint in `NormalFormPass`'s logits, each player's log odds of its first action, and its logit accuracy.
const Endpoint{E} = @NamedTuple{logits::Vector{E}, logit_accuracy::E}
const EndpointResult{E} = Result{Endpoint{E}, TraceFailure}

"""BifurcationKit's endpoint for `payoffs` at `precision` (`trace_endpoint` on `NormalFormPass`), its accuracy its own."""
function traced_result(payoffs::Tuple, precision::E, settings::BifurcationKitSettings{E})::EndpointResult{E} where {E <: Real}
    result = trace_endpoint(LogitGame(NormalFormPass(map(table -> E.(table), payoffs)), precision), length(payoffs), settings)
    if is_error(result)
        return Err(unwrap_error(result))
    end
    (; logits, logit_accuracy) = unwrap(result)
    return Ok((; logits, logit_accuracy))
end

"""
PathFollowing's endpoint for `payoffs` at `precision`: `logit_solve_lambda` on `NormalFormGains` lands there. Its logit
is the second action's log odds, so the sign flips. PathFollowing reports no accuracy, so the accuracy is the Float64
Newton correction at the endpoint (`newton_correction`).
"""
function traced_result(payoffs::Tuple, precision::E, settings::PathFollowingSettings{E})::EndpointResult{E} where {E <: Real}
    result = logit_solve_lambda(NormalFormGains(map(table -> E.(table), payoffs)), precision; settings)
    if is_error(result)
        return Err(unwrap_error(result))
    end
    logits = -only(unwrap(result)).logits
    return Ok((; logits, logit_accuracy = E(newton_correction(NormalFormPass(map(float, payoffs)), logits, precision))))
end

"""BifurcationKit's settings at each tangent, by name: the variants every trace test runs."""
function tangent_settings()::Vector{Pair{String, EngineSettings{Float32}}}
    return ["BifurcationKit Bordered" => BifurcationKitSettings{Float32}(; tangent = BifurcationKit.Bordered()), "BifurcationKit Secant" => BifurcationKitSettings{Float32}(; tangent = BifurcationKit.Secant())]
end

"""PathFollowing's default settings in Float32 and Float64, by name: the engine every trace test runs beside BifurcationKit."""
function path_settings()::Vector{Pair{String, EngineSettings}}
    return ["PathFollowing Float32" => PathFollowingSettings{Float32}(), "PathFollowing Float64" => PathFollowingSettings{widen(Float32)}()]
end

"""BifurcationKit's default settings in Float32 and PathFollowing's (`path_settings`), by name."""
function default_settings()::Vector{Pair{String, EngineSettings}}
    return ["BifurcationKit" => BifurcationKitSettings{Float32}(); path_settings()]
end

"""
Whether a `TraceStuck` `failure` of `settings`' engine stalled where it says: BifurcationKit at a point that did not move
or a step it could not resolve, PathFollowing at an unresolved step or its refusals.
"""
function declared_stall(failure::TraceFailure, ::BifurcationKitSettings)::Bool
    return failure.context.unmoved || failure.context.unresolved
end

function declared_stall(failure::TraceFailure, ::PathFollowingSettings)::Bool
    return failure.context.reason in (:unresolved_step, :refusals)
end

"""The last accepted logits of a failed trace of `settings`' engine, in `NormalFormPass`' logits."""
pass_logits(failure::TraceFailure, ::BifurcationKitSettings)::Vector = collect(failure.context.logits)
pass_logits(failure::TraceFailure, ::PathFollowingSettings)::Vector = -collect(failure.context.logits)

"""First-action probabilities of `logits`."""
function first_choices(logits::AbstractVector{E})::Vector{E} where {E <: Real}
    return inv.(one(E) .+ exp.(-logits))
end

"""Whether two first-action profiles agree to `1e-3` in every probability."""
agreeing(left::AbstractVector, right::AbstractVector)::Bool = maximum(abs, left .- right) < 1.0f-3

"""`settled_logits` for the two-action payoff tables `payoffs` (`NormalFormPass`)."""
function settled_endpoint(payoffs::Tuple, logits::AbstractVector{E}, precision::Real, settings::BifurcationKitSettings{E})::Bool where {E <: Real}
    return settled_logits(NormalFormPass(map(table -> E.(table), payoffs)), logits, precision, settings)
end

"""
The largest endpoint accuracy that `settings`' corrector allows at `logits` of `payoffs` (`settle_bound`): BifurcationKit's
in logits, PathFollowing's in strategies, on `NormalFormGains`' logits, the negated `logits`.
"""
function endpoint_bound(::Tuple, logits::AbstractVector, settings::BifurcationKitSettings{E})::E where {E <: Real}
    return settle_bound(settings.corrector, logits)
end

function endpoint_bound(payoffs::Tuple, logits::AbstractVector, settings::PathFollowingSettings{E})::E where {E <: Real}
    return settle_bound(settings.distance, NormalFormGains(map(table -> E.(table), payoffs)), -logits)
end

"""`settled_strategies` for the two-action payoff tables `payoffs`, on `NormalFormGains`' logits, the negated `logits`."""
function settled_endpoint(payoffs::Tuple, logits::AbstractVector{E}, precision::Real, settings::PathFollowingSettings{E})::Bool where {E <: Real}
    return settled_strategies(NormalFormGains(map(table -> E.(table), payoffs)), -logits, precision, settings.distance)
end

"""
    cross_checked!(payoffs, precision, settings)

Trace `payoffs` to LogitNash's stopping precision past `precision` and test the endpoint against Gambit's branch there,
the oracle, and against `branch_reference`, the Julia cross-check. LogitNash is a cross-check too: a disagreement with it
fails only when Gambit sides with LogitNash. Near an
unfolded pitchfork LogitNash can cross onto the middle arm, as on `UNFOLDED_PITCHFORK_GAME`. A reference that does not
reach the target skips the comparison; every endpoint must also solve the map to the corrector's tolerance. Where
Gambit's branch folds before the target (`gambit_folds_before`), Gambit's answer may be a later crossing, so the Gambit
comparisons are skipped and the reference decides. Returns whether they were skipped.
"""
function cross_checked!(payoffs::Tuple, precision::Real, settings::EngineSettings{E})::Bool where {E <: Real}
    cross = logitnash_equilibrium(payoffs, precision)
    result = traced_result(payoffs, E(cross.precision), settings)
    @test !is_error(result)
    if is_error(result)
        @info "trace failed" payoffs unwrap_error(result).cause
        return false
    end
    logits = unwrap(result).logits
    traced = first_choices(logits)
    folded = gambit_folds_before(payoffs, cross.precision)
    if !folded
        oracle = gambit_branch(payoffs, cross.precision)
        @test agreeing(traced, oracle.first_choices)
        if !agreeing(traced, cross.first_choices)
            @test !agreeing(oracle.first_choices, cross.first_choices)
        end
    end
    reference = branch_reference(NormalFormPass(map(float, payoffs)), length(payoffs), cross.precision)
    # A generic game's branch crosses no singular point.
    @test isempty(reference.crossings)
    if reference.reached
        @test agreeing(traced, reference.first_choices)
    end
    @test settled_endpoint(payoffs, logits, cross.precision, settings)
    return folded
end

@testset "Gambit's λ is the trace's precision: its profile solves z = λ G on the raw payoffs" begin
    rng = fresh_rng()
    for _ in 1:5
        players = rand(rng, 2:3)
        payoffs = ntuple(_ -> 3 .* randn(rng, ntuple(_ -> 2, players)), players)
        precision = 1 + 2 * rand(rng)
        oracle = gambit_branch(payoffs, precision)
        @test abs(oracle.precision - precision) < inv(float(10^6))
        crossing = log.(oracle.first_choices) .- log1p.(-oracle.first_choices)
        response = evaluated(NormalFormPass(payoffs), crossing, oracle.precision).response
        @test maximum(abs, inv.(1 .+ exp.(-response)) .- oracle.first_choices) < inv(float(10^8))
        # Gambit does not normalize payoffs: doubling them doubles λ.
        doubled = gambit_branch(map(table -> 2 .* table, payoffs), precision / 2)
        @test maximum(abs, doubled.first_choices .- oracle.first_choices) < inv(float(10^8))
    end
    # Halving λ alone moves the profile, so the doubling check above can fail. A random game near a pure profile barely
    # moves, so this uses a fixed game: matching pennies, with the row player's payoffs made asymmetric.
    pennies = ([3 -1; -1 1], [-1 1; 1 -1])
    @test maximum(abs, gambit_branch(pennies, 1.0f0).first_choices .- gambit_branch(pennies, 2.0f0).first_choices) > inv(float(10^4))
end

@testset "NormalFormPass: the logit slope is the derivative of the response" begin
    rng = fresh_rng()
    for payoffs in values(LOGITNASH_GAMES)
        pass = NormalFormPass(map(table -> float.(table), payoffs))
        logits = randn(rng, length(payoffs))
        step = inv(float(10^6))
        level = float(3 // 2)
        slope = evaluated(pass, logits, level).logit_slope
        for (rival, _) in pairs(logits)
            nudge = step .* (eachindex(logits) .== rival)
            upper, lower = evaluated(pass, logits .+ nudge, level), evaluated(pass, logits .- nudge, level)
            @test maximum(abs, (upper.response .- lower.response) ./ (2 * step) .- slope[:, rival]) < inv(float(10^6))
        end
        @test evaluated(pass, logits, level).precision_slope ≈ evaluated(pass, logits, 1).response
    end
end

@testset "branch_reference: each endpoint solves the map, and LogitNash's equilibria solve NormalFormPass" begin
    for (name, payoffs) in pairs(LOGITNASH_GAMES), precision in (1, 4)
        cross = logitnash_equilibrium(payoffs, precision)
        pass = NormalFormPass(map(float, payoffs))
        reference = branch_reference(pass, length(payoffs), cross.precision)
        @test reference.reached
        # Only the exact battle of the sexes puts a singular point, its pitchfork, on the branch.
        @test length(reference.crossings) <= (name == :battle_of_the_sexes)
        @test residual_norm(pass, reference.logits, cross.precision * float(1)) < inv(float(10^6))
        # In probability space: LogitNash's profile may round a choice to certainty, an infinite logit.
        crossing = log.(cross.first_choices) .- log1p.(-cross.first_choices)
        response = evaluated(NormalFormPass(map(table -> Float32.(table), payoffs)), crossing, cross.precision).response
        @test maximum(abs, inv.(1 .+ exp.(-response)) .- cross.first_choices) < 1.0f-4
    end
end

@testset "the trace reaches the reference equilibrium on LogitNash's games ($engine)" for (engine, settings) in [tangent_settings(); path_settings()]
    for (name, payoffs) in pairs(LOGITNASH_GAMES), precision in (1, 4)
        if !(name in PITCHFORK_GAMES)
            cross_checked!(payoffs, precision, settings)
        end
    end
end

@testset "past an exact pitchfork the trace ends at an equilibrium or fails at the pitchfork, on the branch ($engine)" for (engine, settings) in [tangent_settings(); path_settings()]
    element = GameTracing.element_type(settings)
    for name in PITCHFORK_GAMES
        pass = NormalFormPass(map(table -> element.(table), LOGITNASH_GAMES[name]))
        reference_pass = NormalFormPass(map(float, LOGITNASH_GAMES[name]))
        result = traced_result(LOGITNASH_GAMES[name], element(4), settings)
        # The cause depends on whether rounding breaks the symmetry: an exact branch point factors singular, while one
        # that rounding unfolds by about eps (battle_of_the_sexes, rcond ≥ 1e-5) is near-singular and exhausts. A step
        # the turn test refuses at the branch point halves until its predictor cannot move the point (`unresolved`), or an
        # unmoved point stops the trace (run 41375663203). PathFollowing declares its stalls as reasons (`declared_stall`).
        if is_error(result)
            failure = unwrap_error(result)
            @test failure.cause in (GameTracing.Singular, GameTracing.PrecisionExhausted) ||
                (failure.cause == GameTracing.TraceStuck && declared_stall(failure, settings))
            # The failure comes at the reference's pitchfork. Measured, the furthest lay 8.7e-4 of its precision before it
            # (BifurcationKit's bordered tangent on battle of the sexes); a window of 1e-2 catches an early stall.
            pitchfork = only(branch_reference(reference_pass, length(LOGITNASH_GAMES[name]), 4).crossings)
            stall = failure.context.precision
            @test abs(stall - pitchfork) <= pitchfork / 100
            # The last accepted point solves the map and lies on the principal branch.
            logits = pass_logits(failure, settings)
            @test residual_norm(pass, logits, stall) < 1.0f-4
            reference = branch_reference(reference_pass, length(LOGITNASH_GAMES[name]), stall)
            @test reference.reached
            @test agreeing(first_choices(logits), reference.first_choices)
        else
            @test residual_norm(pass, unwrap(result).logits, element(4)) < 1.0f-4
        end
    end
end

"""Battle of the sexes with `scale` times standard normal payoffs from `rng` added: a pitchfork unfolded at random."""
function unfolded_pitchfork(rng::AbstractRNG, scale::Real)::Tuple
    return map(table -> table .+ scale .* randn(rng, 2, 2), map(float, LOGITNASH_GAMES.battle_of_the_sexes))
end

"""The unfolded-pitchfork testset's draws from `rng`: three at each scale from 1e-4 to 1e-1, in that order."""
function unfolded_pitchforks(rng::AbstractRNG)::Vector{Tuple}
    return [unfolded_pitchfork(rng, scale) for scale in inv.(float.((10^4, 10^3, 10^2, 10))) for _ in 1:3]
end

@testset "a pitchfork unfolded by a payoff perturbation: the trace turns with the principal branch ($engine)" for (engine, settings) in default_settings()
    # Measured over 40 draws per scale against the arclength reference: the secant tangent followed the principal branch
    # 0/40 at 1e-4, 3/40 at 1e-3, 34/40 at 1e-2 and 39/40 at 1e-1; the bordered tangent 40/40 at every scale.
    # PathFollowing, over 300 draws per scale in Float32 and Float64 alike: 300/300 at 1e-4, 1e-3 and 1e-2, and 297/300
    # at 1e-1 until its corrector's reach refusal; with it, 300/300 at every scale.
    for payoffs in unfolded_pitchforks(fresh_rng())
        cross_checked!(payoffs, 4, settings)
    end
end

# Unfolded pitchforks where PathFollowing's corrector jumped to the other component of the branch: draw 12 of a CI run's
# seed, and the 1e-1 draws of seeds 84, 185 and 260.
const JUMPED_PITCHFORKS = [
    unfolded_pitchforks(Xoshiro(0x0035b4b3c0a0f1ba))[12]; [unfolded_pitchfork(Xoshiro(seed), inv(float(10))) for seed in (84, 185, 260)]
]

@testset "PathFollowing keeps the unfolded pitchforks' principal branch where it jumped; without its reach refusal it jumps ($element)" for element in (Float32, widen(Float32))
    for payoffs in JUMPED_PITCHFORKS
        reference = branch_reference(NormalFormPass(map(float, payoffs)), 2, 4)
        @test reference.reached
        gains = NormalFormGains(map(table -> element.(table), payoffs))
        for (overrides, kept) in (((;), true), ((; reach = Inf32), false))
            landed = only(unwrap(logit_solve_lambda(gains, element(4); settings = PathFollowingSettings{element}(; overrides...))))
            @test agreeing(first_choices(-landed.logits), reference.first_choices) == kept
        end
    end
end

@testset "a pitchfork unfolded by 1e-4: the reference, the default trace and every arm turn onto the (1, 1) arm" begin
    precision = logitnash_equilibrium(UNFOLDED_PITCHFORK_GAME, 4).precision
    pass = NormalFormPass(map(float, UNFOLDED_PITCHFORK_GAME))
    reference = branch_reference(pass, 2, precision)
    @test reference.reached
    @test isempty(reference.crossings)
    # Oracle: Gambit 16.7's logit branch and a 50-digit arclength trace both end at the pure equilibrium (1, 1).
    arm_root = level_root(pass, [float(15), float(10)], precision)
    @test all(>(99 // 100), reference.first_choices)
    @test maximum(abs, reference.logits .- arm_root) < inv(float(10^6))
    oracle = gambit_branch(UNFOLDED_PITCHFORK_GAME, precision)
    @test agreeing(reference.first_choices, oracle.first_choices)
    # At Gambit's precision the trace keeps the 1e-4 rule: its accepted corrections all lie within it.
    double_settings = BifurcationKitSettings{typeof(float(1))}()
    # A failed double trace throws here, an error of this testset.
    doubled = unwrap(traced_result(UNFOLDED_PITCHFORK_GAME, precision * float(1), double_settings))
    double_logits = doubled.logits
    @test doubled.logit_accuracy <= settle_bound(double_settings.corrector, double_logits)
    @test agreeing(first_choices(double_logits), oracle.first_choices)
    @test settled_endpoint(UNFOLDED_PITCHFORK_GAME, double_logits, precision * float(1), double_settings)
    # Every Float32 arm follows the double-precision branch within both traces' accuracy.
    # PathFollowing follows it too, within its own accuracy and the double trace's.
    for (_, settings) in ["default" => BifurcationKitSettings{Float32}(); arm_variants(ARM_WEIGHTS); path_settings()]
        result = traced_result(UNFOLDED_PITCHFORK_GAME, GameTracing.element_type(settings)(precision), settings)
        @test !is_error(result)
        if !is_error(result)
            (; logits, logit_accuracy) = unwrap(result)
            @test logit_accuracy <= endpoint_bound(UNFOLDED_PITCHFORK_GAME, logits, settings)
            @test agreeing(first_choices(logits), oracle.first_choices)
            @test agreeing(first_choices(logits), first_choices(double_logits))
            @test maximum(abs, logits .- double_logits) <= logit_accuracy + doubled.logit_accuracy
            @test settled_endpoint(UNFOLDED_PITCHFORK_GAME, logits, precision, settings)
        end
        # Run to its regret-based end, the trace keeps the principal branch and meets Gambit at its own precision.
        element = GameTracing.element_type(settings)
        solved = logit_solve(NormalFormGains(map(table -> element.(table), UNFOLDED_PITCHFORK_GAME)); maxregret = 1.0f-4, settings)
        @test !is_error(solved)
        if !is_error(solved)
            checked = unwrap(solved)
            # The branch ends near the pure equilibrium (1, 1); the middle arm it passes ends mixed.
            @test all(strategy -> first(strategy) > 99 // 100, checked.profile)
            regret_oracle = gambit_strategies(UNFOLDED_PITCHFORK_GAME, checked.lam)
            @test largest_gap(checked.profile, regret_oracle.strategies) < profile_bound(checked) + GAMBIT_PROBABILITY_TOLERANCE
        end
    end
    # The log term at state weight 0.01, below `ARM_WEIGHTS`, exhausted near λ 0.808 until the dot's precision weight
    # took its root form, a change of rounding alone. It now reaches the (1, 1) arm.
    edge = traced_result(UNFOLDED_PITCHFORK_GAME, precision, arm_settings(ARMS.fisher_log_precision_reach, 0.01f0))
    @test !is_error(edge)
    if !is_error(edge)
        @test agreeing(first_choices(unwrap(edge).logits), oracle.first_choices)
        @test settled_endpoint(UNFOLDED_PITCHFORK_GAME, unwrap(edge).logits, precision, arm_settings(ARMS.fisher_log_precision_reach, 0.01f0))
    end
end

@testset "a step past the target precision lands where the branch first reaches it ($engine)" for (engine, settings) in default_settings()
    # Traced at the requested precisions themselves, both short of the fold, so the reference must reach them.
    for precision in (3.0f0, 3.1f0)
        result = traced_result(OVERSHOT_GAME, GameTracing.element_type(settings)(precision), settings)
        @test !is_error(result)
        reference = branch_reference(NormalFormPass(map(float, OVERSHOT_GAME)), length(OVERSHOT_GAME), precision)
        @test reference.reached
        @test agreeing(first_choices(unwrap(result).logits), reference.first_choices)
        @test agreeing(first_choices(unwrap(result).logits), gambit_branch(OVERSHOT_GAME, precision).first_choices)
        @test settled_endpoint(OVERSHOT_GAME, unwrap(result).logits, precision, settings)
    end
end

@testset "where Gambit's default steps leave the branch, the trace meets Gambit at fine steps ($engine)" for (engine, settings) in default_settings()
    # The oracle is Gambit at a 1e-3 first step that never grows; its default steps end elsewhere (`GAMBIT_LEAVES_GAME`).
    oracle = gambit_branch(GAMBIT_LEAVES_GAME, 3.0f0; first_step = 1 // 1000, max_accel = 1)
    traced = traced_result(GAMBIT_LEAVES_GAME, GameTracing.element_type(settings)(oracle.precision), settings)
    @test !is_error(traced)
    @test agreeing(first_choices(unwrap(traced).logits), oracle.first_choices)
    @test settled_endpoint(GAMBIT_LEAVES_GAME, unwrap(traced).logits, oracle.precision, settings)
end

@testset "random games with two actions each: the trace follows the principal branch ($engine)" for (engine, settings) in default_settings()
    rng = fresh_rng()
    skipped = 0
    for _ in 1:20
        players = rand(rng, 2:4)
        payoffs = ntuple(_ -> 3 .* randn(rng, ntuple(_ -> 2, players)), players)
        skipped += cross_checked!(payoffs, 3, settings)
    end
    @info "Gambit comparisons skipped where its branch folds before the target" skipped
end

@testset "a branch folds before the target only where its precisions decrease before first reaching it" begin
    @test !folds_before(Float32[0, 1, 2, 3.5f0], 3)
    @test folds_before(Float32[0, 1, 2.9f0, 2.5f0, 3.5f0], 3)
    @test !folds_before(Float32[0, 1, 3.5f0, 3.2f0, 4], 3)
end

@testset "on a game whose principal branch folds just past the target, the trace meets the reference ($engine)" for (engine, settings) in default_settings()
    precision = logitnash_equilibrium(FOLDED_GAMBIT_GAME, 3).precision
    reference = branch_reference(NormalFormPass(map(float, FOLDED_GAMBIT_GAME)), length(FOLDED_GAMBIT_GAME), precision)
    @test reference.reached
    traced = first_choices(unwrap(traced_result(FOLDED_GAMBIT_GAME, GameTracing.element_type(settings)(precision), settings)).logits)
    @test agreeing(traced, reference.first_choices)
    # Whether Gambit's comparison runs here depends on pygambit's step sizes, so only the checks it runs are asserted.
    cross_checked!(FOLDED_GAMBIT_GAME, 3, settings)
end
