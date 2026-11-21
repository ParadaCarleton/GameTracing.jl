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
    return failure.cause.reason in (GameTracing.UnmovedPoint, GameTracing.UnresolvedStep)
end

function declared_stall(failure::TraceFailure, ::PathFollowingSettings)::Bool
    return failure.cause.reason in (GameTracing.UnresolvedStep, GameTracing.StepRefusals)
end

"""The last accepted logits of a failed trace of `settings`' engine, in `NormalFormPass`' logits."""
pass_logits(failure::TraceFailure, ::BifurcationKitSettings)::Vector = collect(failure.stop.logits)
pass_logits(failure::TraceFailure, ::PathFollowingSettings)::Vector = -collect(failure.stop.logits)

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
    # Seed 0x5520d0f04c24c404 drew the worst game seen: residual 9.6e-8 and doubled gap 6.3e-7. Over 40,000 draws,
    # Gambit's residual reached 9.5e-8 and the gap 9.8e-8, apart from one branch that folds before its target.
    for rng in (Xoshiro(0x5520d0f04c24c404), fresh_rng()), _ in 1:5
        players = rand(rng, 2:3)
        payoffs = ntuple(_ -> 3 .* randn(rng, ntuple(_ -> 2, players)), players)
        precision = 1 + 2 * rand(rng)
        oracle = gambit_branch(payoffs, precision)
        @test abs(oracle.precision - precision) < inv(float(10^6))
        crossing = log.(oracle.first_choices) .- log1p.(-oracle.first_choices)
        response = evaluated(NormalFormPass(payoffs), crossing, oracle.precision).response
        @test maximum(abs, inv.(1 .+ exp.(-response)) .- oracle.first_choices) < inv(float(10^6))
        # Gambit does not normalize payoffs: doubling them doubles λ. A branch that folds first may land on a later crossing.
        doubled_payoffs = map(table -> 2 .* table, payoffs)
        if !gambit_folds_before(payoffs, precision) && !gambit_folds_before(doubled_payoffs, precision / 2)
            doubled = gambit_branch(doubled_payoffs, precision / 2)
            @test maximum(abs, doubled.first_choices .- oracle.first_choices) < inv(float(10^5))
        end
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
            @test failed_as(failure, TraceCause.Singular, TraceCause.PrecisionExhausted) ||
                (failed_as(failure, TraceCause.TraceStuck) && declared_stall(failure, settings))
            # The failure comes at the reference's pitchfork. Measured, the furthest lay 8.7e-4 of its precision before it
            # (BifurcationKit's bordered tangent on battle of the sexes); a window of 1e-2 catches an early stall.
            pitchfork = only(branch_reference(reference_pass, length(LOGITNASH_GAMES[name]), 4).crossings)
            stall = failure.stop.precision
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

"""The unfolded-pitchfork testset's draws from `rng`, as `scale => payoffs`: three at each scale from 1e-4 to 1e-1, in that order."""
function unfolded_pitchforks(rng::AbstractRNG)::Vector{<:Pair{<:Real, <:Tuple}}
    return [scale => unfolded_pitchfork(rng, scale) for scale in inv.(float.((10^4, 10^3, 10^2, 10))) for _ in 1:3]
end

"""
Whether `logits` at `precision` lie within 1e-3 of `payoffs`' principal branch (`branch_reference`). On
`UNFOLDED_PITCHFORK_GAME`, each measured Float32 stall's last accepted point lay within 3.5e-4 of it, the only root at λ 0.8079.
"""
function on_principal_branch(payoffs::Tuple, logits::AbstractVector{E}, precision::Real)::Bool where {E <: Real}
    reference = branch_reference(NormalFormPass(map(float, payoffs)), 2, precision)
    return reference.reached && maximum(abs, logits .- reference.logits) < 1.0f-3
end

"""
Whether a `PrecisionExhausted` failure's `stalled` point on `payoffs` (`stall_point`) is the stall near the pitchfork:
its failed iterate's `lambda` and its last accepted `precision` both in (0.8, 0.81), and its last accepted `logits` on
the principal branch.
"""
function pitchfork_stall(payoffs::Tuple, stalled::NamedTuple)::Bool
    near = 0.8f0 < stalled.lambda < 0.81f0 && 0.8f0 < stalled.precision < 0.81f0
    return near && on_principal_branch(payoffs, stalled.logits, stalled.precision)
end

"""`(; lambda, precision, logits)` of a `PrecisionExhausted` `failure`: its failed iterate's precision, and its stop's."""
function stall_point(failure::TraceFailure)::NamedTuple
    return (; lambda = failure.cause.stall.precision, failure.stop.precision, failure.stop.logits)
end

"""
Whether BifurcationKit's `failure` on `payoffs`, drawn at noise `scale` and traced to `target` under `settings`, is the
known Float32 stall (issue #132): at noise 1e-4, near the pitchfork (`pitchfork_stall`), and at rounding noise
(`exhausted_by_rounding`). Over 3000 draws, 7 stalled at 1e-4 and none at 1e-3; each corrected by at most 12% of its bound.
"""
function known_float32_stall(failure::TraceFailure, payoffs::Tuple, scale::Real, settings::BifurcationKitSettings{E}, target::E)::Bool where {E <: Real}
    near = scale <= inv(float(10^4)) && failed_as(failure, TraceCause.PrecisionExhausted) && pitchfork_stall(payoffs, stall_point(failure))
    return near && exhausted_by_rounding(NormalFormPass(map(table -> E.(table), payoffs)), failure, settings, target)
end

"""
The settings that cross-check `payoffs`, drawn at noise `scale`: BifurcationKit's Float64 settings where its Float32
trace ends in the known stall (`known_float32_stall`), else `settings`. Float64 reached all 3000 draws at 1e-4.
"""
function pitchfork_settings(payoffs::Tuple, scale::Real, precision::Real, settings::EngineSettings)::EngineSettings
    settings isa BifurcationKitSettings{Float32} || return settings
    target = logitnash_equilibrium(payoffs, precision).precision
    result = traced_result(payoffs, target, settings)
    stalled = is_error(result) && known_float32_stall(unwrap_error(result), payoffs, scale, settings, target)
    return stalled ? BifurcationKitSettings{widen(Float32)}() : settings
end

@testset "a pitchfork unfolded by a payoff perturbation: the trace turns with the principal branch ($engine)" for (engine, settings) in default_settings()
    # Measured over 40 draws per scale against the arclength reference: the secant tangent followed the principal branch
    # 0/40 at 1e-4, 3/40 at 1e-3, 34/40 at 1e-2 and 39/40 at 1e-1; the bordered tangent 40/40 at every scale.
    # PathFollowing, over 300 draws per scale in Float32 and Float64 alike: 300/300 at 1e-4, 1e-3 and 1e-2, and 297/300
    # at 1e-1 until its corrector's reach refusal; with it, 300/300 at every scale.
    checks = [payoffs => pitchfork_settings(payoffs, scale, 4, settings) for (scale, payoffs) in unfolded_pitchforks(fresh_rng())]
    for (payoffs, checked_with) in checks
        cross_checked!(payoffs, 4, checked_with)
    end
    # A fallback swaps the settings. 7 of 3000 draws at 1e-4 stalled, so two of a run's three stall with chance 2e-5.
    @test count(check -> last(check) !== settings, checks) <= 1
end

# Draws 508 and 552 of seed 20261122 at noise 1e-4: two of the 7 whose BifurcationKit Float32 trace stalled near the
# pitchfork on a Meteor Lake and on winwsl.
const FLOAT32_STALLED_PITCHFORKS = let rng = Xoshiro(20261122)
    draws = [unfolded_pitchfork(rng, inv(float(10^4))) for _ in 1:552]
    (draws[508], draws[552])
end

@testset "unfolded pitchforks where BifurcationKit's Float32 trace stalled: Float64 cross-checks them, the fallback only at 1e-4" begin
    float32 = BifurcationKitSettings{Float32}()
    for payoffs in FLOAT32_STALLED_PITCHFORKS
        cross_checked!(payoffs, 4, BifurcationKitSettings{widen(Float32)}())
        # On a CPU whose Float32 trace reaches the target, neither scale falls back.
        stalled = is_error(traced_result(payoffs, logitnash_equilibrium(payoffs, 4).precision, float32))
        @test (pitchfork_settings(payoffs, inv(float(10^4)), 4, float32) isa BifurcationKitSettings{widen(Float32)}) == stalled
        @test pitchfork_settings(payoffs, inv(float(10^3)), 4, float32) isa BifurcationKitSettings{Float32}
    end
end

# The Float32 stall of the first `FLOAT32_STALLED_PITCHFORKS` draw on winwsl, traced to 4.9532833: the fields the checks
# read (`recorded_stall`). Its correction is 7.6% of its rounding bound, 2.4e-3; its residual 3.5% of its bound.
const RECORDED_FLOAT32_STALL = (;
    lambda = 0.80760336f0, precision = 0.8076254f0, logits = [0.2016669f0, -0.2008812f0], correction = 0.00018011034f0,
    stalled_iterate = [0.24992676f0, -0.24851432f0, 0.80760336f0], stalled_pin = 0.80760336f0, stalled_arclength = 9.369129f-8,
    gaps = [0.24970351f0, -0.24873066f0], gap_tangent = [1.3947775f0, 1.9262395f0], tangent_precision = 1.7230258f0,
)

"""A BifurcationKit `PrecisionExhausted` failure mid-trace that holds the fields of `recorded`, as `RECORDED_FLOAT32_STALL` names them."""
function recorded_stall(recorded::NamedTuple)::TraceFailure
    stall = StalledCorrection(recorded.lambda, recorded.correction, recorded.stalled_iterate, recorded.stalled_pin, recorded.stalled_arclength)
    stop = TraceStop(recorded.logits, recorded.gaps, recorded.precision, 0, 0.0f0)
    logit_tangent = recorded.precision .* recorded.gap_tangent .+ recorded.tangent_precision .* recorded.gaps
    engine = EngineRecord.BifurcationKitTrace(0.0f0, 0.0f0, recorded.precision, logit_tangent, recorded.gap_tangent, recorded.tangent_precision, 0.0f0, "")
    return TraceFailure(TraceCause.PrecisionExhausted(GameTracing.MidTrace, stall), stop, engine, (;))
end

@testset "the fallback takes a recorded stall at rounding noise, at 1e-4 only; a moved iterate or a large correction fails" begin
    payoffs, float32 = first(FLOAT32_STALLED_PITCHFORKS), BifurcationKitSettings{Float32}()
    target = logitnash_equilibrium(payoffs, 4).precision
    stall = recorded_stall(RECORDED_FLOAT32_STALL)
    @test known_float32_stall(stall, payoffs, inv(float(10^4)), float32, target)
    @test !known_float32_stall(stall, payoffs, inv(float(10^3)), float32, target)
    stuck = TraceFailure(TraceCause.TraceStuck{Float32}(GameTracing.StoppedShort), stall.stop, stall.engine, stall.context)
    @test !known_float32_stall(stuck, payoffs, inv(float(10^4)), float32, target)
    # An iterate 0.01 off the step's root, a correction of 0.1 (40 times the bound) or NaN, or logits off the branch.
    changes = (
        (; stalled_iterate = RECORDED_FLOAT32_STALL.stalled_iterate .+ Float32[0.01f0, 0, 0]),
        (; correction = 0.1f0), (; correction = NaN32), (; logits = RECORDED_FLOAT32_STALL.logits .+ 0.01f0),
    )
    for change in changes
        failure = recorded_stall(merge(RECORDED_FLOAT32_STALL, change))
        @test !known_float32_stall(failure, payoffs, inv(float(10^4)), float32, target)
    end
end

"""
The step control where the jump tests' games below left their branch: growth 2 and the summed metric at state weight 1,
at the previous targets (`previous_targets`), with the split check at trigger 1. The default growth of 1.1 keeps those
games' branch without any check.
"""
function growth_two(element::Type)::NamedTuple
    tolerance = PathFollowingSettings{element}().split_check.tolerance
    return (;
        previous_targets(element)..., max_step_growth = 2, step_metric = TraceGeometry{element}(; distance = SummedDistance, state_weight = 1),
        split_check = GameTracing.PathFollowing.HalfStepCheck{element}(; trigger = 1, tolerance),
    )
end

# Unfolded pitchforks where PathFollowing's corrector once jumped to the other component of the branch at growth 2: draw
# 12 of a CI run's seed, and the 1e-1 draws of seeds 84, 185 and 260. Under the profile's Fisher–Rao distance and the
# Euclidean logit move, none jumps at `growth_two`, even without the reach refusal and the split check.
const JUMPED_PITCHFORKS = [last(unfolded_pitchforks(Xoshiro(0x0035b4b3c0a0f1ba))[12]); [unfolded_pitchfork(Xoshiro(seed), inv(float(10))) for seed in (84, 185, 260)]]

@testset "PathFollowing keeps the unfolded pitchforks' principal branch where it once jumped, with any of its checks off ($element)" for element in (Float32, widen(Float32))
    for payoffs in JUMPED_PITCHFORKS
        reference = branch_reference(NormalFormPass(map(float, payoffs)), 2, 4)
        @test reference.reached
        gains = NormalFormGains(map(table -> element.(table), payoffs))
        checks = ((;), (; split_check = nothing), (; reach = Inf32), (; reach = Inf32, split_check = nothing))
        # The default step control keeps each branch with any of the checks off.
        arms = [((; overrides...), true) for overrides in checks]
        append!(arms, [((; growth_two(element)..., overrides...), true) for overrides in checks])
        for (overrides, kept) in arms
            landed = only(unwrap(logit_solve_lambda(gains, element(4); settings = PathFollowingSettings{element}(; overrides...))))
            @test agreeing(first_choices(-landed.logits), reference.first_choices) == kept
        end
    end
end

"""Battle of the sexes with `scale` times one standard normal draw from `Xoshiro(seed)` added to both players' payoffs."""
function same_noise_pitchfork(seed::Integer, scale::Real)::Tuple
    return map(table -> table .+ scale .* randn(Xoshiro(seed), 2, 2), map(float, LOGITNASH_GAMES.battle_of_the_sexes))
end

# This pitchfork once jumped at growth 2 without the reach refusal and the split check; under the profile's Fisher–Rao
# distance and the Euclidean logit move it keeps its branch there too.
@testset "PathFollowing keeps an unfolded pitchfork's principal branch without the reach refusal, with or without its split check ($element)" for element in (Float32, widen(Float32))
    payoffs = same_noise_pitchfork(21, inv(float(10)))
    reference = branch_reference(NormalFormPass(map(float, payoffs)), 2, 4)
    gains = NormalFormGains(map(table -> element.(table), payoffs))
    arms = (
        ((; reach = Inf32), true), ((; reach = Inf32, split_check = nothing), true),
        ((; growth_two(element)..., reach = Inf32), true), ((; growth_two(element)..., reach = Inf32, split_check = nothing), true),
    )
    for (overrides, kept) in arms
        landed = only(unwrap(logit_solve_lambda(gains, element(4); settings = PathFollowingSettings{element}(; overrides...))))
        @test agreeing(first_choices(-landed.logits), reference.first_choices) == kept
    end
end

# At `growth_two`, the second step keeps the branch, but its half steps end 1.25 Fisher–Rao from its point, so the split
# check refuses it.
@testset "PathFollowing keeps the branch where its split check refuses a step that kept it ($element)" for element in (Float32, widen(Float32))
    payoffs = same_noise_pitchfork(213, inv(float(10)))
    reference = branch_reference(NormalFormPass(map(float, payoffs)), 2, 4)
    gains = NormalFormGains(map(table -> element.(table), payoffs))
    arms = ((PathFollowingSettings{element}(; growth_two(element)...), 1), (PathFollowingSettings{element}(; growth_two(element)..., split_check = nothing), 0))
    for (settings, refusals) in arms
        tracer = started_tracer(gains, settings)
        tracer, _ = GameTracing.path_step!!(tracer, element(4), settings.max_steps, (;))
        _, second = GameTracing.path_step!!(tracer, element(4), settings.max_steps, (;))
        @test unwrap(second).refusals == refusals
        landed = only(unwrap(logit_solve_lambda(gains, element(4); settings)))
        @test agreeing(first_choices(-landed.logits), reference.first_choices)
    end
end

# The state weights of the pitchfork sweep.
const PITCHFORK_WEIGHTS = (0.003f0, 0.01f0, 0.03f0, 0.1f0, 0.3f0)


"""
Where a BifurcationKit trace of `UNFOLDED_PITCHFORK_GAME` to `precision` under `settings` ends: "arm" on `oracle`'s
(1, 1) arm and settled, "stall" for `PrecisionExhausted` near λ 0.808, or a description of any other end. The
unperturbed game's pitchfork lies at λ 0.80814. At one stall point (λ 0.80785) the gap Jacobian's condition number was
5600 and the last Float32 Newton correction 2.8e-4, above the logit tolerance. The step sequence decides whether a
trace stalls (issue #132).
"""
function pitchfork_outcome(precision::E, settings::BifurcationKitSettings{E}, oracle::NamedTuple)::String where {E <: Real}
    result = traced_result(UNFOLDED_PITCHFORK_GAME, precision, settings)
    if is_error(result)
        failure = unwrap_error(result)
        stalled = failed_as(failure, TraceCause.PrecisionExhausted) && pitchfork_stall(UNFOLDED_PITCHFORK_GAME, stall_point(failure))
        return stalled ? "stall" : "$(failure.cause) past λ $(failure.stop.precision)"
    end
    logits = unwrap(result).logits
    on_arm = agreeing(first_choices(logits), oracle.first_choices) && settled_endpoint(UNFOLDED_PITCHFORK_GAME, logits, precision, settings)
    return on_arm ? "arm" : "off the arm at logits $logits"
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
    # Every Float32 trace follows the double-precision branch within both traces' accuracy, or a BifurcationKit one
    # stalls near the pitchfork (issue #132). Measured: one stall of the 7 on CI's runner and on a Haswell, none on a
    # Meteor Lake; the cap is twice that.
    stalls = String[]
    @testset "$label" for (label, settings) in ["default" => BifurcationKitSettings{Float32}(); arm_variants(ARM_WEIGHTS); path_settings()]
        result = traced_result(UNFOLDED_PITCHFORK_GAME, GameTracing.element_type(settings)(precision), settings)
        if is_error(result)
            failure = unwrap_error(result)
            @test settings isa BifurcationKitSettings && failed_as(failure, TraceCause.PrecisionExhausted) && pitchfork_stall(UNFOLDED_PITCHFORK_GAME, stall_point(failure))
            push!(stalls, label)
        else
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
    @test length(stalls) <= 2
end

@testset "a pitchfork unfolded by 1e-4: Float64 traces settle on the (1, 1) arm; Float32 ones may stall, never leave it" begin
    precision = logitnash_equilibrium(UNFOLDED_PITCHFORK_GAME, 4).precision
    oracle = gambit_branch(UNFOLDED_PITCHFORK_GAME, precision)
    # A stall passes only near the pitchfork on the principal branch: not 0.01 off it, nor last accepted at λ 0.5.
    pass = NormalFormPass(map(float, UNFOLDED_PITCHFORK_GAME))
    at_stall = Float32.(level_root(pass, [0.2f0, -0.2f0], 0.8079f0))
    early = Float32.(level_root(pass, [0.2f0, -0.2f0], 0.5f0))
    measured = (; lambda = 0.8079f0, precision = 0.8079f0, logits = at_stall)
    @test pitchfork_stall(UNFOLDED_PITCHFORK_GAME, measured)
    @test !pitchfork_stall(UNFOLDED_PITCHFORK_GAME, merge(measured, (; logits = at_stall .+ 0.01f0)))
    @test !pitchfork_stall(UNFOLDED_PITCHFORK_GAME, merge(measured, (; precision = 0.5f0, logits = early)))
    for element in (Float32, widen(Float32))
        outcomes = [
            (name, element, weight) => pitchfork_outcome(element(precision), arm_settings(arm, element(weight)), oracle)
                for (name, arm) in pairs(ARMS) for weight in PITCHFORK_WEIGHTS
        ]
        allowed = element == Float32 ? ("arm", "stall") : ("arm",)
        @test filter(outcome -> last(outcome) ∉ allowed, outcomes) == []
        # Measured: the log term stalls at weights 0.003 and 0.01, 2 of the 10; the cap is twice that.
        @test count(outcome -> last(outcome) == "stall", outcomes) <= 4
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

"""The random-games testset's 20 draws from `rng`: two to four players, two actions each, `3 randn` payoffs."""
function two_action_games(rng::AbstractRNG)::Vector{Tuple}
    return map(1:20) do _
        players = rand(rng, 2:4)
        ntuple(_ -> 3 .* randn(rng, ntuple(_ -> 2, players)), players)
    end
end

@testset "random games with two actions each: the trace follows the principal branch ($engine)" for (engine, settings) in default_settings()
    skipped = sum(payoffs -> cross_checked!(payoffs, 3, settings), two_action_games(fresh_rng()))
    @info "Gambit comparisons skipped where its branch folds before the target" skipped
end

# Random games where PathFollowing's corrector jumped branches at growth 2 without its split check: draws of four suite
# runs' fresh seeds, two of a 200-seed sweep and four of a 1000-seed hunt. Each pairs with whether it still jumps at
# `growth_two` without the split check, under the profile's Fisher–Rao distance and the Euclidean logit move.
const JUMPED_TWO_ACTION_GAMES = [
    (two_action_games(Xoshiro(seed))[draw], jumps)
        for (seed, draw, jumps) in (
            (0x0b357a95eb385c17, 4, false), (0xe3d835fcb88261eb, 12, true), (0x8ee16446e3ac8acf, 13, false), (0x501a9b6a3e953473, 1, true),
            (0xb2cb58af5afc41b4, 12, false), (0xa656d4e67e70d45e, 10, true),
            (0x64458f7352e8bfd9, 16, true), (0xb282f30aeb0a2397, 19, true), (0xe3bd68fccdf58be2, 20, true), (0xf61c7e3f6e4606ee, 7, false),
        )
]

@testset "PathFollowing keeps the random games' principal branch where it jumped; at growth 2 without its split check six still jump ($element)" for element in (Float32, widen(Float32))
    @test count(last, JUMPED_TWO_ACTION_GAMES) == 6
    for (payoffs, jumps) in JUMPED_TWO_ACTION_GAMES
        precision = logitnash_equilibrium(payoffs, 3).precision
        reference = branch_reference(NormalFormPass(map(float, payoffs)), length(payoffs), precision)
        @test reference.reached
        gains = NormalFormGains(map(table -> element.(table), payoffs))
        arms = (((;), true), ((; split_check = nothing), true), (growth_two(element), true), ((; growth_two(element)..., split_check = nothing), !jumps))
        for (overrides, kept) in arms
            landed = only(unwrap(logit_solve_lambda(gains, element(precision); settings = PathFollowingSettings{element}(; overrides...))))
            @test agreeing(first_choices(-landed.logits), reference.first_choices) == kept
        end
    end
end

# Games where PathFollowing jumped branches at its previous targets without the split check: draws of a hunt over 1000
# fresh seeds of 20 random two-action games, and of the same-noise pitchforks at 300 seeds per scale. Each carries its
# fixed precision, if any, and whether it still jumps there under the profile's Fisher–Rao distance and Euclidean logits.
const PREVIOUS_TARGET_JUMPS = [
    [(two_action_games(Xoshiro(seed))[draw], nothing, jumps) for (seed, draw, jumps) in ((0x0db119e74ded1c3e, 2, true), (0x0fd92e270e7a0a26, 11, false), (0x2981f81ebefd5d7e, 3, true), (0x585f8d66c83d6b7a, 15, false))];
    [(same_noise_pitchfork(seed, scale), 4, false) for (seed, scale) in ((277, inv(float(100))), (218, inv(float(10))), (223, inv(float(10))))]
]

@testset "PathFollowing keeps the principal branch where its previous targets jumped; there without its split check two still jump ($element)" for element in (Float32, widen(Float32))
    @test count(last, PREVIOUS_TARGET_JUMPS) == 2
    for (payoffs, fixed, jumps) in PREVIOUS_TARGET_JUMPS
        precision = fixed === nothing ? logitnash_equilibrium(payoffs, 3).precision : fixed
        reference = branch_reference(NormalFormPass(map(float, payoffs)), length(payoffs), precision)
        @test reference.reached
        gains = NormalFormGains(map(table -> element.(table), payoffs))
        # The default targets keep each branch without the split check too.
        arms = ((;), (; split_check = nothing), previous_targets(element), (; previous_targets(element)..., split_check = nothing))
        for (overrides, kept) in zip(arms, (true, true, true, !jumps))
            settings = PathFollowingSettings{element}(; overrides...)
            landed = only(unwrap(logit_solve_lambda(gains, element(precision); settings)))
            @test agreeing(first_choices(-landed.logits), reference.first_choices) == kept
        end
    end
end

@testset "a branch folds before the target only where its precisions decrease before first reaching it" begin
    @test !folds_before(Float32[0, 1, 2, 3.5f0], 3)
    @test folds_before(Float32[0, 1, 2.9f0, 2.5f0, 3.5f0], 3)
    @test !folds_before(Float32[0, 1, 3.5f0, 3.2f0, 4], 3)
end

@testset "Gambit's branch points stop at the first at or past the target, or where the branch settles" begin
    steps = first(GAMBIT_STEP_LADDER)
    lambdas = gambit_branch_lambdas(FOLD_PAST_ONE_GAME, steps, 4)
    @test last(lambdas) >= 4 > maximum(lambdas[1:(end - 1)])
    # Gambit's doubles, which no rounding lifts onto a Float32 target.
    @test eltype(lambdas) == widen(Float32)
    # Matching pennies' uniform profile has no regret, so Gambit ends its branch at the start.
    @test gambit_branch_lambdas(LOGITNASH_GAMES.matching_pennies, steps, 4) == [0]
    # The prisoners' dilemma settles to regret 1e-12 far below precision 1e6, past where regret 1e-3 ends.
    settled = gambit_branch_lambdas(LOGITNASH_GAMES.prisoners_dilemma, steps, 10^6)
    @test settled == gambit_regret_lambdas(LOGITNASH_GAMES.prisoners_dilemma, steps, inv(float(10^12)))
    @test last(gambit_regret_lambdas(LOGITNASH_GAMES.prisoners_dilemma, steps, inv(float(10^3)))) < last(settled) < 10^6
    gambit_strategies(FOLD_PAST_ONE_GAME, 4.0f0)
    # 0.44 s on macdev; the branch to regret 1e-12 took 172 s.
    @test (@elapsed gambit_strategies(SLOW_REGRET_GAME, 4.0f0)) < 20
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
