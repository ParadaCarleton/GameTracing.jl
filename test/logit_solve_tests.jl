using ErrorTypes: Ok, Result, is_error, unwrap, unwrap_error
using LinearAlgebra: Diagonal
using PathFollowing: PathFollowing
using Random: AbstractRNG, Xoshiro

const ROCK_PAPER_SCISSORS = ([0 -1 1; 1 0 -1; -1 1 0], [0 1 -1; -1 0 1; 1 -1 0])

"""Each engine's default settings in `E`, by name: BifurcationKit's as `logit_solve` builds them, and PathFollowing's."""
function engine_settings(::Type{E})::Vector{Pair{String, EngineSettings{E}}} where {E <: Real}
    return ["BifurcationKit" => GameTracing.gambit_settings(E, GameTracing.GAMBIT_MAX_ACCEL), "PathFollowing" => PathFollowingSettings{E}()]
end

@testset "logit_solve_lambda agrees with Gambit's logit_solve_lambda at each precision ($engine)" for (engine, settings) in engine_settings(Float32)
    rng = fresh_rng()
    random_games = [ntuple(_ -> 3 .* randn(rng, counts), length(counts)) for counts in ((3, 3), (2, 3, 2), (4, 2))]
    games = [LOGITNASH_GAMES.prisoners_dilemma, LOGITNASH_GAMES.matching_pennies, ROCK_PAPER_SCISSORS, FOLD_PAST_ONE_GAME, random_games...]
    for payoffs in games
        gains = NormalFormGains(map(table -> Float32.(table), payoffs))
        # Out of order, so a single trace must land on each precision in rising order and return them as given.
        profiles = unwrap(logit_solve_lambda(gains, [4, 1]; settings))
        @test [profile.lam for profile in profiles] == Float32[4, 1]
        for profile in profiles
            oracle = gambit_strategies(payoffs, profile.lam)
            @test largest_gap(profile.profile, oracle.strategies) < profile_bound(profile) + GAMBIT_PROBABILITY_TOLERANCE
        end
    end
    @test length(unwrap(logit_solve_lambda(NormalFormGains(map(table -> Float32.(table), ROCK_PAPER_SCISSORS)), 2; settings))) == 1
end

@testset "Gambit's default steps round the fold of FOLD_PAST_ONE_GAME onto a later crossing; a finer rung shows the first" begin
    @test gambit_folds_before(FOLD_PAST_ONE_GAME, 1.0f0)
    @test first_crossing_steps(FOLD_PAST_ONE_GAME, 1.0f0) == GAMBIT_STEP_LADDER[2]
    # The fold peaks below precision 4, so every rung folds before it and the finest stands.
    @test first_crossing_steps(FOLD_PAST_ONE_GAME, 4.0f0) == last(GAMBIT_STEP_LADDER)
    later = gambit_branch(FOLD_PAST_ONE_GAME, 1.0f0).first_choices
    @test maximum(abs, later .- first.(gambit_strategies(FOLD_PAST_ONE_GAME, 1.0f0).strategies)) > 1.0f-1
end

@testset "logit_solve ends at the Nash equilibrium of games whose equilibrium is known ($engine)" for (engine, settings) in engine_settings(Float32)
    uniform = fill(1.0f0 / 3, 3)
    for (payoffs, equilibrium) in (
            (LOGITNASH_GAMES.prisoners_dilemma, (Float32[0, 1], Float32[0, 1])),
            (LOGITNASH_GAMES.matching_pennies, (fill(0.5f0, 2), fill(0.5f0, 2))),
            (ROCK_PAPER_SCISSORS, (uniform, uniform)),
        )
        gains = NormalFormGains(map(table -> Float32.(table), payoffs))
        result = logit_solve(gains; settings)
        @test !is_error(result)
        solved = unwrap(result)
        # Regret r of the payoff range bounds a dominated strategy's weight near r of the gap it loses.
        @test largest_gap(solved.profile, equilibrium) < 1.0f-2
        @test GameTracing.max_regret(gains, solved.logits) <= GameTracing.default_maxregret(Float32) * GameTracing.payoff_range(gains)
    end
end

# The seed whose branch games once stuck at precision 42426 with logits off the map: Gambit ends the (3, 3) game at a
# mixed equilibrium (λ ≈ 47) and the (2, 2, 2) game at the pure profile (2, 1, 2) (λ ≈ 1.5).
const STUCK_BRANCH_SEED = 0x126798e4d8bed9ec

# A fresh draw of a CI run: PathFollowing ends it at logits near ±300, where every strategy has settled but the logits
# sit about 1000 ulps off the map. Its points settle in strategies, PathFollowing's measure, and fail in logits.
const SATURATED_BRANCH_GAME = (
    Float32[1.9945446f0 -6.476125f0; -0.60958946f0 -0.019216493f0;;; 2.4169908f0 1.7390554f0; 2.7044353f0 -5.6631885f0],
    Float32[3.0360756f0 1.9217091f0; 0.7952187f0 -5.8968396f0;;; -2.338289f0 3.3105536f0; -2.648212f0 1.6191502f0],
    Float32[-1.675467f0 -0.60522515f0; 0.72147083f0 -0.13347721f0;;; -5.2404265f0 3.434299f0; 2.0519495f0 3.9997408f0],
)

# A fresh draw of a CI run: its branch folds at λ ≈ 2.7237 before it meets maxregret, and BifurcationKit lands a point on
# the peak. There `I - ∂T/∂z` is singular, so a Newton step at fixed λ grows past the settle bound (`branch_step` frees
# λ), and the λ-fixed checks fail: Gambit's `logit_solve_lambda` returns the first crossing of the end's λ.
const FOLD_PEAK_GAME = (
    Float32[-1.7090197f0 3.4388857f0 4.108635f0; 1.6385761f0 -2.7659307f0 5.6244125f0; -2.6847887f0 -0.8351383f0 -6.3075285f0],
    Float32[-0.11561966f0 3.8349252f0 4.1745343f0; 1.9479157f0 -2.412167f0 -1.7825432f0; -5.1971507f0 3.1131897f0 -5.5004435f0],
)

# A fresh draw of PR 125's CI run: main's Float32 BifurcationKit trace exhausted at the fold peak λ ≈ 26.26, a logit at -88.
const FOLD_EXHAUSTED_GAME = (
    Float32[-0.17346182f0 0.98846f0 -0.047029182f0; -0.15673546f0 1.7735703f0 3.5914543f0; 0.1646234f0 -4.6235147f0 -0.70840126f0],
    Float32[-2.4531052f0 1.945029f0 -6.307946f0; 0.8079069f0 -0.08894292f0 -2.0735276f0; 1.8209918f0 1.0540596f0 -3.7581315f0],
)

"""
Whether a branch point settles as its engine's corrector settles points, by a Newton step with the precision free
(`branch_step`): BifurcationKit's in logits, PathFollowing's in Gambit's norm (`CorrectorDistance`).
"""
settled_point(gains::NormalFormGains, profile::LogitQREProfile, ::BifurcationKitSettings)::Bool =
    settled_branch_logits(gains, profile.logits, profile.lam, BifurcationKitSettings{eltype(profile.logits)}())
settled_point(gains::NormalFormGains, profile::LogitQREProfile, settings::PathFollowingSettings)::Bool =
    settled_strategies(gains, profile.logits, profile.lam, settings.distance)

"""The random games of the branch testset, one per strategy count in `((3, 3), (2, 2, 2))`, drawn from `rng`."""
function branch_games(rng::AbstractRNG)::AbstractVector
    return [ntuple(_ -> 3 .* randn(rng, Float32, counts), length(counts)) for counts in ((3, 3), (2, 2, 2))]
end

# A probability this small at a Float32 stop marks a saturated strategy. The six new Float32 failures of the fold and
# maxregret sweeps and one CI draw each stopped with a probability of at most 2.3e-24.
const UNDERFLOWED_PROBABILITY = 1.0f-20

"""
Whether the Float32 failure `failure` on `payoffs` is a declared one: `PrecisionExhausted` or an unresolved step, where
some strategy's probability at the stop, in double precision, is below `UNDERFLOWED_PROBABILITY`. Gambit's norm counts
such a strategy's log probability, and its Newton step is ill-conditioned in Float32. About 1% of random games end so.
"""
function declared_float32_failure(failure::TraceFailure{Float32}, payoffs::Tuple)::Bool
    stuck = failed_as(failure, TraceCause.TraceStuck) && failure.cause.reason == GameTracing.UnresolvedStep
    wide = NormalFormGains(map(table -> widen(Float32).(table), payoffs))
    least = minimum(minimum, mixed_strategies(wide, widen(Float32).(failure.stop.logits)))
    return (stuck || failed_as(failure, TraceCause.PrecisionExhausted)) && least < UNDERFLOWED_PROBABILITY
end

"""`settings`' engine at its double-precision defaults."""
widened_settings(::BifurcationKitSettings)::EngineSettings = GameTracing.gambit_settings(widen(Float32), GameTracing.GAMBIT_MAX_ACCEL)
widened_settings(::PathFollowingSettings)::EngineSettings = PathFollowingSettings{widen(Float32)}()

"""
The settings that check `payoffs`: `settings`' engine in double precision where its Float32 branch ends in a declared
failure (`declared_float32_failure`), else `settings`.
"""
function branch_settings(payoffs::Tuple, settings::EngineSettings{Float32})::EngineSettings
    result = logit_solve_branch(NormalFormGains(payoffs); settings)
    declared = is_error(result) && declared_float32_failure(unwrap_error(result), payoffs)
    return declared ? widened_settings(settings) : settings
end

@testset "a declared Float32 failure is an unresolved step or exhausted precision at an underflowed strategy" begin
    # The stop of a CI fresh (3, 3) draw on PathFollowing: its third logit of -73 leaves a probability near 1e-32.
    payoffs = map(
        bits -> reinterpret.(Float32, bits),
        (
            UInt32[0xc087a3f5 0x3fb5fcc7 0xbf99202a; 0xc091a200 0xc018d55e 0xbff8c5b6; 0xc08694d8 0xc002fee8 0x402f0878],
            UInt32[0xbe680be8 0xbf72c650 0xc02a574d; 0xc070fe7f 0x4023e948 0x3fca5632; 0x407b4508 0x404aa4e4 0x407b12ea],
        ),
    )
    stop = TraceStop(Float32[-30.195833f0, 3.2393425f0, -73.32123f0, -9.163015f0], Float32[-0.3123064f0, 0.033503544f0, -0.7583394f0, -0.09477031f0], 96.68656f0, 57, 7.483035f-8)
    engine = EngineRecord.PathFollowingStep(PathFollowing.StepFailure.Unresolved{Float32}(PathFollowing.Diverging, 3))
    failure(cause, at) = TraceFailure(cause, at, engine, (;))
    @test declared_float32_failure(failure(TraceCause.TraceStuck{Float32}(GameTracing.UnresolvedStep), stop), payoffs)
    @test !declared_float32_failure(failure(TraceCause.TraceStuck{Float32}(GameTracing.StoppedShort), stop), payoffs)
    # The same cause at logits a tenth as large leaves every probability above 1e-4.
    near = TraceStop(stop.logits ./ 10, stop.gaps, stop.precision / 10, stop.steps, stop.step_length)
    @test !declared_float32_failure(failure(TraceCause.TraceStuck{Float32}(GameTracing.UnresolvedStep), near), payoffs)
end

@testset "logit_solve_branch: each point solves the logit map at its precision, and the last meets maxregret ($engine)" for (engine, settings) in engine_settings(Float32)
    stuck_games = branch_games(Xoshiro(STUCK_BRANCH_SEED))
    # The seed's first payoff, as the failing run logged it.
    @test first(first(first(stuck_games))) == -0.4899592f0
    # The fixed games pass in Float32. A fresh draw may end in a declared Float32 failure; Float64 then checks it.
    fixed = [payoffs => settings for payoffs in (stuck_games..., SATURATED_BRANCH_GAME, FOLD_PEAK_GAME, FOLD_EXHAUSTED_GAME)]
    checks = [fixed; [payoffs => branch_settings(payoffs, settings) for payoffs in branch_games(fresh_rng())]]
    fallbacks = count(check -> last(check) !== settings, checks)
    @info "declared Float32 failures checked in Float64" engine fallbacks
    # About 1% of random games fail so, so two of a run's two fresh draws fail with chance 1e-4.
    @test fallbacks <= 1
    for (payoffs, checked_with) in checks
        element = GameTracing.element_type(checked_with)
        gains = NormalFormGains(map(table -> element.(table), payoffs))
        result = logit_solve_branch(gains; settings = checked_with)
        @test !is_error(result)
        if is_error(result)
            @error "logit_solve_branch failed" payoffs = repr(payoffs) payoff_bits = map(table -> reinterpret.(UInt32, table), payoffs) failure = unwrap_error(result)
            continue
        end
        branch = unwrap(result)
        @test all(profile -> profile.lam > 0, branch)
        @test all(profile -> all(isfinite, profile.logits), branch)
        @test GameTracing.max_regret(gains, last(branch).logits) <= GameTracing.default_maxregret(element) * GameTracing.payoff_range(gains)
        solved = unwrap(logit_solve(gains; settings = checked_with))
        @test solved.lam == last(branch).lam
        for profile in branch
            @test settled_point(gains, profile, checked_with)
        end
        # A fold leaves the λ-fixed checks below without a unique answer (`FOLD_PEAK_GAME`), even past its peak.
        folded = !issorted([profile.lam for profile in branch])
        # A fresh draw may fold too; the fixed game must.
        @test folded || payoffs !== FOLD_PEAK_GAME
        if folded
            continue
        end
        # The last point is on Gambit's branch, and logit_solve_lambda finds it at its precision.
        @test largest_gap(last(branch).profile, gambit_strategies(payoffs, last(branch).lam).strategies) <
            profile_bound(last(branch)) + GAMBIT_PROBABILITY_TOLERANCE
        again = only(unwrap(logit_solve_lambda(gains, solved.lam; settings = checked_with)))
        @test largest_gap(again.profile, solved.profile) < profile_bound(again) + profile_bound(solved)
    end
end

@testset "both engines trace the folding 3x3 game and the folding 2x2x2 game to maxregret 1e-4 ($element, $engine)" for element in (Float32, widen(Float32)), (engine, settings) in engine_settings(element)
    for payoffs in (FOLDING_3X3_GAME, FOLDING_2X2X2_GAME)
        gains = NormalFormGains(map(table -> element.(table), payoffs))
        result = logit_solve_branch(gains; settings, maxregret = 1.0f-4)
        @test !is_error(result)
        if is_error(result)
            @error "logit_solve_branch failed" engine element failure = unwrap_error(result)
            continue
        end
        branch = unwrap(result)
        @test GameTracing.max_regret(gains, last(branch).logits) <= 1.0f-4 * GameTracing.payoff_range(gains)
        @test all(profile -> settled_point(gains, profile, settings), branch)
    end
end

@testset "a game with a player of one strategy traces under the default and each engine's settings ($element)" for element in (Float32, widen(Float32))
    # A 1x2 game and a 2x1x2 game: the lone strategy's player has an empty logit range.
    games = ((element[1 2], element[3 4]), map(table -> element.(table), ([1; 2;;; 3; 4], [5; 6;;; 7; 8], [2; 1;;; 4; 3])))
    for payoffs in games, (engine, settings) in ["default" => BifurcationKitSettings{element}(); engine_settings(element)]
        gains = NormalFormGains(payoffs)
        result = logit_solve(gains; settings)
        @test !is_error(result)
        if !is_error(result)
            solved = unwrap(result)
            @test all(strategy -> length(strategy) > 1 || strategy == [1], solved.profile)
            @test GameTracing.max_regret(gains, solved.logits) <= GameTracing.default_maxregret(element) * GameTracing.payoff_range(gains)
        end
    end
end

# The 3x3 game of seed 8 in benchmark/logitnash_speed.jl. Its Float32 logits pass 1e3 before maxregret 1e-4, where their
# rounding passes `logit_tolerance`; before `settle_ulps` that trace stalled at λ ≈ 541 (TraceStuck).
@testset "logit_solve meets maxregret where the logits' rounding passes the corrector's tolerance ($element, $engine)" for element in (Float32, widen(Float32)), (engine, settings) in engine_settings(element)
    rng = Xoshiro(8)
    payoffs = ntuple(_ -> randn(rng, (3, 3)), 2)
    gains = NormalFormGains(map(table -> element.(table), payoffs))
    result = logit_solve(gains; maxregret = 1.0f-4, settings)
    @test !is_error(result)
    if !is_error(result)
        (; logits) = unwrap(result)
        @test GameTracing.max_regret(gains, logits) <= 1.0f-4 * GameTracing.payoff_range(gains)
        # Logits whose Float32 ulp passes the tolerance: the stall's cause.
        @test maximum(abs, logits) * eps(Float32) > CorrectorSettings{Float32}().logit_tolerance
    end
end

# The 3x3x3 game of seed 5 in benchmark/logitnash_speed.jl. At λ ≈ 3.24 a halved step's stall shrank to 0.5007 of the
# one before, which the old 1/2 split confirmed as noise: Float32 and Float64 both failed PrecisionExhausted there.
@testset "logit_solve passes a stall that halves with the step, on Gambit's branch ($element, $engine)" for element in (Float32, widen(Float32)), (engine, settings) in engine_settings(element)
    rng = Xoshiro(5)
    payoffs = ntuple(_ -> randn(rng, (3, 3, 3)), 3)
    gains = NormalFormGains(map(table -> element.(table), payoffs))
    result = logit_solve(gains; maxregret = 1.0f-4, settings)
    @test !is_error(result)
    if !is_error(result)
        profile = unwrap(result)
        @test profile.lam > 4
        @test GameTracing.max_regret(gains, profile.logits) <= 1.0f-4 * GameTracing.payoff_range(gains)
        oracle = gambit_strategies(payoffs, profile.lam)
        @test largest_gap(profile.profile, oracle.strategies) < profile_bound(profile) + GAMBIT_PROBABILITY_TOLERANCE
    end
end

@testset "PathFollowing's spent steps, refusals and unresolved steps fail as values ($element)" for element in (Float32, widen(Float32))
    gains = NormalFormGains(map(table -> element.(table), LOGITNASH_GAMES.prisoners_dilemma))
    spent = unwrap_error(logit_solve(gains; settings = PathFollowingSettings{element}(; max_steps = 1)))
    @test (spent.cause, spent.stop.steps) == (TraceCause.NoConvergence{element}(GameTracing.MaxSteps), 1)
    @test unwrap_error(logit_solve_lambda(gains, 100; settings = PathFollowingSettings{element}(; max_steps = 1))).context.lam == 100
    # No precision asks for no trace, on either engine.
    for empty_settings in (BifurcationKitSettings{element}(), PathFollowingSettings{element}())
        @test isempty(unwrap(logit_solve_lambda(gains, element[]; settings = empty_settings)))
    end
    # A first correction far past a 1e-6 target is refused, and no refusal is allowed.
    let distance = CorrectorDistance{element}(; tolerance = 1.0f-7, target_correction = 1.0f-6)
        refused = unwrap_error(logit_solve(gains; settings = PathFollowingSettings{element}(; distance, max_refusals = 0)))
        @test (refused.cause, refused.engine.failure.refusals) == (TraceCause.NoConvergence{element}(GameTracing.MaxRefusals), 1)
    end
    # The refusal factor 2 refuses a first correction past 4 times its target. Gambit's norm has no bound, so any target is refusable.
    @test !is_error(logit_solve(gains; settings = PathFollowingSettings{element}(; distance = CorrectorDistance{element}(; target_correction = 100))))
    # A settle tolerance at or past the target first correction is refused; without a target it is free.
    @test_throws ArgumentError CorrectorDistance{element}(; tolerance = 1 // 4, target_correction = 1 // 4)
    @test CorrectorDistance{element}(; tolerance = 1 // 5, target_correction = 1 // 4).tolerance == element(1 // 5)
    @test CorrectorDistance{element}(; tolerance = 1, target_correction = nothing).tolerance == 1
    # A zero settle tolerance, where the corrector settles by ulps alone, gives the default split check no scale.
    let distance = CorrectorDistance{element}(; tolerance = 0)
        @test_throws ArgumentError PathFollowingSettings{element}(; distance)
        # HalfStepCheck's own zero-tolerance check throws an ArgumentError too; only the default's names the fix.
        @test_throws "pass split_check" PathFollowingSettings{element}(; distance)
        @test isnothing(PathFollowingSettings{element}(; distance, split_check = nothing).split_check)
        check = GameTracing.PathFollowing.HalfStepCheck{element}(; trigger = 13 // 10, tolerance = 1 // 1000)
        @test !is_error(logit_solve(gains; settings = PathFollowingSettings{element}(; distance, split_check = check)))
    end
    @test PathFollowingSettings{element}(; distance = CorrectorDistance{element}(; tolerance = 1 // 10^6)).split_check.tolerance == 10 * element(1 // 10^6)
    @test_throws ArgumentError logit_solve(gains; settings = PathFollowingSettings{element}(; reach = 0))
    # A reach finite in a wider type that rounds to Inf would switch the refusal off.
    @test_throws ArgumentError PathFollowingSettings{element}(; reach = floatmax(widen(element)))
    @test PathFollowingSettings{element}(; reach = Inf32).reach == Inf32
    # A distance in another element type is refused.
    @test_throws TypeError PathFollowingSettings{element}(; distance = CorrectorDistance{BigFloat}())
    # The error estimate's safety converts to the element type.
    halved = PathFollowingSettings{element}(; error_estimate = GameTracing.PathFollowing.ContractionEstimate(; safety = 1 // 2))
    @test halved.error_estimate.safety === element(1 // 2)
    @test PathFollowingSettings{element}(; error_estimate = GameTracing.PathFollowing.LastCorrection()).error_estimate === GameTracing.PathFollowing.LastCorrection()
    # 2^10 / eps ulps resolve only the move off precision zero, so the second step is unresolved.
    unresolved = unwrap_error(logit_solve(gains; maxregret = 1.0f-6, settings = PathFollowingSettings{element}(; min_step_ulps = round(Int, 2^10 / eps(element)))))
    @test (unresolved.cause, unresolved.stop.steps) == (TraceCause.TraceStuck{element}(GameTracing.UnresolvedStep), 1)
    # A trace sent back from its start would fall below precision zero, so every step it tries is refused and halves,
    # until the step is below resolution. Without the orientation check, a step of a few ulps rounds above zero and
    # the trace creeps there.
    reversed = started_tracer(gains, PathFollowingSettings{element}())
    reversed.tangent .*= -1
    recorder = GameTracing.BranchRecorder(gains, zero(element), 1)
    below = unwrap_error(GameTracing.recorded_points(reversed, recorder, element(100), 10, (;)))
    @test (below.cause, below.stop.steps) == (TraceCause.TraceStuck{element}(GameTracing.UnresolvedStep), 0)
    @test below.engine.failure.last_refusal == GameTracing.PathFollowing.AcceptRefused
    @test below.stop.precision == 0 && isempty(recorder.precisions)
    poisoned = NormalFormGains((element[NaN32 0; 0 0], element[0 0; 0 0]))
    @test unwrap_error(logit_solve(poisoned; settings = PathFollowingSettings{element}())).cause == TraceCause.NotFinite{element}(GameTracing.AtStart)
    # A failure at the start names the lowest precision asked for.
    poisoned_start = unwrap_error(logit_solve_lambda(poisoned, [2, 1]; settings = PathFollowingSettings{element}()))
    @test (poisoned_start.cause, poisoned_start.context) == (TraceCause.NotFinite{element}(GameTracing.AtStart), (; lam = element(1)))
    # Step control without its turn and contraction terms still solves.
    plain = PathFollowingSettings{element}(; target_contraction = nothing, target_turn = nothing)
    @test GameTracing.max_regret(gains, unwrap(logit_solve(gains; settings = plain)).logits) <= GameTracing.default_maxregret(element) * GameTracing.payoff_range(gains)
    @test_throws ArgumentError logit_solve_lambda(gains, -1; settings = PathFollowingSettings{element}())
    # No step cap by default; the growth bound then must stay finite.
    @test PathFollowingSettings{element}().max_step == Inf32
    @test_throws ArgumentError logit_solve(gains; settings = PathFollowingSettings{element}(; max_step_growth = Inf32))
end

@testset "both engines default to one step metric and Gambit's step growth ($element)" for element in (Float32, widen(Float32))
    metric = GameTracing.default_step_metric(element)
    @test (metric.distance, metric.precision_term) == (FisherDistance, LogPrecisionTerm)
    @test TraceGeometry{element}(; state_weight = 3 // 10).precision_term == LogPrecisionTerm
    gambit = GameTracing.gambit_settings(element, GameTracing.GAMBIT_MAX_ACCEL)
    bare = BifurcationKitSettings{element}()
    @test gambit.step_metric == PathFollowingSettings{element}().step_metric == metric
    @test typeof(gambit) === typeof(bare)
    @testset "bare BifurcationKitSettings field $name matches gambit_settings" for name in fieldnames(BifurcationKitSettings)
        @test getfield(gambit, name) == getfield(bare, name)
    end
    @test PathFollowingSettings{element}().max_step_growth == element(GameTracing.GAMBIT_MAX_ACCEL) == StepControl{element}().max_step_growth
end

@testset "Gambit's growth constant is pygambit's default" begin
    @test GameTracing.GAMBIT_MAX_ACCEL == gambit_max_accel()
end

@testset "both engines open one step from the game: 1 / gap_bound, else one, at most max_step ($element)" for element in (Float32, widen(Float32))
    gains = NormalFormGains(map(table -> element.(table), LOGITNASH_GAMES.stag_hunt))
    game = LogitGame(gains, one(element))
    @test GameTracing.opening_step(game, GameTracing.default_step_metric(element), Inf32) == inv(element(8))
    @test GameTracing.opening_step(game, GameTracing.default_step_metric(element), 1 // 10) == element(1 // 10)
    flat = LogitGame(NormalFormGains(map(table -> zero(table) .+ element(3), LOGITNASH_GAMES.stag_hunt)), one(element))
    @test GameTracing.opening_step(flat, GameTracing.default_step_metric(element), Inf32) == 1
    poisoned = LogitGame(NormalFormGains((element[NaN32 0; 0 0], element[0 0; 0 0])), one(element))
    @test GameTracing.opening_step(poisoned, GameTracing.default_step_metric(element), Inf32) == 1
    # Narrow integer payoffs: the range must not wrap (-1 in Int8) and fall back to a step of one.
    for (low, high) in ((typemin(Int8), typemax(Int8)), (typemin(UInt8), typemax(UInt8)))
        narrow = NormalFormGains(([low high; high low], [high low; low high]))
        @test GameTracing.payoff_range(narrow) == 255
        @test GameTracing.opening_step(LogitGame(narrow, one(element)), GameTracing.default_step_metric(element), Inf32) == inv(element(255))
    end
    # In the target's unit `λ / P`, a unit step moves `λ` by `P`, so the opening shrinks by `P` (issue #38).
    far = LogitGame(gains, element(1000))
    @test GameTracing.opening_step(far, GameTracing.default_step_metric(element), Inf32) == inv(element(8))
    target = TraceGeometry{element}(; precision_term = TargetPrecisionTerm)
    @test GameTracing.opening_step(far, target, Inf32) ≈ inv(element(8000)) rtol = 4eps(element)
    averaged = TraceGeometry{element}(; distance = AveragedDistance, state_weight = 3 // 4)
    @test GameTracing.opening_step(far, averaged, Inf32) ≈ inv(element(16000)) rtol = 4eps(element)
    # Target times bound overflows, yet the step is a representable subnormal; and a target at floatmax has no finite inverse root.
    sixteen = NormalFormGains((element[0 16; 0 0], element[0 0; 0 0]))
    tiny = GameTracing.opening_step(LogitGame(sixteen, floatmax(element) / 4), target, Inf32)
    @test 0 < tiny < floatmin(element)
    @test tiny ≈ inv(4 * big(floatmax(element))) rtol = 1 // 10^4
    narrow = NormalFormGains((element[0 1 // 1024; 0 0], element[0 0; 0 0]))
    @test GameTracing.opening_step(LogitGame(narrow, floatmax(element)), target, Inf32) ≈ 1024 / big(floatmax(element)) rtol = 1 // 10^4
    # PathFollowing's tracer starts at the same step.
    tracer = started_tracer(gains, PathFollowingSettings{element}())
    @test tracer.step_length == GameTracing.opening_step(game, GameTracing.default_step_metric(element), Inf32)
end

@testset "a pass without gap_bound fails at the opening, naming the method to define" begin
    game = LogitGame(UnboundedPass(GAME), TARGET_PRECISION)
    @test_throws ArgumentError trace_endpoint(game, WIDTH, BifurcationKitSettings{Float32}())
    @test_throws r"define GameTracing\.gap_bound\(::UnboundedPass" GameTracing.gap_bound(game.pass!!)
end

@testset "Gambit's max_accel sets the growth limit ($element)" for element in (Float32, widen(Float32))
    faster = GameTracing.gambit_settings(element, 2)
    @test (faster.step_control.max_step_growth, faster.aggressiveness) == (2, 1)
    # BifurcationKit's growth limit must exceed 1: at 1 the step control only refuses.
    held = GameTracing.gambit_settings(element, 1)
    @test isnothing(held.step_control.max_step_growth) && iszero(held.aggressiveness)
    gains = NormalFormGains(map(table -> element.(table), LOGITNASH_GAMES.stag_hunt))
    steps = Dict(accel => length(unwrap(logit_solve_branch(gains; maxregret = 1.0f-4, max_accel = accel))) for accel in (1, 2))
    @test steps[1] > steps[2]
end

@testset "PathFollowing keeps Gambit's branch where a long step ran back along it ($element)" for element in (Float32, widen(Float32))
    gains = NormalFormGains(map(table -> element.(table), REVERSAL_GAME))
    checked = unwrap(logit_solve(gains; maxregret = 1.0f-4, settings = PathFollowingSettings{element}()))
    oracle = gambit_strategies(REVERSAL_GAME, checked.lam)
    @test largest_gap(checked.profile, oracle.strategies) < profile_bound(checked) + GAMBIT_PROBABILITY_TOLERANCE
    gap_to_checked(overrides) = largest_gap(unwrap(logit_solve(gains; maxregret = 1.0f-4, settings = PathFollowingSettings{element}(; overrides...))).profile, checked.profile)
    # The orientation check alone keeps the branch. At the previous targets the split check alone misses the long step,
    # as its half steps run back too, and the trace ends elsewhere, as it does without either check.
    @test gap_to_checked((; split_check = nothing)) < 1 // 1000
    @test gap_to_checked((; previous_targets(element)..., orientation_check = nothing)) > 1 // 2
    # The default targets take no such long step.
    @test gap_to_checked((; orientation_check = nothing)) < 1 // 1000
    @test gap_to_checked((; orientation_check = nothing, split_check = nothing)) > 1 // 2
    # At the loose targets, growth 2, the summed metric and trigger 1, the split check alone refuses the long step too;
    # without either check the trace ends elsewhere there as well.
    summed = (; loose_targets(element)..., max_step_growth = 2, step_metric = TraceGeometry{element}(; distance = SummedDistance, state_weight = 1))
    trigger_one = GameTracing.PathFollowing.HalfStepCheck{element}(; trigger = 1, tolerance = PathFollowingSettings{element}().split_check.tolerance)
    @test gap_to_checked((; summed..., orientation_check = nothing, split_check = trigger_one)) < 1 // 1000
    @test gap_to_checked((; summed..., orientation_check = nothing, split_check = nothing)) > 1 // 2
end

@testset "PathFollowing's measure is Gambit's norm past each strategy's rounding ($element)" for element in (Float32, widen(Float32))
    distance = CorrectorDistance{element}()
    measure = GameTracing.corrector_measure(distance, (1:1, 2:2))
    @test measure == GameTracing.LogProbabilityChange((1:1, 2:2), distance.settle_ulps * eps(element))
    @test GameTracing.PathFollowing.correction_bound(measure, element) === element(Inf32)
    # A strategy outside the support moves 500 logits and no probability; Gambit's norm counts its log probability.
    @test GameTracing.LogProbabilityChange((1:2, 3:3), zero(element))(element[-1000, 0, 0, 1], element[-500, 0, 0, 1]) ≈ 500
    # A binary player at logit -30 whose logit moves by 1: its first strategy's log probability moves by about 1.
    settled = GameTracing.settle_test(distance, GameTracing.LogProbabilityChange((1:1,), distance.settle_ulps * eps(element)))
    @test !settled(element[-30, 1], element[-29, 1], element[], one(element))
    # Four strategies, the second near certain at logit 10, its logit up 9e-5: Gambit's norm is about √3 · 9e-5.
    four = GameTracing.settle_test(distance, GameTracing.LogProbabilityChange((1:3,), distance.settle_ulps * eps(element)))
    @test !four(element[10, 0, 0, 1], element[10 + 9.0f-5, 0, 0, 1], element[], one(element))
    # A large logit's rounding counts zero in the measure, as in the settle test; four allowances do not.
    large = 64 * distance.tolerance / eps(element)
    rounded, beyond = large + 4 * eps(large), large + 4 * distance.settle_ulps * eps(element) * large
    within = GameTracing.LogProbabilityChange((1:1,), distance.settle_ulps * eps(element))
    @test within(element[large, 1], element[rounded, 1]) == 0
    @test settled(element[large, 1], element[rounded, 1], element[], one(element))
    @test within(element[large, 1], element[beyond, 1]) > distance.tolerance
    @test !settled(element[large, 1], element[beyond, 1], element[], one(element))
    # Newton's error estimate scales the move: a tenth of a small one settles.
    @test settled(element[0, 1], element[4 * distance.tolerance, 1], element[], element(1 // 10))
    @test !settled(element[0, 1], element[4 * distance.tolerance, 1], element[], one(element))
    @test !settled(element[NaN32, 1], element[0, 1], element[], one(element))
    # Step control refuses past 4 targets, at Gambit's 0.4.
    @test 4 * distance.target_correction == element(2 // 5)
end

@testset "both engines turn a fold in Float64; in Float32 each fails loudly at it" begin
    # The benchmark's 20x20 game from seed 6 folds back at λ ≈ 315.68, with logits near 600 on strategies near zero.
    rng = Xoshiro(6)
    payoffs = ntuple(_ -> randn(rng, (20, 20)), 2)
    wide = widen(Float32)
    for settings in (PathFollowingSettings{wide}(), BifurcationKitSettings{wide}())
        @test !is_error(logit_solve(NormalFormGains(payoffs); maxregret = 1.0f-4, settings))
    end
    # A declared Float32 failure. The Newton step is ill-conditioned at the fold, and Gambit's norm counts the saturated
    # strategies' log probabilities, so its error there passes any rounding allowance. Fisher–Rao ignored them.
    narrow = NormalFormGains(map(table -> Float32.(table), payoffs))
    path_following = unwrap_error(logit_solve(narrow; maxregret = 1.0f-4, settings = PathFollowingSettings{Float32}()))
    @test path_following.cause == TraceCause.TraceStuck{Float32}(GameTracing.UnresolvedStep)
    @test path_following.engine.failure.last_refusal == PathFollowing.Diverging
    @test abs(path_following.stop.precision - 315.68f0) < 1
    bifurcation_kit = unwrap_error(logit_solve(narrow; maxregret = 1.0f-4, settings = BifurcationKitSettings{Float32}()))
    @test failed_as(bifurcation_kit, TraceCause.PrecisionExhausted)
    @test bifurcation_kit.cause.stage == GameTracing.MidTrace
    @test abs(bifurcation_kit.stop.precision - 315.68f0) < 1
    # In the summed metric the turn spans about 1e-4 in λ, four Float32 ulps: Newton diverges there too.
    summed = TraceGeometry{Float32}(; distance = SummedDistance, state_weight = 1)
    failure = unwrap_error(logit_solve(narrow; maxregret = 1.0f-4, settings = PathFollowingSettings{Float32}(; step_metric = summed)))
    @test failed_as(failure, TraceCause.TraceStuck)
    @test abs(failure.stop.precision - 315.68f0) < 1
end

@testset "PathFollowing's metric is the trace geometry's at target precision one ($element)" for element in (Float32, widen(Float32))
    rng = fresh_rng()
    gains = NormalFormGains((randn(rng, element, 3, 2), randn(rng, element, 3, 2)))
    width = logit_count(gains)
    point = [randn(rng, element, width); 3 * rand(rng, element)]
    for geometry in every_geometry(element(1 // 2))
        frozen = PathFollowing.frozen_metric(GameTracing.trace_metric(geometry, gains, width, one(element), one(element)), point)
        if geometry.distance == FisherDistance
            precision_weight = geometry.precision_term == LogPrecisionTerm ? inv(1 + last(point))^2 : one(element)
            expected = fisher_operator(GameTracing.logit_ranges(gains), point, geometry.state_weight, precision_weight)
        else
            expected = Diagonal([fill(GameTracing.coordinate_weight(geometry, width), width); GameTracing.precision_weight(geometry)])
        end
        @test inner_matrix(frozen, element, width + 1) ≈ expected rtol = sqrt(eps(element))
    end
    pennies = NormalFormGains(map(table -> element.(table), LOGITNASH_GAMES.matching_pennies))
    log_precision = TraceGeometry{element}(; precision_term = LogPrecisionTerm, state_weight = 1 // 2)
    solved = unwrap(logit_solve(pennies; settings = PathFollowingSettings{element}(; step_metric = log_precision)))
    @test GameTracing.max_regret(pennies, solved.logits) <= GameTracing.default_maxregret(element) * GameTracing.payoff_range(pennies)
end

@testset "a PathFollowing step that does not converge fails at its refusal budget, or retries shorter without one ($element)" for element in (Float32, widen(Float32))
    gains = NormalFormGains(map(table -> element.(table), LOGITNASH_GAMES.prisoners_dilemma))
    distance = CorrectorDistance{element}(; tolerance = eps(element), settle_ulps = 0)
    budget, unbounded = (PathFollowingSettings{element}(; newton_iterations = 1, distance, max_refusals) for max_refusals in (2, nothing))
    tracer = started_tracer(gains, budget)
    # The opening predictor passes this precision, so the first attempt corrects at it.
    precision = tracer.step_length * last(tracer.tangent) * 9 // 10
    failure = unwrap_error(last(GameTracing.stepped_landing!!(tracer, precision, 10, GameTracing.AcceptAll(), (; lam = precision))))
    @test (failure.cause, failure.context.lam) == (TraceCause.NoConvergence{element}(GameTracing.MaxRefusals), precision)
    @test failure.engine.failure.last_refusal == PathFollowing.NotConverged
    @test iszero(failure.stop.precision)
    # Without a budget, the refused step halves until one Newton iteration converges it, and the trace moves on.
    moved = unwrap_error(last(GameTracing.stepped_landing!!(started_tracer(gains, unbounded), precision, 10, GameTracing.AcceptAll(), (; lam = precision))))
    @test moved.cause == TraceCause.NoConvergence{element}(GameTracing.MaxSteps)
    @test 0 < moved.stop.precision < precision
    ended = unwrap_error(logit_solve(gains; settings = unbounded))
    @test ended.cause == TraceCause.TraceStuck{element}(GameTracing.UnresolvedStep)
end

@testset "the solvers return the trace's failure when it runs out of steps" begin
    gains = NormalFormGains(map(table -> Float32.(table), LOGITNASH_GAMES.prisoners_dilemma))
    settings = BifurcationKitSettings{Float32}(; max_steps = 1)
    # A one-step budget ends a stalled step as a stall (TraceStuck), a moving one as NoConvergence, as on main.
    spent = (TraceCause.NoConvergence, TraceCause.TraceStuck)
    @test failed_as(unwrap_error(logit_solve_lambda(gains, 100; settings)), spent...)
    for failure in unwrap_error.((logit_solve_branch(gains; settings), logit_solve(gains; settings)))
        @test failed_as(failure, spent...)
    end
end

@testset "the solvers refuse Gambit's invalid arguments" begin
    gains = NormalFormGains(map(table -> Float32.(table), LOGITNASH_GAMES.matching_pennies))
    @test_throws ArgumentError logit_solve(gains; maxregret = 0)
    @test_throws ArgumentError logit_solve_lambda(gains, 1; max_accel = 0.5f0)
    # A wider precision can round to Inf in Float32.
    huge = floatmax(widen(Float32))
    for (_, settings) in engine_settings(Float32)
        # The tiny negative precision rounds to -0 in Float32.
        for lam in (-1, NaN32, huge, -nextfloat(zero(widen(Float32))))
            @test_throws ArgumentError logit_solve_lambda(gains, lam; settings)
        end
    end
    @test_throws ArgumentError trace_endpoint(LogitGame(gains, -1.0f0), 2, BifurcationKitSettings{Float32}())
    # A target given in a wider type is checked as stored too.
    @test LogitGame{typeof(gains), Float32}(gains, 1).target_precision === 1.0f0
    @test_throws ArgumentError LogitGame{typeof(gains), Float32}(gains, huge)
    @test_throws ArgumentError LogitGame{typeof(gains), Float32}(gains, -nextfloat(zero(widen(Float32))))
end

@testset "a zero precision gives the flat start, with no step and no log ($element)" for element in (Float32, widen(Float32))
    payoffs = map(table -> element.(table), LOGITNASH_GAMES.stag_hunt)
    for step_metric in every_geometry(element(1 // 10))
        settings = BifurcationKitSettings{element}(; step_metric)
        traced = unwrap(@test_logs trace_endpoint(LogitGame(NormalFormPass(payoffs), zero(element)), 2, settings))
        @test traced.logits == zeros(element, 2) && traced.steps == 0 && traced.logit_accuracy == 0
    end
    # A target that rounds to zero in the trace's type takes the same path.
    rounded = unwrap(@test_logs trace_endpoint(LogitGame(NormalFormPass(payoffs), big"1e-1000"), 2, BifurcationKitSettings{element}()))
    @test rounded.logits == zeros(element, 2) && rounded.steps == 0
    for (_, settings) in engine_settings(element)
        profiles = unwrap(@test_logs logit_solve_lambda(NormalFormGains(payoffs), [1, 0]; settings))
        @test last(profiles).lam == 0 && all(==(fill(element(0.5f0), 2)), last(profiles).profile)
        @test first(profiles).lam == 1
    end
end

@testset "a target that squares out of range throws in the plain distances; the log-precision distance traces it ($element)" for element in (Float32, widen(Float32))
    pass = NormalFormPass(map(table -> element.(table), LOGITNASH_GAMES.stag_hunt))
    tiny, huge = sqrt(nextfloat(zero(element))), sqrt(floatmax(element)) * 2
    for distance in (AveragedDistance, SummedDistance, FisherDistance), target in (tiny, huge)
        settings = BifurcationKitSettings{element}(step_metric = TraceGeometry{element}(; distance, precision_term = TargetPrecisionTerm))
        @test_throws ArgumentError trace_endpoint(LogitGame(pass, target), 2, settings)
    end
    settings = BifurcationKitSettings{element}(step_metric = TraceGeometry{element}(distance = FisherDistance, precision_term = LogPrecisionTerm))
    # To first order the logits are the target times the gains at even odds: (8 + 0) / 2 - (7 + 5) / 2 = -2 for each player.
    @test unwrap(trace_endpoint(LogitGame(pass, tiny), 2, settings)).logits ≈ fill(-2 * tiny, 2) rtol = 1.0f-3
end

@testset "PathFollowing's first step is finite and under max_step ($element)" for element in (Float32, widen(Float32))
    gains = NormalFormGains((element[0, 1],))
    capped = started_tracer(gains, PathFollowingSettings{element}(; max_step = element(0.01f0)))
    @test capped.step_length == element(0.01f0)
    # The payoff range's inverse overflows.
    flat = NormalFormGains((element[0, nextfloat(zero(element))],))
    @test GameTracing.opening_step(LogitGame(flat, one(element)), GameTracing.default_step_metric(element), Inf32) == 1
    @test !is_error(logit_solve_lambda(flat, one(element); settings = PathFollowingSettings{element}()))
end

@testset "the solvers return profiles in the payoffs' element type and refuse settings in another" begin
    for element in (Float32, float(Int))
        gains = NormalFormGains(map(table -> element.(table), LOGITNASH_GAMES.prisoners_dilemma))
        @test eltype(gains) == element
        @test unwrap(logit_solve(gains)).lam isa element
        @test eltype(unwrap(logit_solve(gains)).logits) == element
        @test eltype(only(unwrap(logit_solve_lambda(gains, 2))).logits) == element
        other = element == Float32 ? float(Int) : Float32
        for settings in (BifurcationKitSettings{other}(), PathFollowingSettings{other}())
            @test_throws ArgumentError logit_solve_lambda(gains, 2; settings)
            @test_throws ArgumentError logit_solve_branch(gains; settings)
            @test_throws ArgumentError logit_solve(gains; settings)
        end
        @test unwrap(logit_solve(gains; settings = PathFollowingSettings{element}())).lam isa element
    end
    # Integer payoffs trace in `float(Int)`, as Julia promotes them.
    integer = NormalFormGains(ROCK_PAPER_SCISSORS)
    @test eltype(integer) == Int
    @test eltype(unwrap(logit_solve(integer)).logits) == float(Int)
    @test eltype(only(unwrap(logit_solve_lambda(integer, 2))).logits) == float(Int)
    @test_throws ArgumentError logit_solve(integer; settings = BifurcationKitSettings{Float32}())
end

@testset "LogitQREProfile computes its profile from its logits and takes no profile" begin
    gains = NormalFormGains(map(table -> Float32.(table), LOGITNASH_GAMES.matching_pennies))
    profile = LogitQREProfile(gains, 1.0f0, Float32[0, 0])
    @test profile.profile == GameTracing.mixed_strategies(gains, profile.logits)
    @test_throws MethodError LogitQREProfile(1.0f0, profile.profile, Float32[3, -3])
end

"""A `BranchRecorder` of `gains` with window `window`, holding `precisions` with least regrets `least`."""
function filled_recorder(gains::NormalFormGains, window::Integer, precisions::Vector{E}, least::Vector{E})::GameTracing.BranchRecorder where {E <: Real}
    recorder = GameTracing.BranchRecorder(gains, zero(E), window)
    append!(recorder.precisions, precisions)
    append!(recorder.least, least)
    append!(recorder.logits, [zeros(E, logit_count(gains)) for _ in precisions])
    return recorder
end

@testset "the regret stalls where window rising points set no new least regret ($element)" for element in (Float32, widen(Float32))
    gains = NormalFormGains(map(table -> element.(table), LOGITNASH_GAMES.matching_pennies))
    rising, held = element[1, 2, 3, 4], element[1 // 2, 1 // 4, 1 // 4, 1 // 4]
    @test GameTracing.regret_stalled(filled_recorder(gains, 2, rising, held))
    # Too few points, a new least regret in the window, or a fold in it.
    @test !GameTracing.regret_stalled(filled_recorder(gains, 4, rising, held))
    @test !GameTracing.regret_stalled(filled_recorder(gains, 2, rising, element[1 // 2, 1 // 4, 1 // 4, 1 // 8]))
    @test !GameTracing.regret_stalled(filled_recorder(gains, 2, element[1, 3, 2, 4], held))
    @test !GameTracing.regret_stalled(filled_recorder(gains, 2, element[1, 2, 2, 4], held))
    # A trace at its target unsettled fails PrecisionExhausted at its last point, stalled or not.
    for window in (2, 4)
        ended = unwrap_error(GameTracing.branch_outcome(filled_recorder(gains, window, rising, held), Result{Nothing, TraceFailure{element, @NamedTuple{}}}(Ok(nothing)), (;)))
        @test failed_as(ended, TraceCause.PrecisionExhausted) && ended.cause.stage == GameTracing.AtEndpoint && ended.stop.precision == 4
    end
    for window in (0, -1)
        @test_throws ArgumentError GameTracing.BranchRecorder(gains, zero(element), window)
        @test_throws ArgumentError logit_solve(gains; stall_window = window, settings = PathFollowingSettings{element}())
    end
end

@testset "a regret trace at its target unsettled fails PrecisionExhausted there ($engine, $element)" for element in (Float32, widen(Float32)), (engine, settings) in engine_settings(element)
    gains = NormalFormGains(map(table -> element.(table), LOGITNASH_GAMES.prisoners_dilemma))
    # A zero tolerance never settles, and a target of 2 comes long before a stall.
    recorder = GameTracing.BranchRecorder(gains, zero(element), GameTracing.DEFAULT_STALL_WINDOW)
    ended = GameTracing.engine_points(gains, settings, element(2), recorder, (;))
    @test !is_error(ended) && last(recorder.precisions) == 2
    failure = unwrap_error(GameTracing.branch_outcome(recorder, ended, (;)))
    @test failed_as(failure, TraceCause.PrecisionExhausted) && failure.cause.stage == GameTracing.AtEndpoint && failure.stop.precision == 2
end

@testset "a trace past its regret floor fails PrecisionExhausted at its stall, with no cap ($engine, $element)" for element in (Float32, widen(Float32)), (engine, settings) in engine_settings(element)
    gains = NormalFormGains(map(table -> element.(table), SLOW_REGRET_GAME))
    # Far under the regret either float type reaches on this game; with no stall check a trace creeps at its floor.
    failure = unwrap_error(logit_solve(gains; maxregret = eps(element)^2, settings))
    @test failed_as(failure, TraceCause.PrecisionExhausted)
    @test failure.context.maxregret == eps(element)^2
    @test 0 < failure.stop.precision < GameTracing.precision_ceiling(element)
    # BifurcationKit's corrector stalls first, mid-trace; PathFollowing runs on to the regret stall.
    @test failure.cause.stage == (engine == "PathFollowing" ? GameTracing.AtEndpoint : GameTracing.MidTrace)
    if failure.cause.stage == GameTracing.AtEndpoint
        # The regret stall: the window's points past the last new least regret, then the endpoint check's correction there.
        @test failure.stop.steps > GameTracing.DEFAULT_STALL_WINDOW
        @test failure.cause.stall.precision == failure.stop.precision
        @test failure.engine == EngineRecord.NoRecord{element}()
        # A longer window runs further along the floor.
        longer = unwrap_error(logit_solve(gains; maxregret = eps(element)^2, stall_window = 2 * GameTracing.DEFAULT_STALL_WINDOW, settings))
        @test failed_as(longer, TraceCause.PrecisionExhausted) && longer.stop.steps > failure.stop.steps
    end
end

@testset "precision_ceiling is eps^-2, whose square is finite, and a regret trace needs a norm free of the target ($element)" for element in (Float32, widen(Float32))
    ceiling = GameTracing.precision_ceiling(element)
    @test isfinite(ceiling^2) && ceiling == exp2(2 * (precision(element) - 1))
    # Float16's eps^-2 squared overflows, so a Float16 regret trace throws before it starts.
    @test_throws ArgumentError GameTracing.precision_ceiling(Float16)
    @test_throws "cannot hold the square" logit_solve(NormalFormGains(map(table -> Float16.(table), LOGITNASH_GAMES.prisoners_dilemma)); settings = PathFollowingSettings{Float16}())
    gains = NormalFormGains(map(table -> element.(table), LOGITNASH_GAMES.prisoners_dilemma))
    # A plain norm scales its steps by the target, so it traces only to a set precision.
    plain = BifurcationKitSettings{element}(; step_metric = TraceGeometry{element}(; precision_term = TargetPrecisionTerm))
    @test_throws ArgumentError logit_solve(gains; settings = plain)
    @test_throws ArgumentError logit_solve_branch(gains; settings = plain)
    @test !is_error(logit_solve_lambda(gains, 100; settings = plain))
end

@testset "logit_solve_lambda lands past precision_ceiling, with no regret stall ($engine, $element)" for element in (Float32, widen(Float32)), (engine, settings) in engine_settings(element)
    gains = NormalFormGains(map(table -> element.(table), LOGITNASH_GAMES.prisoners_dilemma))
    past = 2 * GameTracing.precision_ceiling(element)
    landed = logit_solve_lambda(gains, past; settings)
    @test !is_error(landed) && only(unwrap(landed)).lam == past && all(isfinite, only(unwrap(landed)).logits)
    # A λ endpoint is its own stop, so the stall window is not its keyword.
    @test_throws MethodError logit_solve_lambda(gains, past; stall_window = 1, settings)
end

@testset "no step or refusal budget by default, and a negative budget throws ($element)" for element in (Float32, widen(Float32))
    @test isnothing(BifurcationKitSettings{element}().max_steps)
    @test isnothing(PathFollowingSettings{element}().max_steps) && isnothing(PathFollowingSettings{element}().max_refusals)
    @test_throws ArgumentError BifurcationKitSettings{element}(; max_steps = -1)
    @test_throws ArgumentError PathFollowingSettings{element}(; max_steps = -1)
    @test_throws ArgumentError PathFollowingSettings{element}(; max_refusals = -1)
    # Each engine traces to its target with no budget.
    gains = NormalFormGains(map(table -> element.(table), LOGITNASH_GAMES.matching_pennies))
    for settings in (BifurcationKitSettings{element}(), PathFollowingSettings{element}())
        @test !is_error(trace_endpoint(LogitGame(gains, element(10)), logit_count(gains), settings))
    end
end
