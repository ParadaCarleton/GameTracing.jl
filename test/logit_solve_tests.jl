using ErrorTypes: is_error, unwrap, unwrap_error
using LinearAlgebra: Diagonal
using PathFollowing: PathFollowing
using Random: AbstractRNG, Xoshiro

"""The largest difference in any probability between two profiles of mixed strategies."""
function largest_gap(left::Tuple, right::Tuple)::eltype(first(left))
    return maximum(maximum.(abs, left .- right))
end

# Gambit's corrector stops at c_tol = 1e-4 in log probabilities (src/solvers/nash/logit/path.cc), so ≤ 1e-4 in probability.
const GAMBIT_PROBABILITY_TOLERANCE = 1.0f-4

"""
The largest probability error of the trace's `profile`: softmax slope 1/2 times its logit error, the corrector's
`logit_tolerance` (a settled corrector's last Newton correction) plus 32 ulps of the largest logit.
"""
function profile_bound(profile::LogitQREProfile{E})::E where {E <: Real}
    return (CorrectorSettings{E}().logit_tolerance + 32 * eps(E) * maximum(abs, profile.logits)) / 2
end

const ROCK_PAPER_SCISSORS = ([0 -1 1; 1 0 -1; -1 1 0], [0 1 -1; -1 0 1; 1 -1 0])

"""Each engine's default settings in `E`, by name: BifurcationKit's as `logit_solve` builds them, and PathFollowing's."""
function engine_settings(::Type{E})::Vector{Pair{String, EngineSettings{E}}} where {E <: Real}
    return ["BifurcationKit" => GameTracing.gambit_settings(E(0.03f0), 1.1f0), "PathFollowing" => PathFollowingSettings{E}()]
end

@testset "logit_solve_lambda agrees with Gambit's logit_solve_lambda at each precision ($engine)" for (engine, settings) in engine_settings(Float32)
    rng = fresh_rng()
    random_games = [ntuple(_ -> 3 .* randn(rng, counts), length(counts)) for counts in ((3, 3), (2, 3, 2), (4, 2))]
    games = [LOGITNASH_GAMES.prisoners_dilemma, LOGITNASH_GAMES.matching_pennies, ROCK_PAPER_SCISSORS, random_games...]
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
        # Regret sqrt(eps) of the payoff range bounds a dominated strategy's weight near sqrt(eps) of the gap it loses.
        @test largest_gap(solved.profile, equilibrium) < 1.0f-2
        @test GameTracing.max_regret(gains, solved.logits) <= sqrt(eps(Float32)) * GameTracing.payoff_range(gains)
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
(`branch_step`): in logits, or in strategies (`StrategyDistance`).
"""
settled_point(gains::NormalFormGains, profile::LogitQREProfile, ::BifurcationKitSettings)::Bool =
    settled_branch_logits(gains, profile.logits, profile.lam, BifurcationKitSettings{eltype(profile.logits)}())
settled_point(gains::NormalFormGains, profile::LogitQREProfile, settings::PathFollowingSettings)::Bool =
    settled_strategies(gains, profile.logits, profile.lam, settings.distance)

"""The random games of the branch testset, one per strategy count in `((3, 3), (2, 2, 2))`, drawn from `rng`."""
function branch_games(rng::AbstractRNG)::AbstractVector
    return [ntuple(_ -> 3 .* randn(rng, Float32, counts), length(counts)) for counts in ((3, 3), (2, 2, 2))]
end

@testset "logit_solve_branch: each point solves the logit map at its precision, and the last meets maxregret ($engine)" for (engine, settings) in engine_settings(Float32)
    stuck_games = branch_games(Xoshiro(STUCK_BRANCH_SEED))
    # The seed's first payoff, as the failing run logged it.
    @test first(first(first(stuck_games))) == -0.4899592f0
    for payoffs in (stuck_games..., SATURATED_BRANCH_GAME, FOLD_PEAK_GAME, FOLD_EXHAUSTED_GAME, branch_games(fresh_rng())...)
        gains = NormalFormGains(payoffs)
        result = logit_solve_branch(gains; settings)
        @test !is_error(result)
        if is_error(result)
            @error "logit_solve_branch failed" payoffs = repr(payoffs) payoff_bits = map(table -> reinterpret.(UInt32, table), payoffs) failure = unwrap_error(result)
            continue
        end
        branch = unwrap(result)
        @test all(profile -> profile.lam > 0, branch)
        @test all(profile -> all(isfinite, profile.logits), branch)
        @test GameTracing.max_regret(gains, last(branch).logits) <= sqrt(eps(Float32)) * GameTracing.payoff_range(gains)
        solved = unwrap(logit_solve(gains; settings))
        @test solved.lam == last(branch).lam
        for profile in branch
            @test settled_point(gains, profile, settings)
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
        again = only(unwrap(logit_solve_lambda(gains, solved.lam; settings)))
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
            @test GameTracing.max_regret(gains, solved.logits) <= sqrt(eps(element)) * GameTracing.payoff_range(gains)
        end
    end
end

@testset "logit_solve_branch fails NoConvergence when maximum_precision comes before maxregret ($engine)" for (engine, settings) in engine_settings(Float32)
    gains = NormalFormGains(map(table -> Float32.(table), LOGITNASH_GAMES.prisoners_dilemma))
    failure = unwrap_error(logit_solve_branch(gains; maximum_precision = 1, settings))
    @test failure.cause == GameTracing.NoConvergence
    @test failure.context.reason == :maximum_precision
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
    @test (spent.cause, spent.context.reason, spent.context.steps) == (GameTracing.NoConvergence, :max_steps, 1)
    @test unwrap_error(logit_solve_lambda(gains, 100; settings = PathFollowingSettings{element}(; max_steps = 1))).context.lam == 100
    # A first correction far past a 1e-6 target is refused, and no refusal is allowed.
    for distance in (StrategyDistance{element}(; target_correction = 1.0f-6), LogitDistance{element}(; target_correction = 1.0f-6))
        refused = unwrap_error(logit_solve(gains; settings = PathFollowingSettings{element}(; distance, max_refusals = 0)))
        @test (refused.cause, refused.context.reason, refused.context.refusals) == (GameTracing.TraceStuck, :refusals, 1)
    end
    # A distance in another element type is refused.
    @test_throws MethodError PathFollowingSettings{element}(; distance = StrategyDistance{BigFloat}())
    # The error estimate's safety converts to the element type.
    halved = PathFollowingSettings{element}(; error_estimate = GameTracing.PathFollowing.ContractionEstimate(; safety = 1 // 2))
    @test halved.error_estimate.safety === element(1 // 2)
    @test PathFollowingSettings{element}(; error_estimate = GameTracing.PathFollowing.LastCorrection()).error_estimate === GameTracing.PathFollowing.LastCorrection()
    # 2^10 / eps ulps resolve only the move off precision zero, so the second step is unresolved.
    unresolved = unwrap_error(logit_solve(gains; maxregret = 1.0f-6, settings = PathFollowingSettings{element}(; min_step_ulps = round(Int, 2^10 / eps(element)))))
    @test (unresolved.cause, unresolved.context.reason, unresolved.context.steps) == (GameTracing.TraceStuck, :unresolved_step, 1)
    # A trace sent back from its start would fall below precision zero, so every step it tries is refused. Without the
    # orientation check, a step of a few ulps rounds above zero and the trace creeps there until `max_steps`.
    reversed = unwrap(GameTracing.path_init(gains, PathFollowingSettings{element}()))
    reversed.tangent .*= -1
    recorder = GameTracing.BranchRecorder(gains, zero(element))
    below = unwrap_error(GameTracing.recorded_points(reversed, recorder, element(100), 10, (;)))
    @test (below.cause, below.context.reason, below.context.steps) == (GameTracing.TraceStuck, :refusals, 0)
    @test below.context.last_refusal == GameTracing.PathFollowing.AcceptRefused
    @test below.context.precision == 0 && isempty(recorder.precisions)
    poisoned = NormalFormGains((element[NaN32 0; 0 0], element[0 0; 0 0]))
    @test unwrap_error(logit_solve(poisoned; settings = PathFollowingSettings{element}())).cause == GameTracing.NotFinite
    @test unwrap_error(logit_solve_lambda(poisoned, 1; settings = PathFollowingSettings{element}())).context.reason == :NotFiniteStart
    # Step control without its turn and contraction terms still solves.
    plain = PathFollowingSettings{element}(; target_contraction = nothing, target_turn = nothing)
    @test GameTracing.max_regret(gains, unwrap(logit_solve(gains; settings = plain)).logits) <= sqrt(eps(element)) * GameTracing.payoff_range(gains)
    @test_throws ArgumentError logit_solve_lambda(gains, -1; settings = PathFollowingSettings{element}())
    # No step cap by default; the growth bound then must stay finite.
    @test PathFollowingSettings{element}().max_step == Inf32
    @test_throws ArgumentError logit_solve(gains; settings = PathFollowingSettings{element}(; max_step_growth = Inf32))
    # The first step comes from the game: the precision where a logit moves by one, or one for equal payoffs.
    @test GameTracing.opening_step(gains, element) == inv(element(GameTracing.payoff_range(gains)))
    @test GameTracing.opening_step(NormalFormGains((zeros(element, 2, 2), zeros(element, 2, 2))), element) == 1
    @test GameTracing.opening_step(poisoned, element) == 1
end

@testset "PathFollowing keeps Gambit's branch where a long step ran back along it ($element)" for element in (Float32, widen(Float32))
    gains = NormalFormGains(map(table -> element.(table), REVERSAL_GAME))
    checked = unwrap(logit_solve(gains; maxregret = 1.0f-4, settings = PathFollowingSettings{element}()))
    oracle = gambit_strategies(REVERSAL_GAME, checked.lam)
    @test largest_gap(checked.profile, oracle.strategies) < profile_bound(checked) + GAMBIT_PROBABILITY_TOLERANCE
    # The control: the same trace without the check ends elsewhere.
    unchecked = unwrap(logit_solve(gains; maxregret = 1.0f-4, settings = PathFollowingSettings{element}(; orientation_check = nothing)))
    @test largest_gap(unchecked.profile, checked.profile) > 1 // 2
end

@testset "PathFollowing keeps the principal branch past an unfolded pitchfork ($element)" for element in (Float32, widen(Float32))
    gains = NormalFormGains(map(table -> element.(table), UNFOLDED_PITCHFORK_GAME))
    checked = unwrap(logit_solve(gains; maxregret = 1.0f-4, settings = PathFollowingSettings{element}()))
    # The branch ends near the pure equilibrium (1, 1); the middle arm it passes ends mixed.
    @test all(strategy -> first(strategy) > 99 // 100, checked.profile)
    oracle = gambit_strategies(UNFOLDED_PITCHFORK_GAME, checked.lam)
    @test largest_gap(checked.profile, oracle.strategies) < profile_bound(checked) + GAMBIT_PROBABILITY_TOLERANCE
end

@testset "FisherRaoChange is the largest Fisher–Rao distance any player's strategy moves ($element)" for element in (Float32, widen(Float32))
    one_player = GameTracing.FisherRaoChange((1:1,))
    # From even odds to odds of 1:3 the Bhattacharyya angle is 15°, so the distance is twice that: π / 6.
    @test one_player(element[0, 1], element[log(element(3)), 1]) ≈ element(π) / 6
    # The same strategy at twice the precision and half the gap.
    @test one_player(element[0, 1], element[log(element(3)) / 2, 2]) ≈ element(π) / 6
    @test isnan(one_player(element[NaN32, 1], element[0, 1]))
    two_players = GameTracing.FisherRaoChange((1:2, 3:3))
    @test two_players(element[0, 0, 0, 1], element[0, 0, log(element(3)), 1]) ≈ element(π) / 6
    # A strategy outside the support moves 500 logits, but no probability.
    @test two_players(element[-1000, 0, 0, 1], element[-500, 0, 0, 1]) <= eps(element)
    @test GameTracing.logit_change(element[-1000, 0, 0, 1], element[-500, 0, 0, 1]) == 500
    # A player whose first strategy left the support: its logits move together, and its strategy does not.
    @test two_players(element[1000, 1000, 0, 1], element[1500, 1500, 0, 1]) <= eps(element)
    # At odds 1:3:1 only the middle logit, log 3, is nonzero, so the spread is √(3/5) log 3.
    @test last(GameTracing.fisher_rao_spread(element[0, 0, 1], element[log(element(3)), 0, 1], 1:2)) ≈ sqrt(element(3) / 5) * log(element(3))
end

@testset "a strategy move within its logits' settle_ulps settles; one past it by the tolerance does not ($element)" for element in (Float32, widen(Float32))
    distance = StrategyDistance{element}()
    tolerance, ulps = distance.tolerance, distance.settle_ulps
    settled = GameTracing.settle_test(distance, GameTracing.FisherRaoChange((1:2,)))
    # Strategies 2 and 3 share a large logit; the first has left the support. Their rounding alone passes the tolerance.
    large = 64 * tolerance / eps(element)
    rounded = large + 4 * eps(large)
    @test GameTracing.FisherRaoChange((1:2,))(element[large, large, 1], element[rounded, large, 1]) > tolerance
    @test settled(element[large, large, 1], element[rounded, large, 1], element[], one(element))
    # A logit gap of four allowances moves the strategy two, one allowance past the test.
    beyond = large + 4 * ulps * eps(element) * large
    @test !settled(element[large, large, 1], element[beyond, large, 1], element[], one(element))
    # Newton's error estimate scales the move: a tenth of it settles.
    @test settled(element[large, large, 1], element[beyond, large, 1], element[], element(1 // 10))
    # Small logits have no allowance to speak of: a move of four tolerances stays past it.
    @test !settled(element[0, 0, 1], element[4 * tolerance, 0, 1], element[], one(element))
    @test !settled(element[NaN32, 0, 1], element[0, 0, 1], element[], one(element))
end

@testset "StrategyDistance turns a fold whose start settled only to its tolerance" begin
    # The benchmark's 20x20 game from seed 6 folds back at λ ≈ 315.68, where each start settles only to its tolerance:
    # a leftover error of about 1.5e-5 in Fisher–Rao, more than short predictor moves.
    rng = Xoshiro(6)
    payoffs = ntuple(_ -> randn(rng, (20, 20)), 2)
    wide = widen(Float32)
    @test !is_error(logit_solve(NormalFormGains(payoffs); maxregret = 1.0f-4, settings = PathFollowingSettings{wide}()))
    # The turn spans about 1e-4 in λ, four Float32 ulps: Newton diverges there, and the trace stops loudly at the fold.
    narrow = NormalFormGains(map(table -> Float32.(table), payoffs))
    failure = unwrap_error(logit_solve(narrow; maxregret = 1.0f-4, settings = PathFollowingSettings{Float32}()))
    @test failure.cause == GameTracing.TraceStuck
    @test abs(failure.context.precision - 315.68f0) < 1
end

@testset "StrategyDistance takes long steps where a strategy leaves the support ($element)" for element in (Float32, widen(Float32))
    # The five-player game of `randn(Xoshiro(1), (2, 2, 2, 2, 2))` per player: its branch ends mixed, and a strategy
    # outside the support has a logit that grows like λ. Measured in logits, its corrections hold the steps short.
    rng = Xoshiro(1)
    gains = NormalFormGains(ntuple(_ -> element.(randn(rng, (2, 2, 2, 2, 2))), 5))
    branches = map((StrategyDistance{element}(), LogitDistance{element}())) do distance
        unwrap(logit_solve_branch(gains; maxregret = 1.0f-5, settings = PathFollowingSettings{element}(; distance)))
    end
    @test 2 * length(first(branches)) < length(last(branches))
    @test largest_gap(last(first(branches)).profile, last(last(branches)).profile) < 1.0f-3
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
    @test GameTracing.max_regret(pennies, solved.logits) <= sqrt(eps(element)) * GameTracing.payoff_range(pennies)
end

@testset "a PathFollowing landing that does not converge fails TraceStuck ($element)" for element in (Float32, widen(Float32))
    gains = NormalFormGains(map(table -> element.(table), LOGITNASH_GAMES.prisoners_dilemma))
    for distance in (StrategyDistance{element}(; tolerance = eps(element), settle_ulps = 0), LogitDistance{element}(; tolerance = eps(element), settle_ulps = 0))
        tracer = unwrap(GameTracing.path_init(gains, PathFollowingSettings{element}(; newton_iterations = 1, distance)))
        failure = unwrap_error(GameTracing.landed_point(tracer, element(1), (; lam = 1)))
        @test (failure.cause, failure.context.reason, failure.context.lam) == (GameTracing.TraceStuck, :landing, 1)
    end
end

@testset "the solvers return the trace's failure when it runs out of steps" begin
    gains = NormalFormGains(map(table -> Float32.(table), LOGITNASH_GAMES.prisoners_dilemma))
    settings = BifurcationKitSettings{Float32}(; max_steps = 1)
    # A one-step budget ends a stalled step as a stall (TraceStuck), a moving one as NoConvergence, as on main.
    spent = (GameTracing.NoConvergence, GameTracing.TraceStuck)
    @test unwrap_error(logit_solve_lambda(gains, 100; settings)).cause in spent
    for failure in unwrap_error.((logit_solve_branch(gains; settings), logit_solve(gains; settings)))
        @test failure.cause in spent
        # The trace's own failure, not the one logit_solve_branch builds when the trace ends unsettled.
        @test !haskey(failure.context, :maximum_precision)
    end
end

@testset "the solvers refuse Gambit's invalid arguments" begin
    gains = NormalFormGains(map(table -> Float32.(table), LOGITNASH_GAMES.matching_pennies))
    @test_throws ArgumentError logit_solve(gains; maxregret = 0)
    @test_throws ArgumentError logit_solve(gains; first_step = 0)
    @test_throws ArgumentError logit_solve_lambda(gains, 1; max_accel = 0.5f0)
    # A wider cap can round to 0 or Inf in Float32.
    huge = floatmax(widen(Float32))
    for (_, settings) in engine_settings(Float32)
        for maximum_precision in (-1, 0, NaN32, nextfloat(zero(widen(Float32))), huge)
            @test_throws ArgumentError logit_solve_branch(gains; maximum_precision, maxregret = 10, settings)
        end
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
    for distance in instances(StepDistance)
        settings = BifurcationKitSettings{element}(step_metric = TraceGeometry{element}(; distance))
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
        settings = BifurcationKitSettings{element}(step_metric = TraceGeometry{element}(; distance))
        @test_throws ArgumentError trace_endpoint(LogitGame(pass, target), 2, settings)
    end
    settings = BifurcationKitSettings{element}(step_metric = TraceGeometry{element}(distance = FisherLogPrecisionDistance))
    # To first order the logits are the target times the gains at even odds: (8 + 0) / 2 - (7 + 5) / 2 = -2 for each player.
    @test unwrap(trace_endpoint(LogitGame(pass, tiny), 2, settings)).logits ≈ fill(-2 * tiny, 2) rtol = 1.0f-3
end

@testset "PathFollowing's first step is finite and under max_step ($element)" for element in (Float32, widen(Float32))
    gains = NormalFormGains((element[0, 1],))
    capped = unwrap(GameTracing.path_init(gains, PathFollowingSettings{element}(; max_step = element(0.01f0))))
    @test capped.step_length == element(0.01f0)
    # The payoff range's inverse overflows.
    flat = NormalFormGains((element[0, nextfloat(zero(element))],))
    @test GameTracing.opening_step(flat, element) == 1
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
