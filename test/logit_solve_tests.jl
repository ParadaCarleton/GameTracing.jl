using ErrorTypes: is_error, unwrap, unwrap_error
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

# A fresh draw of a CI run: its branch folds (λ peaks near 2.72) before it meets maxregret. Its λ-fixed map is singular
# at the fold, and Gambit's `logit_solve_lambda` returns the first crossing of the end's λ, so neither check holds there.
const FOLDING_BRANCH_GAME = (
    Float32[-1.7090197f0 3.4388857f0 4.108635f0; 1.6385761f0 -2.7659307f0 5.6244125f0; -2.6847887f0 -0.8351383f0 -6.3075285f0],
    Float32[-0.11561966f0 3.8349252f0 4.1745343f0; 1.9479157f0 -2.412167f0 -1.7825432f0; -5.1971507f0 3.1131897f0 -5.5004435f0],
)

# The (3, 3, 2) game of `3 .* randn(Xoshiro(152), Float32, (3, 3, 2))` per player. A long early PathFollowing step lands
# where its oriented tangent runs back along the branch; without the orientation check the trace ends at another
# equilibrium, 0.9995 away in probability from Gambit's pure one at λ ≈ 3.8.
const REVERSAL_GAME = (
    cat(
        Float32[-0.5337934f0 -3.4574738f0 -0.31516188f0; 1.2419287f0 0.6020585f0 -0.45870376f0; 1.6466057f0 -1.8631278f0 -0.5691585f0],
        Float32[-0.11642508f0 1.8486631f0 1.6131696f0; -4.5490217f0 -1.7635972f0 -4.980587f0; 1.3529017f0 -3.944703f0 -3.9721947f0];
        dims = 3,
    ),
    cat(
        Float32[5.1817837f0 4.2066636f0 -1.6429582f0; 0.27198213f0 3.56285f0 -2.5637565f0; 1.1680062f0 -2.4406242f0 -2.0018725f0],
        Float32[-3.496707f0 0.9063864f0 -1.136579f0; 0.5411101f0 0.86526525f0 -0.27381507f0; 7.237438f0 -3.9687374f0 2.3237944f0];
        dims = 3,
    ),
    cat(
        Float32[-0.90711546f0 1.2572101f0 2.234933f0; 0.7166976f0 -3.4552052f0 -0.39850372f0; 1.7495403f0 -2.8394732f0 -3.8426185f0],
        Float32[-0.48643172f0 5.1313787f0 0.56367534f0; 0.40493193f0 4.1448507f0 0.9289435f0; 4.757027f0 0.33046308f0 -0.8036462f0];
        dims = 3,
    ),
)

"""The random games of the branch testset, one per strategy count in `((3, 3), (2, 2, 2))`, drawn from `rng`."""
function branch_games(rng::AbstractRNG)::AbstractVector
    return [ntuple(_ -> 3 .* randn(rng, Float32, counts), length(counts)) for counts in ((3, 3), (2, 2, 2))]
end

@testset "logit_solve_branch: each point solves the logit map at its precision, and the last meets maxregret ($engine)" for (engine, settings) in engine_settings(Float32)
    stuck_games = branch_games(Xoshiro(STUCK_BRANCH_SEED))
    # The seed's first payoff, as the failing run logged it.
    @test first(first(first(stuck_games))) == -0.4899592f0
    for payoffs in (stuck_games..., FOLDING_BRANCH_GAME, branch_games(fresh_rng())...)
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
        # A fold leaves the λ-fixed checks below without a unique answer (`FOLDING_BRANCH_GAME`), even past its peak.
        folded = !issorted([profile.lam for profile in branch])
        # A fresh draw may fold too; the fixed game must.
        @test folded || payoffs !== FOLDING_BRANCH_GAME
        if folded
            continue
        end
        for profile in branch
            @test settled_logits(gains, profile.logits, profile.lam, TraceSettings{eltype(profile.logits)}())
        end
        # The last point is on Gambit's branch, and logit_solve_lambda finds it at its precision.
        @test largest_gap(last(branch).profile, gambit_strategies(payoffs, last(branch).lam).strategies) <
            profile_bound(last(branch)) + GAMBIT_PROBABILITY_TOLERANCE
        again = only(unwrap(logit_solve_lambda(gains, solved.lam; settings)))
        @test largest_gap(again.profile, solved.profile) < profile_bound(again) + profile_bound(solved)
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
    # A first step a hundred times past its nominal distance is refused, and no refusal is allowed.
    for distance in (StrategyDistance{element}(; nominal = 1.0f-2), LogitDistance{element}(; nominal = 1.0f-2))
        refused = unwrap_error(logit_solve(gains; settings = PathFollowingSettings{element}(; first_step = 100, distance, max_refusals = 0)))
        @test (refused.cause, refused.context.reason, refused.context.refusals) == (GameTracing.TraceStuck, :refusals, 1)
    end
    # A distance in another element type is refused.
    @test_throws MethodError PathFollowingSettings{element}(; distance = StrategyDistance{BigFloat}())
    # 2^10 / eps ulps resolve only the move off precision zero, so the second step is unresolved.
    unresolved = unwrap_error(logit_solve(gains; maxregret = 1.0f-6, settings = PathFollowingSettings{element}(; resolution_ulps = round(Int, 2^10 / eps(element)))))
    @test (unresolved.cause, unresolved.context.reason, unresolved.context.steps) == (GameTracing.TraceStuck, :unresolved_step, 1)
    poisoned = NormalFormGains((element[NaN32 0; 0 0], element[0 0; 0 0]))
    @test unwrap_error(logit_solve(poisoned; settings = PathFollowingSettings{element}())).cause == GameTracing.NotFinite
    @test unwrap_error(logit_solve_lambda(poisoned, 1; settings = PathFollowingSettings{element}())).context.reason == :NotFiniteStart
    # Step control without its turn and contraction terms still solves.
    plain = PathFollowingSettings{element}(; nominal_contraction = nothing, nominal_turn = nothing)
    @test GameTracing.max_regret(gains, unwrap(logit_solve(gains; settings = plain)).logits) <= sqrt(eps(element)) * GameTracing.payoff_range(gains)
    @test_throws ArgumentError logit_solve_lambda(gains, -1; settings = PathFollowingSettings{element}())
end

@testset "PathFollowing keeps Gambit's branch where a long step ran back along it ($element)" for element in (Float32, widen(Float32))
    gains = NormalFormGains(map(table -> element.(table), REVERSAL_GAME))
    checked = unwrap(logit_solve(gains; maxregret = 1.0f-4, settings = PathFollowingSettings{element}()))
    oracle = gambit_strategies(REVERSAL_GAME, checked.lam)
    @test largest_gap(checked.profile, oracle.strategies) < profile_bound(checked) + GAMBIT_PROBABILITY_TOLERANCE
    # The control: the same trace without the check ends elsewhere.
    unchecked = unwrap(logit_solve(gains; maxregret = 1.0f-4, settings = PathFollowingSettings{element}(; orientation = nothing)))
    @test largest_gap(unchecked.profile, checked.profile) > 1 // 2
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
end

@testset "a strategy move within its logits' settle_ulps settles; one past it by the tolerance does not ($element)" for element in (Float32, widen(Float32))
    distance = StrategyDistance{element}()
    tolerance, ulps = distance.tolerance, distance.settle_ulps
    settled = GameTracing.settle_test(distance, GameTracing.FisherRaoChange((1:2,)))
    # Strategies 2 and 3 share a large logit; the first has left the support. Their rounding alone passes the tolerance.
    large = 64 * tolerance / eps(element)
    rounded = large + 4 * eps(large)
    @test GameTracing.FisherRaoChange((1:2,))(element[large, large, 1], element[rounded, large, 1]) > tolerance
    @test settled(element[large, large, 1], element[rounded, large, 1], element[])
    # A logit gap of four allowances moves the strategy two, one allowance past the test.
    beyond = large + 4 * ulps * eps(element) * large
    @test !settled(element[large, large, 1], element[beyond, large, 1], element[])
    # Small logits have no allowance to speak of: a move of four tolerances stays past it.
    @test !settled(element[0, 0, 1], element[4 * tolerance, 0, 1], element[])
    @test !settled(element[NaN32, 0, 1], element[0, 0, 1], element[])
end

@testset "WithinReach passes a correction within the settle tolerance, however short the step ($element)" for element in (Float32, widen(Float32))
    within = GameTracing.WithinReach(element(2), element(1.0f-3), GameTracing.logit_change)
    start, predicted = element[0, 1], element[1.0f-6, 1]
    @test within(start, predicted, element[5.0f-4, 1])
    @test !within(start, predicted, element[2.0f-3, 1])
    # Past the tolerance, the predictor's own move sets the reach.
    @test within(start, element[1, 1], element[2.5f0, 1])
    @test !within(start, element[1, 1], element[4, 1])
    @test within(start, start, element[4, 1])
end

@testset "StrategyDistance turns a fold whose start settled only to its tolerance" begin
    # The benchmark's 20x20 game from seed 6 folds back at λ ≈ 315.68. There the start's leftover error, about 1.5e-5
    # in Fisher–Rao, outweighed reach times every short predictor move, and the trace stopped TraceStuck.
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

"""The Bernoulli Fisher weight `σ(z)σ(-z)` of a logit `z`."""
fisher_weight(logit::Real)::Real = exp(-abs(logit)) / (1 + exp(-abs(logit)))^2

@testset "PathFollowing's geometry weighs each payoff gap and the precision ($element)" for element in (Float32, widen(Float32))
    # Payoff gaps (1, -2) at precision 3, so logits (3, -6).
    point = element[1, -2, 3]
    weights = zeros(element, 3)
    log_precision = TraceGeometry{element}(; distance = FisherLogPrecisionDistance, state_weight = 1 // 2)
    GameTracing.GeometryWeights(log_precision)(weights, point)
    @test weights ≈ element[fisher_weight(3) / 2, fisher_weight(-6) / 2, 1 // 16]
    GameTracing.GeometryWeights(TraceGeometry{element}(; distance = FisherDistance, state_weight = 1 // 2))(weights, point)
    @test weights[3] == 1
    GameTracing.GeometryWeights(TraceGeometry{element}(; distance = SummedDistance, state_weight = 1 // 2))(weights, point)
    @test weights == element[1 // 2, 1 // 2, 1]
    gains = NormalFormGains(map(table -> element.(table), LOGITNASH_GAMES.matching_pennies))
    solved = unwrap(logit_solve(gains; settings = PathFollowingSettings{element}(; geometry = log_precision)))
    @test GameTracing.max_regret(gains, solved.logits) <= sqrt(eps(element)) * GameTracing.payoff_range(gains)
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
    settings = TraceSettings{Float32}(; max_steps = 1)
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
end

@testset "the solvers return profiles in the payoffs' element type and refuse settings in another" begin
    for element in (Float32, float(Int))
        gains = NormalFormGains(map(table -> element.(table), LOGITNASH_GAMES.prisoners_dilemma))
        @test eltype(gains) == element
        @test unwrap(logit_solve(gains)).lam isa element
        @test eltype(unwrap(logit_solve(gains)).logits) == element
        @test eltype(only(unwrap(logit_solve_lambda(gains, 2))).logits) == element
        other = element == Float32 ? float(Int) : Float32
        for settings in (TraceSettings{other}(), PathFollowingSettings{other}())
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
    @test_throws ArgumentError logit_solve(integer; settings = TraceSettings{Float32}())
end

@testset "LogitQREProfile computes its profile from its logits and takes no profile" begin
    gains = NormalFormGains(map(table -> Float32.(table), LOGITNASH_GAMES.matching_pennies))
    profile = LogitQREProfile(gains, 1.0f0, Float32[0, 0])
    @test profile.profile == GameTracing.mixed_strategies(gains, profile.logits)
    @test_throws MethodError LogitQREProfile(1.0f0, profile.profile, Float32[3, -3])
end
