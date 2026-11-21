# The two engines' `trace_endpoint`, BifurcationKit and PathFollowing at their default settings, on random games of 2 to
# 9 players with two actions each, in Float32. Mode `timing`: warm wall time, every arm of a game in one process,
# interleaved, on two passes: the generic test pass `NormalFormPass` (test/games.jl: in place, vector logit ranges, no
# work arrays) and `NormalFormGains` (work arrays, tuple ranges). Mode `branches`: each endpoint against a fine-step
# Float64 reference (`branch_reference`), as the largest strategy error and whether it left the branch.
# Usage: julia --project=benchmark benchmark/endpoint_engines.jl timing|branches [games per size] [target precision] [seconds per timing]
using GameTracing
using ErrorTypes: is_error, unwrap, unwrap_error
using Chairmarks: @be
using Printf: @printf
using Random: AbstractRNG, Xoshiro
using Statistics: median

include(joinpath(@__DIR__, "..", "test", "games.jl"))

const PLAYERS = 2:9
const ELEMENT = Float32
# The timings' and the reference's float type.
const WIDE = widen(Float32)

"""`players` standard normal payoff tables of two actions per player in `ELEMENT`, from `rng`."""
function binary_payoffs(rng::AbstractRNG, players::Integer)::Tuple
    return ntuple(_ -> ELEMENT.(randn(rng, ntuple(Returns(2), players))), players)
end

"""The median seconds of a warm `trace_endpoint` of `game` on `settings`, sampled for at most `seconds`."""
function endpoint_seconds(game::LogitGame, width::Integer, settings::GameTracing.EngineSettings, seconds::Real)::WIDE
    trace_endpoint(game, width, settings)
    return median(@be(trace_endpoint($game, $width, $settings), seconds = seconds)).time
end

"""Each engine's outcome on `game`: its endpoint's logits and steps, or its failure's cause."""
function outcome(game::LogitGame, width::Integer, settings::GameTracing.EngineSettings)::NamedTuple
    result = trace_endpoint(game, width, settings)
    is_error(result) && return (; logits = fill(ELEMENT(NaN32), width), steps = -1, cause = string(unwrap_error(result).cause))
    return (; unwrap(result).logits, unwrap(result).steps, cause = "ok")
end

function main(games::Integer, target::ELEMENT, seconds::Real)::Nothing
    engines = (; bifurcation_kit = BifurcationKitSettings{ELEMENT}(), path_following = PathFollowingSettings{ELEMENT}())
    @printf("players pass game bk_ms pf_ms ratio bk_steps pf_steps bk_cause pf_cause logit_gap\n")
    for players in PLAYERS
        rng = Xoshiro(1000 + players)
        ratios = Dict{Symbol, Vector{WIDE}}()
        for game_index in 1:games
            payoffs = binary_payoffs(rng, players)
            for (name, pass) in ((:NormalFormPass, NormalFormPass(payoffs)), (:NormalFormGains, NormalFormGains(payoffs)))
                game = LogitGame(pass, target)
                outcomes = map(settings -> outcome(game, players, settings), engines)
                times = map(settings -> endpoint_seconds(game, players, settings, seconds), engines)
                ratio = times.bifurcation_kit / times.path_following
                push!(get!(ratios, name, WIDE[]), ratio)
                gap = maximum(abs, outcomes.bifurcation_kit.logits .- outcomes.path_following.logits)
                @printf(
                    "%d %s %d %.3f %.3f %.2f %d %d %s %s %.2e\n", players, name, game_index, 1000 * times.bifurcation_kit,
                    1000 * times.path_following, ratio, outcomes.bifurcation_kit.steps, outcomes.path_following.steps,
                    outcomes.bifurcation_kit.cause, outcomes.path_following.cause, gap,
                )
            end
        end
        for (name, collected) in ratios
            @printf("SUMMARY players %d %s median bk/pf %.2f min %.2f max %.2f\n", players, name, median(collected), minimum(collected), maximum(collected))
        end
    end
    return nothing
end

# A strategy this far from the reference's is off its branch: 100 times Gambit's corrector tolerance in probability.
const BRANCH_TOLERANCE = 1.0f-2

"""
The largest probability error of a `NormalFormGains` trace's logits against the reference's `first_choices`, or Inf32
for a failed trace. Its logit `z` is the second action's against the first, so the first has probability `1 / (1 + eᶻ)`.
"""
function strategy_error(traced::NamedTuple, first_choices::AbstractVector)::WIDE
    traced.cause == "ok" || return Inf32
    return maximum(abs, inv.(1 .+ exp.(WIDE.(traced.logits))) .- first_choices)
end

function branches(games::Integer, target::ELEMENT)::Nothing
    engines = (; bifurcation_kit = BifurcationKitSettings{ELEMENT}(), path_following = PathFollowingSettings{ELEMENT}())
    @printf("players game reached bk_error pf_error bk_cause pf_cause crossings\n")
    for players in PLAYERS
        rng = Xoshiro(2000 + players)
        off_branch = (; bifurcation_kit = 0, path_following = 0)
        largest = (; bifurcation_kit = zero(WIDE), path_following = zero(WIDE))
        for game_index in 1:games
            payoffs = binary_payoffs(rng, players)
            pass = NormalFormGains(payoffs)
            reference = branch_reference(NormalFormPass(map(table -> WIDE.(table), payoffs)), players, target)
            outcomes = map(settings -> outcome(LogitGame(pass, target), players, settings), engines)
            errors = map(traced -> strategy_error(traced, reference.first_choices), outcomes)
            @printf(
                "%d %d %s %.2e %.2e %s %s %d\n", players, game_index, reference.reached, errors.bifurcation_kit, errors.path_following,
                outcomes.bifurcation_kit.cause, outcomes.path_following.cause, length(reference.crossings),
            )
            # A game whose reference crosses a singular point has no single branch past it.
            (reference.reached && isempty(reference.crossings)) || continue
            off_branch = map((count, error) -> count + (error > BRANCH_TOLERANCE), off_branch, errors)
            largest = map((worst, error) -> max(worst, ifelse(isfinite(error), error, zero(WIDE))), largest, errors)
        end
        @printf(
            "SUMMARY players %d off-branch bk %d pf %d largest on-branch error bk %.2e pf %.2e\n", players, off_branch.bifurcation_kit,
            off_branch.path_following, largest.bifurcation_kit, largest.path_following,
        )
    end
    return nothing
end

const MODE = get(ARGS, 1, "timing")
const GAMES = parse(Int, get(ARGS, 2, "10"))
const TARGET = parse(ELEMENT, get(ARGS, 3, "4"))
if MODE == "timing"
    main(GAMES, TARGET, parse(WIDE, get(ARGS, 4, "1")))
else
    branches(GAMES, TARGET)
end
