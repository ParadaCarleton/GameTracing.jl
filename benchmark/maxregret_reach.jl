# Whether `logit_solve` reaches its default `maxregret` (`default_maxregret`) on the test games, on each engine in
# Float32 and Float64, and the precision it needs. Games: LogitNash's two-action games but the exact pitchforks, the
# fixture games, and random games like the branch testset's ((3, 3) and (2, 2, 2), 3 randn) and two-action games of
# 2 to 5 players. The cap is `CAP`, far past the default `maximum_precision`, so a row shows the precision it took.
# Usage: julia --project=benchmark benchmark/maxregret_reach.jl [random games per size]
using GameTracing
using GameTracing: max_regret, payoff_range
using ErrorTypes: is_error, unwrap, unwrap_error
using Printf: @printf
using Random: AbstractRNG, Xoshiro

include(joinpath(@__DIR__, "..", "test", "games.jl"))
include(joinpath(@__DIR__, "..", "test", "fixtures.jl"))
include(joinpath(@__DIR__, "..", "test", "causes.jl"))

const WIDE = widen(Float32)
const CAP = 1.0f12
const RANDOM_SIZES = ((3, 3), (2, 2, 2), (2, 2), (2, 2, 2, 2), (2, 2, 2, 2, 2))

"""Named payoff tables: the fixture games, then `count` random draws of each of `RANDOM_SIZES` from `rng`."""
function reach_games(rng::AbstractRNG, count::Integer)::Vector{Pair{String, Tuple}}
    games = Pair{String, Tuple}[string(name) => payoffs for (name, payoffs) in pairs(LOGITNASH_GAMES) if !(name in PITCHFORK_GAMES)]
    fixtures = (; OVERSHOT_GAME, FOLDED_GAMBIT_GAME, FOLD_PAST_ONE_GAME, SLOW_REGRET_GAME, GAMBIT_LEAVES_GAME, UNFOLDED_PITCHFORK_GAME, REVERSAL_GAME)
    append!(games, [string(name) => payoffs for (name, payoffs) in pairs(fixtures)])
    for counts in RANDOM_SIZES, draw in 1:count
        push!(games, "random$(counts)_$(draw)" => ntuple(_ -> 3 .* randn(rng, WIDE, counts), length(counts)))
    end
    return games
end

"""One `logit_solve` of `payoffs` in `E` on `settings` to `CAP`: its cause or `ok`, its precision and its regret share."""
function reach(payoffs::Tuple, settings::GameTracing.EngineSettings{E})::NamedTuple where {E <: Real}
    gains = NormalFormGains(map(table -> E.(table), payoffs))
    result = logit_solve(gains; settings, maximum_precision = CAP)
    is_error(result) && return (; cause = cause_label(unwrap_error(result).cause), precision = WIDE(unwrap_error(result).stop.precision), share = WIDE(NaN32))
    solved = unwrap(result)
    return (; cause = "ok", precision = WIDE(solved.lam), share = WIDE(max_regret(gains, solved.logits) / payoff_range(gains)))
end

function main(count::Integer)::Nothing
    @printf("element engine game cause precision regret_share maxregret\n")
    for element in (Float32, WIDE), (engine, settings) in (("bk", BifurcationKitSettings{element}()), ("pf", PathFollowingSettings{element}()))
        needed, failures = zero(WIDE), 0
        for (name, payoffs) in reach_games(Xoshiro(20261120), count)
            row = reach(payoffs, settings)
            @printf("%s %s %s %s %.3e %.3e %.2e\n", element, engine, name, row.cause, row.precision, row.share, default_maxregret(element))
            if row.cause == "ok"
                needed = max(needed, row.precision)
            else
                failures += 1
            end
        end
        @printf("SUMMARY %s %s largest_precision %.3e failures %d\n", element, engine, needed, failures)
    end
    return nothing
end

main(isempty(ARGS) ? 20 : parse(Int, ARGS[1]))
