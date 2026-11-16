# PathFollowingSettings on hard games: random normal-form games, common-interest games whose players' payoffs correlate
# at 0.9 and 0.97 (GAMUT's covariant games) and the test fixtures. Each endpoint is checked against a cautious trace of
# its game; one over 0.05 away in probability left its branch. Prints a summary per class and arm, and every miss.
# Usage: julia --project=benchmark benchmark/hard_games.jl <out.csv>
using GameTracing
using GameTracing: NormalFormGains, PathFollowingSettings, StrategyDistance, logit_solve_branch
using ErrorTypes: is_error, unwrap, unwrap_error
using CSV: CSV
using LinearAlgebra: dot, norm
using Printf: @printf
using Random: Xoshiro

include(joinpath(@__DIR__, "..", "test", "games.jl"))
include(joinpath(@__DIR__, "..", "test", "fixtures.jl"))

const NORMAL_SIZES = ((2, 2), (3, 3), (5, 5), (10, 10), (20, 20), (2, 2, 2), (3, 3, 3), (5, 5, 5), (2, 2, 2, 2), (3, 3, 3, 3), (2, 2, 2, 2, 2))
const COMMON_SIZES = ((5, 5), (10, 10), (20, 20), (4, 4, 4), (3, 3, 3, 3))
const WIDE = widen(Float32)
const MAXREGRET = 1.0f-6
const OFF_BRANCH = 0.05f0
const ARMS = (
    ("default", (;)), ("growth 4", (; max_step_growth = 4)), ("turn 0.3", (; target_turn = 0.3f0)), ("turn off", (; target_turn = nothing)),
)
const REFERENCE = (; target_turn = 0.1f0, max_step_growth = 1.5f0, distance = StrategyDistance{WIDE}(; target_correction = 0.05f0))

"""Standard normal payoffs over `counts` strategies, one table per player."""
function random_game(counts::Tuple, seed::Integer)::Tuple
    rng = Xoshiro(seed)
    return ntuple(_ -> randn(rng, counts), length(counts))
end

"""Payoffs whose players' entries correlate at `correlation` across profiles (GAMUT's covariant game)."""
function covariant_game(counts::Tuple, seed::Integer, correlation::Real)::Tuple
    rng = Xoshiro(seed)
    common = randn(rng, counts)
    return ntuple(_ -> sqrt(correlation) .* common .+ sqrt(1 - correlation) .* randn(rng, counts), length(counts))
end

"""Every payoff table of `payoffs` in `WIDE`."""
wide(payoffs::Tuple)::Tuple = map(table -> WIDE.(table), payoffs)

const GAMES = vcat(
    [("normal", "$(join(counts, "x")) s$seed", random_game(counts, seed)) for counts in NORMAL_SIZES for seed in 11:20],
    [("common 0.9", "$(join(counts, "x")) s$seed", covariant_game(counts, seed, 9 // 10)) for counts in COMMON_SIZES for seed in 31:90],
    [("common 0.97", "$(join(counts, "x")) s$seed", covariant_game(counts, 100 + seed, 97 // 100)) for counts in COMMON_SIZES[[1, 2, 4]] for seed in 21:60],
    [("fixture", String(name), wide(LOGITNASH_GAMES[name])) for name in keys(LOGITNASH_GAMES)],
    [
        ("fixture", name, wide(game)) for (name, game) in (
                ("overshot", OVERSHOT_GAME), ("folded_gambit", FOLDED_GAMBIT_GAME), ("gambit_leaves", GAMBIT_LEAVES_GAME),
                ("unfolded_pitchfork", UNFOLDED_PITCHFORK_GAME), ("reversal", REVERSAL_GAME),
            )
    ],
)

"""Every accepted point of a trace of `gains` under `overrides` as a flat profile, or the failure's cause and reason."""
function traced(gains::NormalFormGains, overrides::NamedTuple)::NamedTuple
    settings = PathFollowingSettings{WIDE}(; max_steps = 20_000, overrides...)
    branch = logit_solve_branch(gains; maxregret = MAXREGRET, settings)
    if is_error(branch)
        failure = unwrap_error(branch)
        return (; outcome = "$(failure.cause) $(get(failure.context, :reason, ""))", points = Vector{WIDE}[], lam = WIDE(NaN32))
    end
    points = unwrap(branch)
    return (; outcome = "Ok", points = [reduce(vcat, point.profile) for point in points], lam = last(points).lam)
end

"""The distance from `point` to the polyline through `points`, or to their one point."""
function polyline_distance(point::AbstractVector, points::AbstractVector)::WIDE
    best = norm(point - last(points))
    for (start, stop) in zip(points[1:(end - 1)], points[2:end])
        along = stop - start
        share = clamp(dot(point - start, along) / max(dot(along, along), eps(WIDE)), 0, 1)
        best = min(best, norm(point - (start + share * along)))
    end
    return best
end

function main(path::AbstractString)::Nothing
    rows = NamedTuple[]
    for (class, name, payoffs) in GAMES
        gains = NormalFormGains(payoffs)
        reference = traced(gains, REFERENCE)
        for (arm, overrides) in ARMS
            run = traced(gains, overrides)
            settled = run.outcome == "Ok" && reference.outcome == "Ok"
            off = settled ? polyline_distance(last(run.points), reference.points) : WIDE(NaN32)
            push!(rows, (; class, name, arm, run.outcome, steps = length(run.points), run.lam, off, reference = reference.outcome))
        end
    end
    CSV.write(path, rows)
    @printf("maxregret %.0e; off: endpoints over %.2f from the reference branch\n", MAXREGRET, OFF_BRANCH)
    @printf("%-11s %-9s %6s %6s %7s %4s\n", "class", "arm", "games", "solved", "steps", "off")
    for class in unique(first.(GAMES)), (arm, _) in ARMS
        group = filter(row -> row.class == class && row.arm == arm, rows)
        @printf(
            "%-11s %-9s %6d %6d %7d %4d\n", class, arm, length(group), count(row -> row.outcome == "Ok", group),
            sum(row -> row.steps, group), count(row -> row.off > OFF_BRANCH, group)
        )
    end
    println("\nfailures and endpoints off the reference branch:")
    for row in rows
        if row.outcome != "Ok" || row.reference != "Ok" || row.off > OFF_BRANCH
            @printf("  %-11s %-12s %-9s %-30s off %.3f λ %.4g reference %s\n", row.class, row.name, row.arm, row.outcome, row.off, row.lam, row.reference)
        end
    end
    return nothing
end

main(ARGS[1])
