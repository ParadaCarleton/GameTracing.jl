using CondaPkg: CondaPkg

# Gambit's logit QRE is the test oracle, at the version checked by hand against a 50-digit trace.
# The pin goes into the test environment, not a package-level CondaPkg.toml that every user of GameTracing would read.
const PYGAMBIT_VERSION = "16.7.0"
CondaPkg.add("python"; version = "=3.12", resolve = false)
CondaPkg.add_pip("pygambit"; version = "==$PYGAMBIT_VERSION")
# PythonCall resolves the environment and starts Python when it loads, so it loads after the pins.
using PythonCall: pyconvert, pyimport

"""
    gambit_branch(payoffs, precision; first_step = 3 // 100, max_accel = 11 // 10)

`(; precision, first_choices)` of Gambit's logit QRE branch of the two-action game `payoffs` (one table per player, axis
`k` player `k`'s action, first action first), from `pygambit.qre.logit_solve_lambda` at `precision`. Gambit's
`π ∝ exp(λ u)` on the raw payoffs is the logit map `z = λ G` with `G` the first action's payoff gain, so Gambit's `λ` is
the trace's precision (the test "Gambit's λ is the trace's precision"). `precision` is the `λ` Gambit reports reaching; both come back in `precision`'s type, the compared trace's.
`first_step` and `max_accel` are Gambit's first step length and largest step growth, its defaults by default.
"""
function gambit_branch(
        payoffs::Tuple,
        precision::E;
        first_step::Real = 3 // 100,
        max_accel::Real = 11 // 10,
    )::NamedTuple{(:precision, :first_choices), Tuple{E, Vector{E}}} where {E <: Real}
    gambit, numpy = pyimport("pygambit"), pyimport("numpy")
    game = gambit.Game.from_arrays((numpy.asarray(float.(table)) for table in payoffs)...)
    profile = first(
        gambit.qre.logit_solve_lambda(
            game;
            lam = [float(precision)],
            first_step = float(first_step),
            max_accel = float(max_accel),
        )
    )
    first_choices = [pyconvert(E, profile.profile[first(player.strategies)]) for player in game.players]
    return (; precision = pyconvert(E, profile.lam), first_choices)
end

"""Gambit's default `max_accel` in `Float32`, read from `pygambit.qre.logit_solve_lambda`'s signature."""
function gambit_max_accel()::Float32
    parameters = pyimport("inspect").signature(pyimport("pygambit").qre.logit_solve_lambda).parameters
    return pyconvert(Float32, parameters["max_accel"].default)
end

"""Whether the branch points' precisions `lambdas` decrease anywhere before they first reach `target`."""
function folds_before(lambdas::AbstractVector{E}, target::Real)::Bool where {E <: Real}
    rising = collect(Iterators.takewhile(<(target), lambdas))
    return any(splat(>), zip(rising, Iterators.drop(rising, 1)))
end

"""
Gambit's step settings, coarse to fine (`first_crossing_steps`): its defaults, then each a tenth of the first step and
of the growth past one. Near a fold, each rung's steps are about a tenth of the last one's.
"""
const GAMBIT_STEP_LADDER = (
    (; first_step = 3 / float(100), max_accel = 11 / float(10)),
    (; first_step = 3 / float(1000), max_accel = 101 / float(100)),
    (; first_step = 3 / float(10000), max_accel = 1001 / float(1000)),
)

"""
The `λ` of each point of `pygambit.qre.logit_solve_branch` on `payoffs` at `steps` (a `GAMBIT_STEP_LADDER` rung) to
regret `maxregret` of the payoff range, in order, as Gambit's doubles.
"""
function gambit_regret_lambdas(payoffs::Tuple, steps::NamedTuple, maxregret::Real)::Vector{<:Real}
    gambit, numpy = pyimport("pygambit"), pyimport("numpy")
    game = gambit.Game.from_arrays((numpy.asarray(float.(table)) for table in payoffs)...)
    return [pyconvert(widen(Float32), point.lam) for point in gambit.qre.logit_solve_branch(game; maxregret = float(maxregret), steps...)]
end

"""
The `λ` of each point of Gambit's branch of `payoffs` at `steps` (`gambit_regret_lambdas`), through the first point at
or past `target`. They stay Gambit's doubles: rounding to a narrower target's type could lift a point onto `target`. The
regret bound falls from `1e-3` by factors of `1000` until the branch reaches `target`: a far bound can take a million
points. A branch that ends before `target` at `1e-12` has settled at a Nash equilibrium to that bound, and all its
points come back. Past a strict equilibrium the branch converges without folding; past a degenerate one it is taken to
stay. A single point is a branch Gambit ended at its start, where the uniform profile has no regret.
"""
function gambit_branch_lambdas(payoffs::Tuple, steps::NamedTuple, target::Real)::Vector{<:Real}
    lambdas = widen(Float32)[]
    for exponent in 3:3:12
        lambdas = gambit_regret_lambdas(payoffs, steps, inv(float(10^exponent)))
        reached = findfirst(>=(target), lambdas)
        if !isnothing(reached)
            return lambdas[1:reached]
        end
    end
    return lambdas
end

"""
Whether Gambit's logit branch of `payoffs` at its default steps folds back in `λ` before it first reaches `precision`
(`folds_before`). `logit_solve_lambda` takes the same steps and finds a crossing only between consecutive points, so a
step that rounds such a fold can hide the first crossing.
"""
function gambit_folds_before(payoffs::Tuple, precision::Real)::Bool
    return folds_before(gambit_branch_lambdas(payoffs, first(GAMBIT_STEP_LADDER), precision), precision)
end

"""
The coarsest rung of `GAMBIT_STEP_LADDER` whose branch of `payoffs` reaches `precision` before it folds back
(`folds_before`). That rung's `logit_solve_lambda` returns the first crossing, except past a fold too sharp to show
between two points. Where every rung folds before `precision`, the fold peaks below it, and the finest rung stands.
The finest rung can still round a fold that peaks just past `precision`; near a fold its steps are a hundredth of
Gambit's defaults.
"""
function first_crossing_steps(payoffs::Tuple, precision::Real)::NamedTuple
    for steps in GAMBIT_STEP_LADDER
        if !folds_before(gambit_branch_lambdas(payoffs, steps, precision), precision)
            return steps
        end
    end
    return last(GAMBIT_STEP_LADDER)
end

"""
Gambit's logit QRE of the normal-form game `payoffs` at the first crossing of `precision`, from
`pygambit.qre.logit_solve_lambda`: `(; precision, strategies)`, each player's mixed strategy in `precision`'s type.
Gambit finds a crossing only between consecutive points, so a step that rounds a fold peaking past `precision` hides the
first crossing (`FOLD_PAST_ONE_GAME`). The steps come from `first_crossing_steps`.
"""
function gambit_strategies(payoffs::Tuple, precision::E)::NamedTuple where {E <: Real}
    gambit, numpy = pyimport("pygambit"), pyimport("numpy")
    game = gambit.Game.from_arrays((numpy.asarray(float.(table)) for table in payoffs)...)
    steps = first_crossing_steps(payoffs, precision)
    profile = first(gambit.qre.logit_solve_lambda(game; lam = [float(precision)], steps...))
    strategies = Tuple([pyconvert(E, profile.profile[strategy]) for strategy in player.strategies] for player in game.players)
    return (; precision = pyconvert(E, profile.lam), strategies)
end

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
