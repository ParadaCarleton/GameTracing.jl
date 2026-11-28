# Gambit's logit QRE is the test oracle. runtests.jl pins its version before any worker loads PythonCall.
using PythonCall: pyconvert, pyimport

"""
Gambit's step settings, coarse to fine (`gambit_strategies`): its defaults, then each a tenth of the first step and of
the growth past one. Near a fold, each rung's steps are about a tenth of the last one's.
"""
const GAMBIT_STEP_LADDER = (
    (; first_step = 3 / float(100), max_accel = 11 / float(10)),
    (; first_step = 3 / float(1000), max_accel = 101 / float(100)),
    (; first_step = 3 / float(10000), max_accel = 1001 / float(1000)),
)

"""
    gambit_profile(payoffs, precision, steps)

`(; precision, strategies)` from `pygambit.qre.logit_solve_lambda` on the normal-form game `payoffs` (one table per
player, axis `k` player `k`'s action) at `precision`, with Gambit's `steps` `(; first_step, max_accel)`. `strategies`
holds each player's mixed strategy and `precision` the `λ` Gambit reports, both in `precision`'s type. Gambit's
`π ∝ exp(λ u)` on the raw payoffs is the logit map `z = λ G`, so its `λ` is the trace's precision.
"""
function gambit_profile(payoffs::Tuple, precision::E, steps::NamedTuple)::NamedTuple where {E <: Real}
    gambit, numpy = pyimport("pygambit"), pyimport("numpy")
    game = gambit.Game.from_arrays((numpy.asarray(float.(table)) for table in payoffs)...)
    lam = [float(precision)]
    profile = first(gambit.qre.logit_solve_lambda(game; lam, first_step = float(steps.first_step), max_accel = float(steps.max_accel)))
    strategies = Tuple([pyconvert(E, profile.profile[strategy]) for strategy in player.strategies] for player in game.players)
    return (; precision = pyconvert(E, profile.lam), strategies)
end

"""
Whether two of Gambit's profiles at one precision lie on one crossing: each lies within `GAMBIT_PROBABILITY_TOLERANCE` of
the branch, so two on one crossing differ by at most twice that in any probability.
"""
function same_crossing(left::NamedTuple, right::NamedTuple)::Bool
    return largest_gap(left.strategies, right.strategies) <= 2 * GAMBIT_PROBABILITY_TOLERANCE
end

"""
    gambit_strategies(payoffs, precision)

Gambit's logit QRE of `payoffs` at the first crossing of `precision` (`gambit_profile`): the finest
`GAMBIT_STEP_LADDER` rung's profile, where the rung before it lands on the same crossing (`same_crossing`). Else the
step size has not converged, no rung is a sound oracle, and it throws an `ErrorException`.

Gambit finds a crossing only between consecutive points, so a step that rounds a fold lands on a later crossing
(`FOLD_PAST_ONE_GAME`, `FOLDED_GAMBIT_GAME`, `FOLD_TIP_PAST_TARGET_GAME`). Two coarse rungs can round one tip alike, so
only the finest pair decides. Over 200 random two-action games the last two rungs agreed on all, each within 3e-8 of
`branch_reference`. A fourth rung, a tenth of the third, took 0.3 to 2.8 s a game, so the ladder stops at three.
"""
function gambit_strategies(payoffs::Tuple, precision::E)::NamedTuple where {E <: Real}
    (finer, finest) = [gambit_profile(payoffs, precision, steps) for steps in GAMBIT_STEP_LADDER[(end - 1):end]]
    same_crossing(finer, finest) || error("Gambit's two finest steps land on different crossings of precision $precision: $finer, $finest")
    return finest
end

"""
    gambit_branch(payoffs, precision)
    gambit_branch(payoffs, precision, steps)

`(; precision, first_choices)` of Gambit's logit QRE of the two-action game `payoffs` at `precision`: each player's
probability of its first action. Without `steps` it is the first crossing (`gambit_strategies`); with them, Gambit's
answer at those steps alone (`gambit_profile`).
"""
function gambit_branch(payoffs::Tuple, precision::E)::NamedTuple{(:precision, :first_choices), Tuple{E, Vector{E}}} where {E <: Real}
    return first_choice_profile(gambit_strategies(payoffs, precision))
end

function gambit_branch(payoffs::Tuple, precision::E, steps::NamedTuple)::NamedTuple{(:precision, :first_choices), Tuple{E, Vector{E}}} where {E <: Real}
    return first_choice_profile(gambit_profile(payoffs, precision, steps))
end

"""Gambit's default `max_accel` in `Float32`, read from `pygambit.qre.logit_solve_lambda`'s signature."""
function gambit_max_accel()::Float32
    parameters = pyimport("inspect").signature(pyimport("pygambit").qre.logit_solve_lambda).parameters
    return pyconvert(Float32, parameters["max_accel"].default)
end

"""`profile`'s precision and each player's probability of its first action."""
function first_choice_profile(profile::NamedTuple)::NamedTuple
    return (; profile.precision, first_choices = first.(collect(profile.strategies)))
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
