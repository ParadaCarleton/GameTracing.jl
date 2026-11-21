using CommonSolve: CommonSolve
using ErrorTypes: Err, Ok, Result, is_error, unwrap, unwrap_error
using KernelAbstractions: KernelAbstractions
using LinearSolve: LinearSolve
using Logging: @debug
using Moshi.Data: isa_variant
using PathFollowing: PathFollowing
using StaticArrays: SArray, SVector, Size, StaticVector

"""
How the corrector measures a move between two points `[w; λ]` (`corrector_measure`), with its settle tolerance and
step control's target first correction in the same units: `StrategyDistance` or `LogitDistance`.
"""
abstract type CorrectorDistance{E <: Real} end

"""
    StrategyDistance{E}(; tolerance, settle_ulps, target_correction)

Measure a move by its Fisher–Rao distance on the product of the players' simplices: the root sum of squares of each
player's distance (`FisherRaoChange`). A strategy counts by its probability, so one that has left the support stops
counting. The corrector settles once a correction moves the profile no further than `tolerance` past the distance
`settle_ulps` ulps of each logit move it (`FisherRaoSettled`). Step control's target first correction is
`target_correction`, or `nothing` to drop the term. A set target must lie above `tolerance` (`settle_tolerance`).

Gambit's tracer measures a correction by its Euclidean norm in log probabilities (path.cc). A player's Fisher–Rao move
is at most that norm over √2, with equality for two strategies at even odds (Popoviciu's inequality). So the profile's
distance is at most Gambit's norm over √2, for any number of players, and Gambit's limits divide by √2 here.
"""
struct StrategyDistance{E <: Real} <: CorrectorDistance{E}
    tolerance::E
    settle_ulps::Int
    target_correction::Union{Nothing, E}
    function StrategyDistance{E}(tolerance::E, settle_ulps::Int, target_correction::Union{Nothing, E})::StrategyDistance{E} where {E <: Real}
        return new{E}(settle_tolerance(tolerance, target_correction), settle_ulps, target_correction)
    end
end

# Defaults: half the `LogitDistance` defaults, as a logit move at even odds is twice its Fisher–Rao distance. Gambit's
# 1e-4 is 7.1e-5 here; its refusal of a correction of 0.4 is 0.28, and step control refuses past 4 targets.
function StrategyDistance{E}(;
        tolerance::Real = 5.0f-5,
        settle_ulps::Integer = 16,
        target_correction::Union{Nothing, Real} = 7 // 100,
    )::StrategyDistance{E} where {E <: Real}
    return StrategyDistance{E}(E(tolerance), Int(settle_ulps), optional_limit(E, target_correction))
end

"""
    LogitDistance{E}(; tolerance, settle_ulps, target_correction)

Measure a move by the Euclidean length of the logits' change (`logit_change`), at most √2 times Gambit's norm in log
probabilities. The corrector settles once a correction moves the logits no more than `tolerance` past `settle_ulps` ulps
of their own size (`LogitSettled`). Step control's target first
correction is `target_correction`, or `nothing` to drop the term. Do not use it to high precision when a strategy leaves the
support: its logit grows like `λ`, and its corrections, `λ` times the square of the others', hold the steps short.
A set target must lie above `tolerance` (`settle_tolerance`).
"""
struct LogitDistance{E <: Real} <: CorrectorDistance{E}
    tolerance::E
    settle_ulps::Int
    target_correction::Union{Nothing, E}
    function LogitDistance{E}(tolerance::E, settle_ulps::Int, target_correction::Union{Nothing, E})::LogitDistance{E} where {E <: Real}
        return new{E}(settle_tolerance(tolerance, target_correction), settle_ulps, target_correction)
    end
end

"""
    LogitSettled(tolerance, rounding)

PathFollowing's `converged` over points `[w; λ]`: the last correction's logit move, times Newton's `scale`, is at
most `tolerance` past `rounding` times the logits' own size (`logit_corrections`).
"""
struct LogitSettled{E <: Real}
    tolerance::E
    rounding::E
end

function (settled::LogitSettled{E})(previous::AbstractVector{E}, iterate::AbstractVector{E}, ::AbstractVector{E}, scale::E)::Bool where {E <: Real}
    return last(logit_corrections(previous, iterate, settled.rounding, scale)) <= settled.tolerance
end

"""
`tolerance`, which must lie under `target_correction` when one is set. A corrector that settles looser than its
target first correction leaves step control nothing to measure, and the reach passes any move within its noise.
"""
function settle_tolerance(tolerance::E, target_correction::Union{Nothing, E})::E where {E <: Real}
    isnothing(target_correction) || tolerance < target_correction ||
        throw(ArgumentError("the settle tolerance $tolerance must lie under the target first correction $target_correction"))
    return tolerance
end

# Defaults: Gambit's limits, times √2 in logits. Its 1e-4 is 1.4e-4 here; its refusal of a correction of 0.4 is 0.57,
# and step control refuses past 4 targets. A target of 1 jumped branches on two 5x5x5 games; 0.5 did not.
function LogitDistance{E}(;
        tolerance::Real = 1.0f-4,
        settle_ulps::Integer = 16,
        target_correction::Union{Nothing, Real} = 14 // 100,
    )::LogitDistance{E} where {E <: Real}
    return LogitDistance{E}(E(tolerance), Int(settle_ulps), optional_limit(E, target_correction))
end

"""
    PathFollowingSettings{E}(; max_step, max_steps, distance, newton_iterations, contraction_limit, error_estimate,
                             target_contraction, target_turn, max_step_growth, reach, orientation_check, split_check,
                             max_refusals, min_step_ulps, step_metric, linear_solver, endpoint_tolerance, endpoint_ulps)

The controls of a trace on PathFollowing's pseudo-arclength continuation, an `EngineSettings` for `logit_solve` and
`trace_endpoint`. The trace runs in `[w; λ]`, the payoff gaps `w = z / λ` and the precision `λ`.

- `step_metric`: a `TraceGeometry`, the norm of steps and turns in `[w; λ]` (`trace_metric`) at target precision
  one, frozen at each accepted point. The default, `default_step_metric`, is `gambit_settings`' too.
- `max_step`: the largest step in that norm; `Inf` sets none. The first step comes from the game (`opening_step`).
- `max_steps`: accepted steps before the trace fails as `NoConvergence` (`MaxSteps`).
- `distance`: a `CorrectorDistance`, how the corrector measures a move: `StrategyDistance` (Fisher–Rao, in mixed
  strategies) or `LogitDistance`. It carries the corrector's settle tolerance and step control's target correction.
- `newton_iterations`, `contraction_limit`, `error_estimate`: PathFollowing's `Newton`, its corrections under
  `distance`. `LastCorrection()` settles on the last correction, as Gambit does. `ContractionEstimate(; safety)` also
  needs the error left in the iterate to be at most `safety` times the settle tolerance. Its numbers convert to `E`.
- `target_contraction`, `target_turn`, `max_step_growth`, `reach`: PathFollowing's `StepControl`; `nothing` drops a
  target. The turn target is in radians under `step_metric`. A converged point farther than `reach` step lengths from
  its predictor refuses the step, as one that jumped to another branch, unless `distance` puts the move within the
  corrector's settle tolerance. `Inf` switches the refusal off.
- `orientation_check`: PathFollowing's orientation check. `DeterminantSign` refuses every step where
  `sign det([J; tᵀ])` changes: the step ran back along the branch or crossed a bifurcation point. `nothing` switches
  the check off.
- `split_check`: PathFollowing's `HalfStepCheck{E}`, or `nothing` to switch it off. A step with factor at least
  `trigger` runs again in two half steps, which must reach its point within `tolerance` in `distance`. The default
  trigger is 1.3 (`default_split_check`), and the tolerance 10 times `distance`'s settle tolerance. A zero settle
  tolerance has no default: pass `split_check`. Without the check, at the default step control, 4 of 20,000 random
  two-action games and 3 of 1200 unfolded pitchforks left the principal branch on a step that passed every other test.
  Tolerance evidence, at growth 2 and the summed metric, with `StrategyDistance` in Float32 and Float64. A sweep with
  the reach on and off: 203 seeds of 20 two-action games, and battle of the sexes plus 1e-4 to 1e-1 `randn` payoffs,
  300 draws per scale. The half steps flagged all 22 jumps and 148 of 196,594 steps that kept the branch (0.075%).
  Kept steps reached their point within 2.5e-5 at the 99.99th percentile. Each jump failed or landed at least 0.27
  away. So the 5e-4 tolerance lies 20 times above that percentile and 550 times under the nearest jump. At the default
  step control, trigger 1.3 refused 21 of 565,078 kept steps in the hunt (0.004%).
- `max_refusals`: refused attempts in one step before the trace fails as `TraceStuck` (`StepRefusals`).
- `min_step_ulps`: a predictor that moves no coordinate more than this many ulps fails the trace as `TraceStuck`
  (`UnresolvedStep`).
- `linear_solver`: the LinearSolve.jl algorithm for the bordered systems.
- `endpoint_tolerance`, `endpoint_ulps`: `trace_endpoint`'s last Newton correction at the target must move the logits
  no more than `endpoint_tolerance` past `endpoint_ulps` ulps of their size (`LogitSettled`, `verified_landing`). The defaults
  are BifurcationKit's `logit_tolerance` and `settle_ulps`, so both engines bound `Traced.logit_accuracy` alike.
"""
struct PathFollowingSettings{E <: Real, L, D <: CorrectorDistance{E}} <: EngineSettings{E}
    max_step::E
    max_steps::Int
    distance::D
    newton_iterations::Int
    contraction_limit::E
    error_estimate::Union{PathFollowing.LastCorrection, PathFollowing.ContractionEstimate{E}}
    target_contraction::Union{Nothing, E}
    target_turn::Union{Nothing, E}
    max_step_growth::E
    reach::E
    orientation_check::Union{Nothing, PathFollowing.DeterminantSign}
    split_check::Union{Nothing, PathFollowing.HalfStepCheck{E}}
    max_refusals::Int
    min_step_ulps::Int
    step_metric::TraceGeometry{E}
    linear_solver::L
    endpoint_check::LogitSettled{E}
end

# Without the orientation check, five of ten logitnash_speed.jl games ran back. ContractionEstimate alone at safety 1
# lost the 20x20 seed-6 fold, and 0.5 left ten times the branch error; it now also needs the last correction to pass.
function PathFollowingSettings{E}(;
        max_step::Real = Inf32,
        max_steps::Integer = 2_000,
        distance::D = StrategyDistance{E}(),
        newton_iterations::Integer = 8,
        contraction_limit::Real = 0.5f0,
        error_estimate::Union{PathFollowing.LastCorrection, PathFollowing.ContractionEstimate} = PathFollowing.ContractionEstimate(; safety = 0.2f0),
        target_contraction::Union{Nothing, Real} = 0.25f0,
        # Step control refuses past twice the target: π/2, Gambit's refusal of a tangent that turns back (`t·newT < 0`).
        target_turn::Union{Nothing, Real} = π / 4,
        # Both engines' default. Without the split check, all 10 random jump games of logitnash_tests.jl leave their
        # branch at growth 2 with the summed metric, and none at 1.1.
        max_step_growth::Real = GAMBIT_MAX_ACCEL,
        # On unfolded pitchforks a step that jumped branches moved 4.2 or more of its lengths, one that kept it 1.86 at most.
        reach::Real = 3,
        orientation_check::Union{Nothing, PathFollowing.DeterminantSign} = PathFollowing.DeterminantSign(),
        split_check::Union{Nothing, PathFollowing.HalfStepCheck{E}} = default_split_check(distance),
        max_refusals::Integer = 80,
        min_step_ulps::Integer = 4,
        # Both engines' default. At growth 1.1 it took 35-68% fewer hard_games steps than the summed metric.
        step_metric::TraceGeometry{E} = default_step_metric(E),
        linear_solver::L = LinearSolve.LUFactorization(),
        endpoint_tolerance::Real = 1.0f-4,
        endpoint_ulps::Integer = 16,
    )::PathFollowingSettings{E, L, D} where {E <: Real, L, D <: CorrectorDistance{E}}
    return PathFollowingSettings{E, L, D}(
        E(max_step), Int(max_steps), distance, Int(newton_iterations), E(contraction_limit), estimate_in(E, error_estimate),
        optional_limit(E, target_contraction), optional_limit(E, target_turn), E(max_step_growth), PathFollowing.reach_in(E, reach), orientation_check,
        split_check, Int(max_refusals), Int(min_step_ulps), step_metric, linear_solver,
        LogitSettled(E(endpoint_tolerance), endpoint_ulps * eps(E)),
    )
end

"""
`PathFollowingSettings`' default split check: trigger 1.3 and tolerance 10 times `distance`'s settle tolerance. At
growth 1.1 a step held by a step-control target rides at `f` near 1. So trigger 1 fired on up to 67% of
logitnash_speed's steps and cost up to 33% per game (median). Trigger 1.3 fired on 0.4-2.7% of them and cost at most
2%. A hunt over 20,000 two-action games and 1200 unfolded pitchforks found the same 7 jumps in Float32 and Float64.
Trigger 1.3 fired on and refused each; the smallest jump `f` was 1.41. At growth 2 with the summed metric, jumps came
as low as `f` 1.1: a looser step control needs trigger 1. A zero settle tolerance gives the tolerance no scale, so it
throws an `ArgumentError` that asks for an explicit `split_check`.
"""
function default_split_check(distance::CorrectorDistance{E})::PathFollowing.HalfStepCheck{E} where {E <: Real}
    iszero(distance.tolerance) && throw(
        ArgumentError(
            "the default split check's tolerance is 10 times the settle tolerance, which is zero here; pass split_check = " *
                "HalfStepCheck{$E}(; trigger, tolerance), with a tolerance above the corrector's settled error, or nothing",
        ),
    )
    return PathFollowing.HalfStepCheck{E}(; trigger = 13 // 10, tolerance = 10 * distance.tolerance)
end

"""`estimate` with its numbers in `E`."""
estimate_in(::Type{<:Real}, estimate::PathFollowing.LastCorrection)::PathFollowing.LastCorrection = estimate
function estimate_in(::Type{E}, estimate::PathFollowing.ContractionEstimate)::PathFollowing.ContractionEstimate{E} where {E <: Real}
    return PathFollowing.ContractionEstimate(; safety = E(estimate.safety))
end

"""`limit` in `E`, or `nothing` where it is off."""
optional_limit(::Type{<:Real}, ::Nothing)::Nothing = nothing
function optional_limit(::Type{E}, limit::Real)::E where {E <: Real}
    return E(limit)
end

"""
    QRESystem(game, logits, pass_work)

`game`'s branch as PathFollowing's `system!!` over points `[w; λ]`: `branch_rows!!` into the tracer's residual and
Jacobian, with logits and pass work arrays (`pass_workspace`) of its own. Static arrays and a static game (`SArray`
payoffs) make it isbits, so a GPU work item runs it (`logit_solve_lambda_batch`).
"""
struct QRESystem{G <: LogitGame, L <: AbstractVector, W}
    game::G
    logits::L
    pass_work::W
end

function (system::QRESystem)(residual::R, jacobian::J, point::AbstractVector{E})::Tuple{R, J} where {E <: Real, R <: AbstractVector{E}, J <: AbstractMatrix{E}}
    residual, jacobian, _, _ = branch_rows!!(residual, jacobian, system.logits, system.pass_work, system.game, point)
    return residual, jacobian
end

"""
    FisherRaoSettled(ranges, tolerance, rounding)

PathFollowing's `converged` under `StrategyDistance`, each player's logits in its range of `ranges` (`logit_ranges`). The
last correction, times Newton's `scale`, moved the profile (`FisherRaoChange`) no further than `tolerance` plus
`rounding` times the root sum of squares of the players' logit spreads (`fisher_rao_spread`). That product bounds the
distance a relative rounding of `rounding` in each logit moves the profile. Tuple ranges make it isbits.
"""
struct FisherRaoSettled{R, E <: Real}
    ranges::R
    tolerance::E
    rounding::E
end

function (settled::FisherRaoSettled{R, E})(previous::AbstractVector{E}, iterate::AbstractVector{E}, ::AbstractVector{E}, scale::E)::Bool where {R, E <: Real}
    squared_distance, squared_spread = zero(E), zero(E)
    for range in settled.ranges
        (distance, spread) = fisher_rao_spread(previous, iterate, range)
        squared_distance += distance^2
        squared_spread += spread^2
    end
    # A NaN distance fails the comparison, so it never settles.
    return scale * sqrt(squared_distance) <= settled.tolerance + settled.rounding * sqrt(squared_spread)
end

"""
    FisherRaoChange(ranges)

The Fisher–Rao distance a profile moves between two points `[w; λ]` on the product of the players' simplices: the root
sum of squares of each player's distance (`fisher_rao_spread`). Each player's logits `z = λ w` lie in its range of
`ranges` (`logit_ranges`), against its first strategy. Tuple ranges make it isbits.
"""
struct FisherRaoChange{R} <: Function
    ranges::R
end

function (change::FisherRaoChange)(from::AbstractVector{E}, to::AbstractVector{E})::E where {E <: Real}
    squared = zero(E)
    for range in change.ranges
        squared += first(fisher_rao_spread(from, to, range))^2
    end
    return sqrt(squared)
end

"""The square root of strategy `index`'s probability at `point` (`log_probability`), its player's `log_normalizer` `normalizer`."""
function root_probability(point::AbstractVector{E}, index::Integer, normalizer::Tuple{Int, E})::E where {E <: Real}
    return exp(log_probability(point, index, normalizer) / 2)
end

"""
A player's move from `from` to `to`, its logits in `range`, and its logits' spread at `to`. The move is the Fisher–Rao
distance `4 asin(h / 2)`, `h` the Euclidean distance between the square roots of the two strategies: a geodesic on the
probability simplex, at most `π`. A small move of logits `Δz` has length `√Var_π(Δz)`. The spread is the root mean
square `√(Σ π z²)` of the logits `z = λ w[range]` at `to` under their own mixed strategy `π`.
"""
function fisher_rao_spread(from::AbstractVector{E}, to::AbstractVector{E}, range::UnitRange{Int})::NTuple{2, E} where {E <: Real}
    from_normalizer, to_normalizer = log_normalizer(from, range), log_normalizer(to, range)
    # The first strategy's logit is zero, so it adds nothing to the spread.
    squared = (root_probability(to, 0, to_normalizer) - root_probability(from, 0, from_normalizer))^2
    spread = zero(E)
    for index in range
        logit = last(to) * to[index]
        root = root_probability(to, index, to_normalizer)
        squared += (root - root_probability(from, index, from_normalizer))^2
        # `√π z` stays finite where `π` underflows and `z²` overflows.
        spread += (root * logit)^2
    end
    return (4 * asin(sqrt(squared) / 2), sqrt(spread))
end

"""
The callable that measures a corrector's moves under `distance` over `width` logits of `pass!!`'s game: `logit_change`,
or `FisherRaoChange` over its `logit_ranges` (`checked_ranges`).
"""
corrector_measure(::LogitDistance, ::Any, ::Integer)::Function = logit_change

function corrector_measure(::StrategyDistance, pass!!::Any, width::Integer)::Function
    return FisherRaoChange(checked_ranges(logit_ranges(pass!!), width))
end

# A Fisher–Rao distance on one simplex is at most π, so on the product of `N` it is at most π√N; a logit move has no bound.
PathFollowing.correction_bound(change::FisherRaoChange, ::Type{E}) where {E <: Real} = E(π) * sqrt(E(length(change.ranges)))
PathFollowing.correction_bound(::typeof(logit_change), ::Type{E}) where {E <: Real} = E(Inf32)

"""The corrector's settle test under `distance`, whose moves `measure` measures: `LogitSettled` or `FisherRaoSettled`."""
function settle_test(distance::LogitDistance{E}, ::Any)::LogitSettled{E} where {E <: Real}
    return LogitSettled(distance.tolerance, distance.settle_ulps * eps(E))
end

function settle_test(distance::StrategyDistance{E}, measure::FisherRaoChange{R})::FisherRaoSettled{R, E} where {R, E <: Real}
    return FisherRaoSettled(measure.ranges, distance.tolerance, distance.settle_ulps * eps(E))
end

"""
    PositivePrecision(accept_corrected)

PathFollowing's `accept`: it refuses a corrected point at precision zero or below, and one that `accept_corrected`
refuses (`corrected_accepted`, `converged` true). The principal branch meets precision zero only at its start, so such a
point went back along it; the refusal retries a shorter step. `AcceptAll()` refuses no other point, and builds no logits.
"""
struct PositivePrecision{A}
    accept_corrected::A
end

function (accept::PositivePrecision)(::AbstractVector, ::AbstractVector, corrected::AbstractVector)::Bool
    return last(corrected) > 0 && corrected_accepted(accept.accept_corrected, corrected, last(corrected), true)
end

"""
PathFollowing's `PseudoArclength` for `settings` on `problem`'s game (`path_problem`), its corrected points judged by
`accept` (`PositivePrecision`) and its metric in the storage of the start (`point_metric`). A static problem, an isbits
`accept` and `linear_solver = PathFollowing.StaticLU()` make it isbits.
"""
function path_algorithm(
        settings::PathFollowingSettings{E},
        problem::PathFollowing.CurveProblem,
        accept::PositivePrecision,
    )::PathFollowing.PseudoArclength where {E <: Real}
    (; start) = problem
    pass!!, width = problem.system!!.game.pass!!, length(start) - 1
    measure = corrector_measure(settings.distance, pass!!, width)
    return PathFollowing.PseudoArclength(;
        corrector = PathFollowing.Newton(;
            max_iterations = settings.newton_iterations,
            converged = settle_test(settings.distance, measure),
            settings.contraction_limit,
            settings.error_estimate,
            distance = measure,
        ),
        step_control = PathFollowing.StepControl(;
            settings.target_contraction, settings.distance.target_correction, settings.target_turn,
            settings.max_step_growth, settings.max_step, settings.reach,
        ),
        # PathFollowing measures the precision at target one, and has no PALC state weight to divide by.
        metric = point_metric(trace_metric(settings.step_metric, pass!!, width, one(E), one(E)), start),
        settings.orientation_check,
        settings.split_check,
        accept,
        settings.linear_solver,
        settings.min_step_ulps,
        settings.max_refusals,
    )
end

"""
`metric` in the storage of `start`: `FixedWeights` take static weights at a static start, so the algorithm is isbits.
A `FisherMetric` reads only its ranges and numbers, so it serves every storage.
"""
point_metric(metric::PathFollowing.Metric, ::AbstractVector)::PathFollowing.Metric = metric

function point_metric(metric::PathFollowing.FixedWeights, ::StaticVector{P, E})::PathFollowing.FixedWeights{SVector{P, E}} where {P, E <: Real}
    return PathFollowing.FixedWeights(SVector{P, E}(metric.weights))
end

"""
PathFollowing's `CurveProblem` of the principal branch of `pass!!`'s game (`LogitGame`) in `[w; λ]`: from the flat
start (`trace_start`) toward rising `λ`, in the storage of `flat`, one zero per logit. The game's target precision is
one, where PathFollowing measures the precision; a `Bound` sets where the steps stop. A static `flat` and pass give an
isbits problem.
"""
function path_problem(pass!!::Any, flat::AbstractVector{E})::PathFollowing.CurveProblem where {E <: Real}
    pass_work = pass_workspace(pass!!, flat)
    return path_problem(pass!!, flat, trace_start(LogitGame(pass!!, one(E)), flat, pass_work), pass_work)
end

"""`path_problem` from a `start` (`trace_start`) already built on the pass work `pass_work`."""
function path_problem(pass!!::Any, flat::AbstractVector{E}, start::AbstractVector{E}, pass_work::Any)::PathFollowing.CurveProblem where {E <: Real}
    direction = set_entry!!(zero(start), one(E), lastindex(start))
    return PathFollowing.CurveProblem(QRESystem(LogitGame(pass!!, one(E)), zero(flat), pass_work), start, direction)
end

"""The first step of a trace of `problem` (`path_problem`) under `settings` (`opening_step`)."""
function path_first_step(problem::PathFollowing.CurveProblem, settings::PathFollowingSettings{E})::E where {E <: Real}
    return opening_step(problem.system!!.game, settings.step_metric, settings.max_step)
end

"""
PathFollowing's `init` of `problem` (`path_problem`) under `settings`, its corrected points judged by `accept`
(`PositivePrecision`): a `Tracer` from the flat start toward rising `λ`, or the `StepFailure` there.
"""
function path_init(
        problem::PathFollowing.CurveProblem,
        settings::PathFollowingSettings{E},
        accept::PositivePrecision,
    )::Result{<:PathFollowing.Tracer{E}, PathFollowing.StepFailure.Type{E}} where {E <: Real}
    return CommonSolve.init(problem, path_algorithm(settings, problem, accept); first_step = path_first_step(problem, settings))
end

"""The `TraceStop` of a PathFollowing trace at `point = [w; λ]` after `steps` steps, holding a step of `step_length`."""
function TraceStop(point::AbstractVector{E}, steps::Integer, step_length::E)::TraceStop{E} where {E <: Real}
    return TraceStop(collect(logits_at(point)), collect(payoff_gaps(point)), last(point), Int(steps), step_length)
end

"""
The `TraceCause` of a trace that PathFollowing's `failure` ended: `TooManySteps` spent `MaxSteps`, an `Unresolved` step
or `TooManyRefusals` stuck it, and a start that is not finite or has no tangent fails `AtStart`.
"""
function path_cause(failure::PathFollowing.StepFailure.Type{E})::TraceCause.Type{E} where {E <: Real}
    isa_variant(failure, PathFollowing.StepFailure.TooManySteps) && return TraceCause.NoConvergence{E}(MaxSteps)
    isa_variant(failure, PathFollowing.StepFailure.Unresolved) && return TraceCause.TraceStuck{E}(UnresolvedStep)
    isa_variant(failure, PathFollowing.StepFailure.TooManyRefusals) && return TraceCause.TraceStuck{E}(StepRefusals)
    isa_variant(failure, PathFollowing.StepFailure.NotFiniteStart) && return TraceCause.NotFinite{E}(AtStart)
    isa_variant(failure, PathFollowing.StepFailure.SingularStart) || throw(ArgumentError(lazy"no trace cause for $failure"))
    return TraceCause.Singular{E}(AtStart)
end

"""
    TraceFailure(failure::PathFollowing.StepFailure, point, steps, step_length, context)

The failure of a trace that PathFollowing's `failure` stopped at `point = [w; λ]` after `steps` steps, holding a step
of `step_length`, with the caller's `context`: its `path_cause`, and `failure` as the engine's record.
"""
function TraceFailure(
        failure::PathFollowing.StepFailure.Type{E},
        point::AbstractVector{E},
        steps::Integer,
        step_length::E,
        context::C,
    )::TraceFailure{E, C} where {E <: Real, C <: NamedTuple}
    cause = path_cause(failure)
    @debug "trace stopped" _group = :trace cause
    return TraceFailure(cause, TraceStop(point, steps, step_length), EngineRecord.PathFollowingStep(failure), context)
end

"""
    TraceFailure(cause, tracer::PathFollowing.Tracer, context)

The failure for `cause` of a trace at `tracer`'s point, with the caller's `context`, where GameTracing's own check
stopped it: PathFollowing reported no failure, so the engine has no record.
"""
function TraceFailure(cause::TraceCause.Type{E}, tracer::PathFollowing.Tracer{E}, context::C)::TraceFailure{E, C} where {E <: Real, C <: NamedTuple}
    @debug "trace stopped" _group = :trace cause
    return TraceFailure(cause, TraceStop(tracer.point, tracer.steps, tracer.step_length), EngineRecord.NoRecord{E}(), context)
end

"""
`tracer` advanced by one accepted step that stops at precision `precision` (PathFollowing's `step!!` toward a `Bound`),
and the step (PathFollowing's `AcceptedStep`), or the `TraceFailure` that ends the trace: `NoConvergence` once it has
taken `max_steps` (as PathFollowing's `TooManySteps`), else the `path_cause` of PathFollowing's `Unresolved` or
`TooManyRefusals`. `context` rides on the failure. Use only the returned tracer.
"""
function path_step!!(
        tracer::T,
        precision::E,
        max_steps::Integer,
        context::C,
    )::Tuple{T, Result{PathFollowing.AcceptedStep{E}, TraceFailure{E, C}}} where {E <: Real, C <: NamedTuple, T <: PathFollowing.Tracer{E}}
    if tracer.steps >= max_steps
        return tracer, Err(TraceFailure(PathFollowing.StepFailure.TooManySteps{E}(), tracer.point, tracer.steps, tracer.step_length, context))
    end
    tracer, stepped = PathFollowing.step!!(tracer, PathFollowing.Bound(length(tracer.point), precision))
    if is_error(stepped)
        return tracer, Err(TraceFailure(unwrap_error(stepped), tracer.point, tracer.steps, tracer.step_length, context))
    end
    @debug "trace point" _group = :trace step = tracer.steps precision = last(tracer.point) step_length = unwrap(stepped).step_length
    return tracer, Ok(unwrap(stepped))
end

"""
`settled_points` on PathFollowing: the accepted points of `gains`' principal branch, to the first whose largest regret is
at most `tolerance`. The steps stop at `maximum_precision` (`path_step!!`), where the trace settles or fails
`NoConvergence` (`MaximumPrecision`).
"""
function engine_points(
        gains::NormalFormGains,
        settings::PathFollowingSettings{E},
        maximum_precision::E,
        tolerance::E,
        context::C,
    )::Result{BranchPoints{E}, TraceFailure{E, C}} where {E <: Real, C <: NamedTuple}
    problem = path_problem(gains, zeros(E, logit_count(gains)))
    started = path_init(problem, settings, PositivePrecision(AcceptAll()))
    is_error(started) && return Err(TraceFailure(unwrap_error(started), problem.start, 0, zero(E), context))
    return recorded_points(unwrap(started), BranchRecorder(gains, tolerance), maximum_precision, settings.max_steps, context)
end

"""`engine_points` on a started `tracer`, each accepted point into `recorder`."""
function recorded_points(
        tracer::PathFollowing.Tracer{E},
        recorder::BranchRecorder,
        maximum_precision::E,
        max_steps::Integer,
        context::C,
    )::Result{BranchPoints{E}, TraceFailure{E, C}} where {E <: Real, C <: NamedTuple}
    tracer, stepped = path_step!!(tracer, maximum_precision, max_steps, context)
    while !is_error(stepped)
        point = tracer.point
        if !recorder(logits_at(point), last(point))
            return Ok((; recorder.precisions, recorder.logits))
        end
        if last(point) >= maximum_precision
            return Err(TraceFailure(TraceCause.NoConvergence{E}(MaximumPrecision), tracer, context))
        end
        tracer, stepped = path_step!!(tracer, maximum_precision, max_steps, context)
    end
    return Err(unwrap_error(stepped))
end

"""
`logit_solve_lambda` on PathFollowing: the logits of `gains`' principal branch at each precision of `precisions`, from
one trace whose steps stop at each in rising order (`stepped_landing!!`), or the first failure. No precision takes no
trace, as on BifurcationKit; a failure at the start holds the lowest precision.
"""
function engine_lambdas(
        gains::NormalFormGains,
        precisions::AbstractVector{E},
        settings::PathFollowingSettings{E},
    )::Result{Vector{Vector{E}}, TraceFailure{E, LambdaContext{E}}} where {E <: Real}
    isempty(precisions) && return Ok(Vector{E}[])
    problem = path_problem(gains, zeros(E, logit_count(gains)))
    started = path_init(problem, settings, PositivePrecision(AcceptAll()))
    is_error(started) && return Err(TraceFailure(unwrap_error(started), problem.start, 0, zero(E), (; lam = minimum(precisions))))
    return landed_lambdas(unwrap(started), precisions, settings.max_steps)
end

"""`engine_lambdas` on a started `tracer`."""
function landed_lambdas(
        tracer::PathFollowing.Tracer{E},
        precisions::AbstractVector{E},
        max_steps::Integer,
    )::Result{Vector{Vector{E}}, TraceFailure{E, LambdaContext{E}}} where {E <: Real}
    points = Vector{Vector{E}}(undef, length(precisions))
    for index in sortperm(precisions)
        tracer, landed = stepped_landing!!(tracer, precisions[index], max_steps, AcceptAll(), (; lam = precisions[index]))
        if is_error(landed)
            return Err(unwrap_error(landed))
        end
        points[index] = unwrap(landed)
    end
    return Ok([collect(logits_at(point)) for point in points])
end

"""
`tracer` advanced to `precision`, and the point of its branch there: its steps stop there (`path_step!!`), at the first
crossing a predictor or a corrected point shows. `keep_going(logits, λ)` runs at each accepted point, and `false` short
of `precision` fails the trace `TraceStuck` (`StoppedShort`). `context` rides on every failure. Use only the returned
tracer.
"""
function stepped_landing!!(
        tracer::T,
        precision::E,
        max_steps::Integer,
        keep_going::Any,
        context::C,
    )::Tuple{T, Result{Vector{E}, TraceFailure{E, C}}} where {E <: Real, C <: NamedTuple, T <: PathFollowing.Tracer{E}}
    while last(tracer.point) < precision
        tracer, stepped = path_step!!(tracer, precision, max_steps, context)
        if is_error(stepped)
            return tracer, Err(unwrap_error(stepped))
        end
        if !kept_going(keep_going, tracer.point) && last(tracer.point) < precision
            return tracer, Err(TraceFailure(TraceCause.TraceStuck{E}(StoppedShort), tracer, context))
        end
    end
    return tracer, Ok(copy(tracer.point))
end

"""Whether `keep_going` lets a trace go on from the accepted point `point = [w; λ]`, from its logits."""
kept_going(keep_going::Any, point::AbstractVector)::Bool = keep_going(logits_at(point), last(point))
kept_going(::AcceptAll, ::AbstractVector)::Bool = true

check_redraw(::PathFollowingSettings, ::NoRedraw)::Nothing = nothing
function check_redraw(::PathFollowingSettings, redraw::Any)::Nothing
    throw(ArgumentError(lazy"PathFollowing traces one game: pass BifurcationKitSettings to redraw it; got redraw = $redraw"))
end

"""
    engine_endpoint(game, start, pass_work, settings::PathFollowingSettings, callbacks, context)

`trace_endpoint` on PathFollowing from the finite `start = [w; 0]` (`trace_start`) on `game`'s pass work `pass_work`,
to a nonzero target: one trace whose steps stop there (`stepped_landing!!`), with `callbacks = (; keep_going,
accept_corrected, redraw)`. `accept_corrected` sees converged points only (`converged` is `true`), and its `false`
refuses the point, which shrinks the step (`PositivePrecision`). A stop short of the target fails `TraceStuck` for
PathFollowing's `Unresolved` or `TooManyRefusals`, and a start PathFollowing refuses fails `NotFinite` or `Singular`
(`path_cause`); the engine record holds PathFollowing's `StepFailure`. The endpoint passes `verified_landing`.
"""
function engine_endpoint(
        game::LogitGame,
        start::AbstractVector{E},
        pass_work::Any,
        settings::PathFollowingSettings{E},
        callbacks::NamedTuple,
        context::C,
    )::TraceResult{E, C} where {E <: Real, C <: NamedTuple}
    problem = path_problem(game.pass!!, zero(start[begin:(end - 1)]), start, pass_work)
    return landed_endpoint(
        path_init(problem, settings, PositivePrecision(callbacks.accept_corrected)),
        start, E(game.target_precision), settings, callbacks.keep_going, context,
    )
end

"""
`trace_endpoint` behind a function barrier on `started` (`path_init`): its `StepFailure` at `start`, or its landing at
`target` (`stepped_landing!!`), checked. The settings pick the tracer's type at run time, so `started` is not inferred.
"""
function landed_endpoint(
        started::Result{<:PathFollowing.Tracer{E}, PathFollowing.StepFailure.Type{E}},
        start::AbstractVector{E},
        target::E,
        settings::PathFollowingSettings{E},
        keep_going::Any,
        context::C,
    )::TraceResult{E, C} where {E <: Real, C <: NamedTuple}
    is_error(started) && return Err(TraceFailure(unwrap_error(started), start, 0, zero(E), context))
    tracer, landed = stepped_landing!!(unwrap(started), target, settings.max_steps, keep_going, context)
    is_error(landed) && return Err(unwrap_error(landed))
    return verified_landing(tracer, settings.endpoint_check, context)
end

"""
The `TraceResult` at `tracer`'s point on its target precision, after one more Newton correction there with `λ` pinned
(`pinned_correction`), the check of BifurcationKit's `verified_endpoint`. That correction fails `NotFinite` or
`Singular` (`AtEndpoint`). It must settle under the trace's own corrector (`settle_test`) and in logits
(`endpoint_check`, a `LogitSettled`), else it fails `PrecisionExhausted` (`AtEndpoint`). Its `StalledCorrection`
holds the correction's logit move (`logit_change`) and its iterate, pinned at the endpoint's precision, as BifurcationKit's
does. Else the point is `Ok(Traced)`, its `logit_accuracy` that logit move. The logit test bounds the endpoint's logits
where the default `StrategyDistance` does not: a strategy off the support counts by its probability.
"""
function verified_landing(tracer::PathFollowing.Tracer{E}, endpoint_check::LogitSettled{E}, context::C)::TraceResult{E, C} where {E <: Real, C <: NamedTuple}
    (; point, steps) = tracer
    (; game, pass_work) = tracer.problem.system!!
    pinned = pinned_correction(game, point, pass_work)
    is_error(pinned) && return Err(TraceFailure(unwrap_error(pinned), tracer, context))
    (; corrected, residual) = unwrap(pinned)
    precision = last(point)
    correction = logit_change(point, corrected)
    settled = tracer.algorithm.corrector.converged(point, corrected, residual, one(E))
    if !(settled && endpoint_check(point, corrected, residual, one(E)))
        stall = StalledCorrection(precision, correction, corrected, precision, zero(E))
        return Err(TraceFailure(TraceCause.PrecisionExhausted(AtEndpoint, stall), tracer, context))
    end
    return Ok(Traced(collect(logits_at(point)), steps, correction))
end

"""
    logit_solve_lambda_batch(games, lam; settings, backend, workgroupsize)

`logit_solve_lambda` on PathFollowing for each of `games` at the one precision `lam`: a `ProfileResult` per game, in
order, on the host. Each game traces in its own work item on the KernelAbstractions `backend` (PathFollowing's
`land_batch`), from its flat start with its own first step (`opening_step`). A failed trace gives the `TraceFailure`
that `logit_solve_lambda` gives, its context `lam` (`LambdaContext`).

- `games`: `NormalFormGains` of one concrete type with static payoffs (`SArray`), every player with the same strategy count.
- `settings`: `PathFollowingSettings` in the payoffs' float type, with `linear_solver = PathFollowing.StaticLU()`.
- `backend`: the device that runs the traces. The problems copy to it, and the landings copy back.
- `workgroupsize`: the work items per group. The CPU backend runs each group on one thread, so use a small group there.

`lam` must be non-negative and finite in the trace's element type (`ArgumentError` otherwise). The traces run on
static arrays and `StaticLU`, whose arithmetic differs from the mutable trace's: the profiles agree with
`logit_solve_lambda`'s to the corrector's tolerance, not to the bit.
"""
function logit_solve_lambda_batch(
        games::AbstractVector{G},
        lam::Real;
        settings::PathFollowingSettings{E},
        backend::KernelAbstractions.Backend,
        workgroupsize::Integer,
    )::AbstractVector{ProfileResult{E, N, LambdaContext{E}}} where {N, G <: NormalFormGains{N, <:SArray}, E <: Real}
    isconcretetype(G) || throw(ArgumentError(lazy"the games must share one concrete type; got $G"))
    if !(settings.linear_solver isa PathFollowing.StaticLU)
        throw(ArgumentError(lazy"a batch needs linear_solver = PathFollowing.StaticLU(); got $(settings.linear_solver)"))
    end
    precision = E(lam)
    # Checked before and after conversion to `E`, as in `logit_solve_lambda`.
    if !(reachable(lam) && reachable(precision))
        throw(ArgumentError(lazy"lam must be non-negative and finite in $E; got $lam"))
    end
    isempty(games) && return ProfileResult{E, N, LambdaContext{E}}[]
    require_payoff_type(first(games), settings)
    return batch_profiles(games, batch_inputs(games, precision, settings), settings.max_steps; backend, workgroupsize)
end

"""
The inputs of `land_batch` for `games` at precision `precision`, on the host: a static `CurveProblem` per game
(`path_problem`), the isbits algorithm (`path_algorithm`), each game's first step (`path_first_step`) and the bound.
"""
function batch_inputs(games::AbstractVector{<:NormalFormGains}, precision::E, settings::PathFollowingSettings{E})::NamedTuple where {E <: Real}
    flat = flat_logits(first(games), E)
    problems = path_problem.(games, (flat,))
    # The algorithm's type follows the settings' values, so no local holds it; `batch_profiles` takes it concretely.
    return (;
        problems,
        algorithm = path_algorithm(settings, first(problems), PositivePrecision(AcceptAll())),
        first_steps = path_first_step.(problems, (settings,)),
        bound = PathFollowing.Bound(length(flat) + 1, precision),
    )
end

"""
Each of `games`' `ProfileResult` on the host, from `land_batch` of its `inputs` (`batch_inputs`) on `backend` in groups
of `workgroupsize`, at most `max_steps` steps each.
"""
function batch_profiles(
        games::AbstractVector{<:NormalFormGains},
        inputs::NamedTuple,
        max_steps::Integer;
        backend::KernelAbstractions.Backend,
        workgroupsize::Integer,
    )::AbstractVector{<:ProfileResult}
    (; problems, algorithm, first_steps, bound) = inputs
    landed = PathFollowing.land_batch(on_device(backend, problems), algorithm, on_device(backend, first_steps), bound, max_steps; workgroupsize)
    return landed_profile.(games, Array(landed), bound.value)
end

"""The flat logits of a static game: one static zero in `E` per logit."""
function flat_logits(::NormalFormGains{N, A}, ::Type{E})::StaticVector{<:Any, E} where {N, A <: SArray, E <: Real}
    return zeros(SVector{sum(Tuple(Size(A))) - N, E})
end

"""`array` copied into a new array on `backend` (`KernelAbstractions.allocate`)."""
function on_device(backend::KernelAbstractions.Backend, array::AbstractVector{T})::AbstractVector{T} where {T}
    return copyto!(KernelAbstractions.allocate(backend, T, length(array)), array)
end

"""
The `ProfileResult` of a batch trace of `gains` to precision `precision` that ended in `landing`: the profile there, or
the failure `logit_solve_lambda` gives.
"""
function landed_profile(
        gains::NormalFormGains{N},
        landing::PathFollowing.Landing{<:StaticVector{<:Any, E}, E},
        precision::E,
    )::ProfileResult{E, N, LambdaContext{E}} where {N, E <: Real}
    landed = PathFollowing.landing_result(landing)
    is_error(landed) || return Ok(LogitQREProfile(gains, precision, collect(logits_at(unwrap(landed)))))
    return Err(TraceFailure(unwrap_error(landed), landing.point, landing.steps, landing.step_length, (; lam = precision)))
end
