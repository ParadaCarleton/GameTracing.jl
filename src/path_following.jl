using CommonSolve: CommonSolve
using ErrorTypes: Err, Ok, Result, is_error, unwrap, unwrap_error
using LinearSolve: LinearSolve
using Logging: @debug
using PathFollowing: PathFollowing

"""
How the corrector measures a move between two points `[w; λ]` (`corrector_measure`), with its settle tolerance and
step control's target first correction in the same units: `StrategyDistance` or `LogitDistance`.
"""
abstract type CorrectorDistance{E <: Real} end

"""
    StrategyDistance{E}(; tolerance, settle_ulps, target_correction)

Measure a move by the largest Fisher–Rao distance any player's mixed strategy travels (`FisherRaoChange`). A strategy
counts by its probability, so one that has left the support stops counting. The corrector settles once a correction
moves no strategy further than `tolerance` past the distance `settle_ulps` ulps of each logit move it
(`FisherRaoSettled`). Step control's target first correction is `target_correction`, or `nothing` to drop the term.
A set target must lie above `tolerance` (`settle_tolerance`).
"""
struct StrategyDistance{E <: Real} <: CorrectorDistance{E}
    tolerance::E
    settle_ulps::Int
    target_correction::Union{Nothing, E}
    function StrategyDistance{E}(tolerance::E, settle_ulps::Int, target_correction::Union{Nothing, E})::StrategyDistance{E} where {E <: Real}
        return new{E}(settle_tolerance(tolerance, target_correction), settle_ulps, target_correction)
    end
end

# Defaults: half the `LogitDistance` defaults, as a logit move at even odds is twice its Fisher–Rao distance.
function StrategyDistance{E}(;
        tolerance::Real = 5.0f-5,
        settle_ulps::Integer = 16,
        target_correction::Union{Nothing, Real} = 0.25f0,
    )::StrategyDistance{E} where {E <: Real}
    return StrategyDistance{E}(E(tolerance), Int(settle_ulps), optional_limit(E, target_correction))
end

"""
    LogitDistance{E}(; tolerance, settle_ulps, target_correction)

Measure a move by the largest change of any logit (`logit_change`). The corrector settles once a correction moves no
logit more than `tolerance` past `settle_ulps` ulps of its own size (`LogitSettled`). Step control's target first
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
`tolerance`, which must lie under `target_correction` when one is set. A corrector that settles looser than its
target first correction leaves step control nothing to measure, and the reach passes any move within its noise.
"""
function settle_tolerance(tolerance::E, target_correction::Union{Nothing, E})::E where {E <: Real}
    isnothing(target_correction) || tolerance < target_correction ||
        throw(ArgumentError("the settle tolerance $tolerance must lie under the target first correction $target_correction"))
    return tolerance
end

# Defaults: the cheapest step control that kept Gambit's branch on ten benchmark games where others left it. A target
# correction of 1 jumped branches on two 5x5x5 games; 0.5 did not, at about the same step count as 1.
function LogitDistance{E}(;
        tolerance::Real = 1.0f-4,
        settle_ulps::Integer = 16,
        target_correction::Union{Nothing, Real} = 0.5f0,
    )::LogitDistance{E} where {E <: Real}
    return LogitDistance{E}(E(tolerance), Int(settle_ulps), optional_limit(E, target_correction))
end

"""
    PathFollowingSettings{E}(; max_step, max_steps, distance, newton_iterations, contraction_limit, error_estimate,
                             target_contraction, target_turn, max_step_growth, reach, orientation_check, split_check,
                             max_refusals, min_step_ulps, step_metric, linear_solver)

The controls of a trace on PathFollowing's pseudo-arclength continuation, an `EngineSettings` for `logit_solve`. The
trace runs in `[w; λ]`, the payoff gaps `w = z / λ` and the precision `λ`.

- `step_metric`: a `TraceGeometry`, the norm of steps and turns in `[w; λ]` (`trace_metric`) at target precision
  one, frozen at each accepted point.
- `max_step`: the largest step in that norm; `Inf` sets none. The first step comes from the game (`opening_step`).
- `max_steps`: accepted steps before the trace fails as `NoConvergence`.
- `distance`: a `CorrectorDistance`, how the corrector measures a move: `StrategyDistance` (Fisher–Rao, in mixed
  strategies) or `LogitDistance`. It carries the corrector's settle tolerance and step control's target correction.
- `newton_iterations`, `contraction_limit`, `error_estimate`: PathFollowing's `Newton`, its corrections under
  `distance`. `ContractionEstimate(; safety)` settles on the error left in the iterate, at most `safety` times the
  settle tolerance, often one iteration sooner than `LastCorrection()`. Its numbers convert to `E`.
- `target_contraction`, `target_turn`, `max_step_growth`, `reach`: PathFollowing's `StepControl`; `nothing` drops a
  target. The turn target is in radians under `step_metric`. A converged point farther than `reach` step lengths from
  its predictor refuses the step, as one that jumped to another branch, unless `distance` puts the move within the
  corrector's settle tolerance. `Inf` switches the refusal off.
- `orientation_check`: PathFollowing's orientation check. `DeterminantSign` refuses every step where
  `sign det([J; tᵀ])` changes: the step ran back along the branch or crossed a bifurcation point. `nothing` switches
  the check off.
- `split_check`: PathFollowing's `HalfStepCheck{E}`, or `nothing` to switch it off. A step with factor at least
  `trigger` runs again in two half steps, which must reach its point within `tolerance` in `distance`. The default
  trigger is 1.3, and the tolerance 10 times `distance`'s settle tolerance. Without the check, about 1 random
  two-action game in 1200 left its principal branch on a step that passed every other test.
  Evidence, with `StrategyDistance` in Float32 and Float64, the reach on and off: 203 seeds of 20 two-action games, and
  battle of the sexes plus 1e-4 to 1e-1 `randn` payoffs, 300 draws per scale. The half steps flagged all 22 jumps and
  148 of 196,594 steps that kept the branch (0.075%). Kept steps reached their point within 2.5e-5 at the 99.99th
  percentile. Each jump failed or landed at least 0.27 away. So the 5e-4 tolerance lies 20 times above that
  percentile and 550 times under the nearest jump. Every jump had `f` at least 1.52; a trigger of 1.3 fires on 1.9%
  of kept steps. With the reach off, the check keeps three of the four unfolded pitchforks that the reach keeps. At
  the fourth, both half steps jump with the step.
- `max_refusals`: refused attempts in one step before the trace fails as `TraceStuck`.
- `min_step_ulps`: a predictor that moves no coordinate more than this many ulps fails the trace as `TraceStuck`.
- `linear_solver`: the LinearSolve.jl algorithm for the bordered systems.
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
end

# On logitnash_speed.jl's games and 100 larger ones, a 1 rad turn target took 14-18% fewer steps than 0.3 and kept every
# branch. Without the orientation check, five of ten games ran back. ContractionEstimate at safety 0.2 ran 8.6% faster
# than LastCorrection; safety 1 lost the 20x20 seed-6 fold, and 0.5 left ten times the branch error.
function PathFollowingSettings{E}(;
        max_step::Real = Inf32,
        max_steps::Integer = 2_000,
        distance::D = StrategyDistance{E}(),
        newton_iterations::Integer = 8,
        contraction_limit::Real = 0.5f0,
        error_estimate::Union{PathFollowing.LastCorrection, PathFollowing.ContractionEstimate} = PathFollowing.ContractionEstimate(; safety = 0.2f0),
        target_contraction::Union{Nothing, Real} = 0.25f0,
        # On hard_games.jl's 543 games, 0.3 left no endpoint off its branch to 1 rad's 3, at 15-62% more steps.
        target_turn::Union{Nothing, Real} = 1.0f0,
        # Growth 4 left 8 endpoints off their branch to growth 2's 3. Results: benchmark/results/hard-b5a6b4d-summary.txt.
        max_step_growth::Real = 2.0f0,
        # On unfolded pitchforks a step that jumped branches moved 4.2 or more of its lengths, one that kept it 1.86 at most.
        reach::Real = 3,
        orientation_check::Union{Nothing, PathFollowing.DeterminantSign} = PathFollowing.DeterminantSign(),
        split_check::Union{Nothing, PathFollowing.HalfStepCheck{E}} = PathFollowing.HalfStepCheck{E}(; trigger = 13 // 10, tolerance = 10 * distance.tolerance),
        max_refusals::Integer = 80,
        min_step_ulps::Integer = 4,
        # Kept over the Fisher arms (benchmark/results/tune-4dd74d7-summary.txt): the log term at 0.1 took 27% fewer steps
        # on two-action games and 37% fewer on categorical ones, but stalls at battle of the sexes' exact pitchfork.
        step_metric::TraceGeometry{E} = TraceGeometry{E}(; distance = SummedDistance, state_weight = 1),
        linear_solver::L = LinearSolve.LUFactorization(),
    )::PathFollowingSettings{E, L, D} where {E <: Real, L, D <: CorrectorDistance{E}}
    return PathFollowingSettings{E, L, D}(
        E(max_step), Int(max_steps), distance, Int(newton_iterations), E(contraction_limit), estimate_in(E, error_estimate),
        optional_limit(E, target_contraction), optional_limit(E, target_turn), E(max_step_growth), PathFollowing.reach_in(E, reach), orientation_check,
        split_check, Int(max_refusals), Int(min_step_ulps), step_metric, linear_solver,
    )
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
    QRESystem(game, workspace)

`game`'s branch as PathFollowing's `system!` over points `[w; λ]`: `branch_system!` without
its pin row, written into `workspace` first.
"""
struct QRESystem{G <: LogitGame, W <: BranchWorkspace}
    game::G
    workspace::W
end

function (system::QRESystem)(residual::AbstractVector{E}, jacobian::AbstractMatrix{E}, point::AbstractVector{E})::Nothing where {E <: Real}
    filled = branch_system!(system.workspace, system.game, point, last(point))
    width = length(residual)
    residual .= @view filled.residual[1:width]
    jacobian .= @view filled.jacobian[1:width, :]
    return nothing
end

"""
    LogitSettled(tolerance, rounding)

PathFollowing's `converged` over points `[w; λ]`: the last correction, times Newton's `scale`, moved no logit more
than `tolerance` past `rounding` times that logit's own size (`logit_corrections`).
"""
struct LogitSettled{E <: Real}
    tolerance::E
    rounding::E
end

function (settled::LogitSettled{E})(previous::AbstractVector{E}, iterate::AbstractVector{E}, ::AbstractVector{E}, scale::E)::Bool where {E <: Real}
    return last(logit_corrections(previous, iterate, settled.rounding, scale)) <= settled.tolerance
end

"""
    FisherRaoSettled(ranges, tolerance, rounding)

PathFollowing's `converged` under `StrategyDistance`, each player's logits in its range of `ranges`. The last correction,
times Newton's `scale`, moved no player's strategy further than `tolerance` plus `rounding` times its logits' spread
(`fisher_rao_spread`). That product bounds the Fisher–Rao distance a relative rounding of `rounding` in each logit
moves the strategy.
"""
struct FisherRaoSettled{N, E <: Real}
    ranges::NTuple{N, UnitRange{Int}}
    tolerance::E
    rounding::E
end

function (settled::FisherRaoSettled{N, E})(previous::AbstractVector{E}, iterate::AbstractVector{E}, ::AbstractVector{E}, scale::E)::Bool where {N, E <: Real}
    for range in settled.ranges
        (distance, spread) = fisher_rao_spread(previous, iterate, range)
        # A NaN distance fails the comparison, so it never settles.
        scale * distance <= settled.tolerance + settled.rounding * spread || return false
    end
    return true
end

"""
    FisherRaoChange(ranges)

The largest Fisher–Rao distance any player's mixed strategy moves between two points `[w; λ]`, each player's logits
`z = λ w` in its range of `ranges`, against its first strategy (`fisher_rao_spread`).
"""
struct FisherRaoChange{N}
    ranges::NTuple{N, UnitRange{Int}}
end

function (change::FisherRaoChange)(from::AbstractVector{E}, to::AbstractVector{E})::E where {E <: Real}
    largest = zero(E)
    for range in change.ranges
        largest = max(largest, first(fisher_rao_spread(from, to, range)))
    end
    return largest
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

"""The callable that measures a corrector's moves under `distance` on `gains`: `logit_change` or `FisherRaoChange`."""
corrector_measure(::LogitDistance, ::NormalFormGains)::Function = logit_change

function corrector_measure(::StrategyDistance, gains::NormalFormGains{N})::FisherRaoChange{N} where {N}
    return FisherRaoChange(logit_ranges(gains))
end

# A Fisher–Rao distance on the simplex is at most π; a logit move has no bound.
PathFollowing.correction_bound(::FisherRaoChange)::Real = π
PathFollowing.correction_bound(::typeof(logit_change))::Real = Inf32

"""The corrector's settle test under `distance`, whose moves `measure` measures: `LogitSettled` or `FisherRaoSettled`."""
function settle_test(distance::LogitDistance{E}, ::Any)::LogitSettled{E} where {E <: Real}
    return LogitSettled(distance.tolerance, distance.settle_ulps * eps(E))
end

function settle_test(distance::StrategyDistance{E}, measure::FisherRaoChange{N})::FisherRaoSettled{N, E} where {N, E <: Real}
    return FisherRaoSettled(measure.ranges, distance.tolerance, distance.settle_ulps * eps(E))
end

"""
PathFollowing's `accept` that refuses a corrected point at precision zero or below. The principal branch meets
precision zero only at its start, so such a point went back along it; the refusal retries a shorter step.
"""
struct PositivePrecision end

(::PositivePrecision)(::AbstractVector, ::AbstractVector, corrected::AbstractVector)::Bool = last(corrected) > 0

"""PathFollowing's `PseudoArclength` for `settings` on `gains`."""
function path_algorithm(settings::PathFollowingSettings{E}, gains::NormalFormGains)::PathFollowing.PseudoArclength where {E <: Real}
    measure = corrector_measure(settings.distance, gains)
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
        metric = trace_metric(settings.step_metric, gains, logit_count(gains), one(E), one(E)),
        settings.orientation_check,
        settings.split_check,
        accept = PositivePrecision(),
        settings.linear_solver,
        settings.min_step_ulps,
        settings.max_refusals,
    )
end

"""
PathFollowing's `init` of the principal branch of `gains` in `[w; λ]`: a `Tracer` from the flat start toward rising
`λ`, or the `StepFailure` there (`start_cause`, `start_context`).
"""
function path_init(
        gains::NormalFormGains,
        settings::PathFollowingSettings{E},
    )::Result{<:PathFollowing.Tracer{E}, PathFollowing.StepFailure{E}} where {E <: Real}
    game = LogitGame(gains, one(E))
    width = logit_count(gains)
    start = trace_start(game, zeros(E, width))
    direction = setindex!(zeros(E, width + 1), one(E), width + 1)
    return CommonSolve.init(
        PathFollowing.CurveProblem(QRESystem(game, BranchWorkspace(game, start)), start, direction),
        path_algorithm(settings, gains);
        first_step = opening_step(gains, E),
    )
end

"""
The first step of a trace of `gains`: `1 / payoff_range(gains)` in `λ`, where no logit `z = λ w` has moved by more
than one. Step control then grows it by up to `max_step_growth` per step, and `max_step` caps it. A range without a
positive finite inverse (equal, non-finite or subnormal payoffs) steps one; a non-finite start fails anyway
(`NotFiniteStart`).
"""
function opening_step(gains::NormalFormGains, ::Type{E})::E where {E <: Real}
    step = inv(E(payoff_range(gains)))
    return step > 0 && isfinite(step) ? step : one(E)
end

"""The `TraceCause` of PathFollowing's start `failure`: `NotFinite` for a non-finite start, `Singular` otherwise."""
function start_cause(failure::PathFollowing.StepFailure)::TraceCause
    return ifelse(failure.cause == PathFollowing.NotFiniteStart, NotFinite, Singular)
end

"""The context of PathFollowing's start `failure` over `width` logits: `context` and the flat start."""
function start_context(failure::PathFollowing.StepFailure{E}, width::Integer, context::NamedTuple)::NamedTuple where {E <: Real}
    return failure_context(context, (; reason = Symbol(failure.cause), precision = zero(E), logits = zeros(E, width), steps = 0))
end

"""
The context of a trace that stopped at `point = [w; λ]` after `steps` steps for `reason`: `context`, the precision
and logits there, the steps and `details`.
"""
function stopped_context(
        reason::Symbol,
        point::AbstractVector{E},
        steps::Integer,
        context::NamedTuple,
        details::NamedTuple,
    )::NamedTuple where {E <: Real}
    @debug "trace stopped" _group = :trace reason
    return failure_context(context, (; reason, precision = last(point), logits = collect(logits_at(point)), steps, details...))
end

"""
One accepted step of `tracer` (PathFollowing's `AcceptedStep`) that stops at precision `precision` (PathFollowing's
`Bound`), or the `TraceFailure` that ends the trace: `NoConvergence` once it has taken `max_steps`, else `TraceStuck`
for PathFollowing's `Unresolved` or `TooManyRefusals`. `context` rides on the failure.
"""
function path_step!(
        tracer::PathFollowing.Tracer{E},
        precision::E,
        max_steps::Integer,
        context::NamedTuple,
    )::Result{PathFollowing.AcceptedStep{E}, TraceFailure} where {E <: Real}
    if tracer.steps >= max_steps
        return Err(TraceFailure(NoConvergence, stopped_context(:max_steps, tracer.point, tracer.steps, context, (;))))
    end
    stepped = CommonSolve.step!(tracer, PathFollowing.Bound(length(tracer.point), precision))
    if is_error(stepped)
        failure = unwrap_error(stepped)
        reason = ifelse(failure.cause == PathFollowing.Unresolved, :unresolved_step, :refusals)
        details = (; failure.last_refusal, failure.step_length, failure.refusals)
        return Err(TraceFailure(TraceStuck, stopped_context(reason, tracer.point, tracer.steps, context, details)))
    end
    @debug "trace point" _group = :trace step = tracer.steps precision = last(tracer.point) step_length = unwrap(stepped).step_length
    return Ok(unwrap(stepped))
end

"""
`settled_points` on PathFollowing: the accepted points of `gains`' principal branch, to the first whose largest regret is
at most `tolerance`. The steps stop at `maximum_precision` (`path_step!`), where the trace settles or fails
`NoConvergence` with reason `:maximum_precision`.
"""
function engine_points(
        gains::NormalFormGains,
        settings::PathFollowingSettings{E},
        maximum_precision::E,
        tolerance::E,
        context::NamedTuple,
    )::Result{BranchPoints{E}, TraceFailure} where {E <: Real}
    started = path_init(gains, settings)
    if is_error(started)
        failure = unwrap_error(started)
        return Err(TraceFailure(start_cause(failure), start_context(failure, logit_count(gains), context)))
    end
    return recorded_points(unwrap(started), BranchRecorder(gains, tolerance), maximum_precision, settings.max_steps, context)
end

"""`engine_points` on a started `tracer`, each accepted point into `recorder`."""
function recorded_points(
        tracer::PathFollowing.Tracer{E},
        recorder::BranchRecorder,
        maximum_precision::E,
        max_steps::Integer,
        context::NamedTuple,
    )::Result{BranchPoints{E}, TraceFailure} where {E <: Real}
    stepped = path_step!(tracer, maximum_precision, max_steps, context)
    while !is_error(stepped)
        point = tracer.point
        if !recorder(logits_at(point), last(point))
            return Ok((; recorder.precisions, recorder.logits))
        end
        if last(point) >= maximum_precision
            return Err(TraceFailure(NoConvergence, stopped_context(:maximum_precision, point, tracer.steps, context, (; maximum_precision))))
        end
        stepped = path_step!(tracer, maximum_precision, max_steps, context)
    end
    return Err(unwrap_error(stepped))
end

"""
`logit_solve_lambda` on PathFollowing: the logits of `gains`' principal branch at each precision of `precisions`, from
one trace whose steps stop at each in rising order (`stepped_landing!`), or the first failure.
"""
function engine_lambdas(
        gains::NormalFormGains,
        precisions::AbstractVector{E},
        settings::PathFollowingSettings{E},
    )::Result{Vector{Vector{E}}, TraceFailure} where {E <: Real}
    started = path_init(gains, settings)
    if is_error(started)
        failure = unwrap_error(started)
        return Err(TraceFailure(start_cause(failure), start_context(failure, logit_count(gains), (;))))
    end
    return landed_lambdas(unwrap(started), precisions, settings.max_steps)
end

"""`engine_lambdas` on a started `tracer`."""
function landed_lambdas(
        tracer::PathFollowing.Tracer{E},
        precisions::AbstractVector{E},
        max_steps::Integer,
    )::Result{Vector{Vector{E}}, TraceFailure} where {E <: Real}
    points = Vector{Vector{E}}(undef, length(precisions))
    for index in sortperm(precisions)
        landed = stepped_landing!(tracer, precisions[index], max_steps)
        if is_error(landed)
            return Err(unwrap_error(landed))
        end
        points[index] = unwrap(landed)
    end
    return Ok([collect(logits_at(point)) for point in points])
end

"""
The point of `tracer`'s branch at `precision`: its steps stop there (`path_step!`), at the first crossing a predictor
or a corrected point shows.
"""
function stepped_landing!(tracer::PathFollowing.Tracer{E}, precision::E, max_steps::Integer)::Result{Vector{E}, TraceFailure} where {E <: Real}
    context = (; lam = precision)
    while last(tracer.point) < precision
        stepped = path_step!(tracer, precision, max_steps, context)
        if is_error(stepped)
            return Err(unwrap_error(stepped))
        end
    end
    return Ok(copy(tracer.point))
end
