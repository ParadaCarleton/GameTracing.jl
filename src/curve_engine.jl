using CommonSolve: CommonSolve
using CurveTracing: CurveTracing
using ErrorTypes: Err, Ok, Result, is_error, unwrap, unwrap_error
using LinearSolve: LinearSolve
using LogExpFunctions: logistic
using Logging: @debug

"""
    CurveSettings{E}(; first_step, max_step, max_steps, logit_tolerance, settle_ulps, reach, newton_iterations,
                     contraction_limit, nominal_contraction, nominal_distance, nominal_turn, growth_limit,
                     orientation, max_refusals, resolution_ulps, geometry, linear_solver)

The controls of a trace on CurveTracing's pseudo-arclength continuation, an `EngineSettings` for `logit_solve`. The
trace runs in `[w; λ]`, the payoff gaps `w = z / λ` and the precision `λ`.

- `geometry`: a `TraceGeometry`, the norm of steps and turns in `[w; λ]` (`GeometryWeights`), reweighted at each
  accepted point.
- `first_step`, `max_step`: the first and the largest step in that norm.
- `max_steps`: accepted steps before the trace fails as `NoConvergence`.
- `logit_tolerance`, `settle_ulps`: the corrector converges on a correction that moves no logit more than
  `logit_tolerance` past `settle_ulps` ulps of its own size (`LogitSettled`).
- `reach`: refuse a corrected point whose logits move further from the predictor's than `reach` times the
  predictor's own largest logit move (`WithinReach`). `Inf32` switches the test off.
- `newton_iterations`, `contraction_limit`: CurveTracing's `Newton`, its corrections in logits (`logit_change`).
- `nominal_contraction`, `nominal_distance`, `nominal_turn`, `growth_limit`: CurveTracing's `StepControl`. The
  distance is the first correction's largest logit move; `nothing` drops a term.
- `orientation`: CurveTracing's orientation check. `DeterminantSign` refuses a step where `sign det([J; tᵀ])`
  changes, as a step that runs back along the branch; a change that persists is a bifurcation crossing. `nothing`
  switches the check off.
- `max_refusals`: refused attempts in one step before the trace fails as `TraceStuck`.
- `resolution_ulps`: a predictor that moves no coordinate more than this many ulps fails the trace as `TraceStuck`.
- `linear_solver`: the LinearSolve.jl algorithm for the bordered systems.
"""
struct CurveSettings{E <: Real, L} <: EngineSettings{E}
    first_step::E
    max_step::E
    max_steps::Int
    logit_tolerance::E
    settle_ulps::Int
    reach::E
    newton_iterations::Int
    contraction_limit::E
    nominal_contraction::Union{Nothing, E}
    nominal_distance::Union{Nothing, E}
    nominal_turn::Union{Nothing, E}
    growth_limit::E
    orientation::Union{Nothing, CurveTracing.DeterminantSign}
    max_refusals::Int
    resolution_ulps::Int
    geometry::TraceGeometry{E}
    linear_solver::L
end

# Defaults: the cheapest step control that keeps Gambit's branch on ten benchmark games where others left it. Nominal
# distance 1 jumped branches on two 5x5x5 games; 0.5 did not, at about the same step count as 1.
# Without the orientation check, five of them ran back along the branch. `benchmark/logitnash_speed.jl` compares engines.
function CurveSettings{E}(;
        first_step::Real = 0.03f0,
        max_step::Real = 1.0f7,
        max_steps::Integer = 2_000,
        logit_tolerance::Real = 1.0f-4,
        settle_ulps::Integer = 16,
        reach::Real = 2.0f0,
        newton_iterations::Integer = 8,
        contraction_limit::Real = 0.5f0,
        nominal_contraction::Union{Nothing, Real} = 0.25f0,
        nominal_distance::Union{Nothing, Real} = 0.5f0,
        nominal_turn::Union{Nothing, Real} = 0.3f0,
        growth_limit::Real = 4.0f0,
        orientation::Union{Nothing, CurveTracing.DeterminantSign} = CurveTracing.DeterminantSign(; persistence = 4),
        max_refusals::Integer = 80,
        resolution_ulps::Integer = 4,
        geometry::TraceGeometry{E} = TraceGeometry{E}(; distance = SummedDistance, state_weight = 1),
        linear_solver::L = LinearSolve.LUFactorization(),
    )::CurveSettings{E, L} where {E <: Real, L}
    return CurveSettings{E, L}(
        E(first_step), E(max_step), Int(max_steps), E(logit_tolerance), Int(settle_ulps), E(reach), Int(newton_iterations),
        E(contraction_limit), optional_limit(E, nominal_contraction), optional_limit(E, nominal_distance),
        optional_limit(E, nominal_turn), E(growth_limit), orientation, Int(max_refusals),
        Int(resolution_ulps), geometry, linear_solver,
    )
end

"""`limit` in `E`, or `nothing` where it is off."""
optional_limit(::Type{<:Real}, ::Nothing)::Nothing = nothing
function optional_limit(::Type{E}, limit::Real)::E where {E <: Real}
    return E(limit)
end

"""
    GeometryWeights(geometry)

CurveTracing's `PointWeights` for `geometry` over points `[w; λ]`, at precision `λ` in place of the relative one
(`StepDistance`). Each gap weighs `coordinate_weight`, or for a Fisher distance `state_weight σ(z)σ(-z)` at its logit
`z = λ w`. The precision weighs `level_weight`, or `1 / (1 + λ)²` for `FisherLogPrecisionDistance`.
"""
struct GeometryWeights{E <: Real}
    geometry::TraceGeometry{E}
end

function (weighting::GeometryWeights{E})(weights::AbstractVector{E}, point::AbstractVector{E})::AbstractVector{E} where {E <: Real}
    (; distance, state_weight) = weighting.geometry
    precision, gaps = last(point), payoff_gaps(point)
    if distance == FisherDistance || distance == FisherLogPrecisionDistance
        payoff_gaps(weights) .= state_weight .* logistic.(precision .* gaps) .* logistic.(.-(precision .* gaps))
    else
        fill!(payoff_gaps(weights), coordinate_weight(weighting.geometry, length(gaps)))
    end
    weights[end] = ifelse(distance == FisherLogPrecisionDistance, inv(one(E) + abs(precision))^2, level_weight(weighting.geometry))
    return weights
end

"""
    QRESystem(game, workspace)

`game`'s branch as CurveTracing's `system!` over points `[w; λ]`: `branch_system!` at target precision one, without
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

CurveTracing's `converged` over points `[w; λ]`: the last correction moved no logit more than `tolerance` past
`rounding` times that logit's own size (`logit_corrections`).
"""
struct LogitSettled{E <: Real}
    tolerance::E
    rounding::E
end

function (settled::LogitSettled{E})(previous::AbstractVector{E}, iterate::AbstractVector{E}, ::AbstractVector{E})::Bool where {E <: Real}
    return last(logit_corrections(previous, iterate, settled.rounding)) <= settled.tolerance
end

"""
    WithinReach(reach)

CurveTracing's `accept` over points `[w; λ]`: the corrected point's logits moved no further from the predictor's than
`reach` times the predictor's own largest logit move (`logit_change`). A predictor that moved no logit passes.
"""
struct WithinReach{E <: Real}
    reach::E
end

function (within::WithinReach{E})(start::AbstractVector{E}, predicted::AbstractVector{E}, corrected::AbstractVector{E})::Bool where {E <: Real}
    move = logit_change(start, predicted)
    return !(move > 0) || logit_change(predicted, corrected) <= within.reach * move
end

"""CurveTracing's `PseudoArclength` for `settings`."""
function curve_algorithm(settings::CurveSettings{E})::CurveTracing.PseudoArclength where {E <: Real}
    return CurveTracing.PseudoArclength(;
        corrector = CurveTracing.Newton(;
            max_iterations = settings.newton_iterations,
            converged = LogitSettled(settings.logit_tolerance, settings.settle_ulps * eps(E)),
            settings.contraction_limit,
            distance = logit_change,
        ),
        step_control = CurveTracing.StepControl(;
            settings.nominal_contraction, settings.nominal_distance, settings.nominal_turn, settings.growth_limit,
            settings.max_step,
        ),
        metric = CurveTracing.PointWeights(GeometryWeights(settings.geometry)),
        settings.orientation,
        settings.linear_solver,
        accept = WithinReach(settings.reach),
        settings.resolution_ulps,
        settings.max_refusals,
    )
end

"""
CurveTracing's `init` of the principal branch of `gains` in `[w; λ]`: a `Tracer` from the flat start toward rising
`λ`, or the `StepFailure` there (`start_context`).
"""
function curve_init(
        gains::NormalFormGains,
        settings::CurveSettings{E},
    )::Result{<:CurveTracing.Tracer{E}, CurveTracing.StepFailure{E}} where {E <: Real}
    game = LogitGame(gains, one(E))
    width = logit_count(gains)
    start = trace_start(game, zeros(E, width))
    direction = setindex!(zeros(E, width + 1), one(E), width + 1)
    return CommonSolve.init(
        CurveTracing.CurveProblem(QRESystem(game, BranchWorkspace(game, start)), start, direction),
        curve_algorithm(settings);
        settings.first_step,
    )
end

"""The context of CurveTracing's start `failure` over `width` logits: `context` and the flat start."""
function start_context(failure::CurveTracing.StepFailure{E}, width::Integer, context::NamedTuple)::NamedTuple where {E <: Real}
    return (; context..., reason = Symbol(failure.cause), precision = zero(E), logits = zeros(E, width), steps = 0)
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
    return (; context..., reason, precision = last(point), logits = collect(logits_at(point)), steps, details...)
end

"""
One accepted step of `tracer` (CurveTracing's `AcceptedStep`), or the `TraceFailure` that ends the trace:
`NoConvergence` once it has taken `max_steps`, else `TraceStuck` for CurveTracing's `Unresolved` or
`TooManyRefusals`. `context` rides on the failure.
"""
function curve_step!(
        tracer::CurveTracing.Tracer{E},
        max_steps::Integer,
        context::NamedTuple,
    )::Result{CurveTracing.AcceptedStep{E}, TraceFailure} where {E <: Real}
    if tracer.steps >= max_steps
        return Err(TraceFailure(NoConvergence, stopped_context(:max_steps, tracer.point, tracer.steps, context, (;))))
    end
    stepped = CommonSolve.step!(tracer)
    if is_error(stepped)
        failure = unwrap_error(stepped)
        reason = ifelse(failure.cause == CurveTracing.Unresolved, :unresolved_step, :refusals)
        details = (; failure.last_refusal, failure.step_length, failure.refusals)
        return Err(TraceFailure(TraceStuck, stopped_context(reason, tracer.point, tracer.steps, context, details)))
    end
    @debug "trace point" _group = :trace step = tracer.steps precision = last(tracer.point) step_length = unwrap(stepped).step_length
    return Ok(unwrap(stepped))
end

"""
The point of `tracer`'s branch at precision `precision`, which its last step reached or passed (`land!`), or the
`TraceStuck` failure of that corrector with `context`.
"""
function landed_point(tracer::CurveTracing.Tracer{E}, precision::E, context::NamedTuple)::Result{Vector{E}, TraceFailure} where {E <: Real}
    landed = CurveTracing.land!(tracer, length(tracer.point), precision)
    if is_error(landed)
        details = (; target = precision, refusal = unwrap_error(landed))
        return Err(TraceFailure(TraceStuck, stopped_context(:landing, tracer.point, tracer.steps, context, details)))
    end
    return Ok(unwrap(landed))
end

"""
`settled_points` on CurveTracing: the accepted points of `gains`' principal branch, to the first whose largest regret is
at most `tolerance`. A step past `maximum_precision` lands there (`landed_point`), which settles or fails
`NoConvergence` with reason `:maximum_precision`.
"""
function engine_points(
        gains::NormalFormGains,
        settings::CurveSettings{E},
        maximum_precision::E,
        tolerance::E,
        context::NamedTuple,
    )::Result{BranchPoints{E}, TraceFailure} where {E <: Real}
    started = curve_init(gains, settings)
    if is_error(started)
        failure = unwrap_error(started)
        cause = ifelse(failure.cause == CurveTracing.NotFiniteStart, NotFinite, Singular)
        return Err(TraceFailure(cause, start_context(failure, logit_count(gains), context)))
    end
    return recorded_points(unwrap(started), BranchRecorder(gains, one(E), tolerance), maximum_precision, settings.max_steps, context)
end

"""`engine_points` on a started `tracer`, each accepted point into `recorder`."""
function recorded_points(
        tracer::CurveTracing.Tracer{E},
        recorder::BranchRecorder,
        maximum_precision::E,
        max_steps::Integer,
        context::NamedTuple,
    )::Result{BranchPoints{E}, TraceFailure} where {E <: Real}
    stepped = capped_step!(tracer, maximum_precision, max_steps, context)
    while !is_error(stepped)
        point = unwrap(stepped)
        if !recorder(logits_at(point), last(point))
            return Ok((; recorder.precisions, recorder.logits))
        end
        if last(point) >= maximum_precision
            return Err(TraceFailure(NoConvergence, stopped_context(:maximum_precision, point, tracer.steps, context, (; maximum_precision))))
        end
        stepped = capped_step!(tracer, maximum_precision, max_steps, context)
    end
    return Err(unwrap_error(stepped))
end

"""
The point of `tracer`'s next accepted step (`curve_step!`), or of its landing at `maximum_precision` once the step
passes it (`landed_point`), or the failure of either.
"""
function capped_step!(
        tracer::CurveTracing.Tracer{E},
        maximum_precision::E,
        max_steps::Integer,
        context::NamedTuple,
    )::Result{Vector{E}, TraceFailure} where {E <: Real}
    stepped = curve_step!(tracer, max_steps, context)
    if is_error(stepped)
        return Err(unwrap_error(stepped))
    end
    if last(tracer.point) < maximum_precision
        return Ok(tracer.point)
    end
    return landed_point(tracer, maximum_precision, context)
end

"""
`logit_solve_lambda` on CurveTracing: the logits of `gains`' principal branch at each precision of `precisions`, from
one trace that lands on each in rising order at its first crossing (`landed_point`), or the first failure.
"""
function engine_lambdas(
        gains::NormalFormGains,
        precisions::AbstractVector{E},
        settings::CurveSettings{E},
    )::Result{Vector{Vector{E}}, TraceFailure} where {E <: Real}
    if any(<(0), precisions)
        throw(ArgumentError("precisions must be non-negative; got $(minimum(precisions))"))
    end
    started = curve_init(gains, settings)
    if is_error(started)
        failure = unwrap_error(started)
        cause = ifelse(failure.cause == CurveTracing.NotFiniteStart, NotFinite, Singular)
        return Err(TraceFailure(cause, start_context(failure, logit_count(gains), (;))))
    end
    return landed_lambdas(unwrap(started), precisions, settings.max_steps)
end

"""`engine_lambdas` on a started `tracer`."""
function landed_lambdas(
        tracer::CurveTracing.Tracer{E},
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

"""The point of `tracer`'s branch at `precision`: steps until one reaches it, then lands there (`landed_point`)."""
function stepped_landing!(tracer::CurveTracing.Tracer{E}, precision::E, max_steps::Integer)::Result{Vector{E}, TraceFailure} where {E <: Real}
    context = (; lam = precision)
    while last(tracer.point) < precision
        stepped = curve_step!(tracer, max_steps, context)
        if is_error(stepped)
            return Err(unwrap_error(stepped))
        end
    end
    return landed_point(tracer, precision, context)
end
