using CommonSolve: CommonSolve
using CurveTracing: CurveTracing
using ErrorTypes: Err, Ok, Result, is_error, unwrap, unwrap_error
using LinearSolve: LinearSolve
using Logging: @debug

"""
    CurveSettings{E}(; first_step, max_step, max_steps, logit_tolerance, settle_ulps, reach, newton_iterations,
                     contraction_limit, nominal_contraction, nominal_distance, nominal_turn, growth_limit,
                     max_refusals, resolution_ulps, linear_solver)

The controls of a trace on CurveTracing's pseudo-arclength continuation, an `EngineSettings` for `logit_solve`. The
trace runs in `[w; λ]`, the payoff gaps `w = z / λ` and the precision `λ`. Its steps are Euclidean arclength there.

- `first_step`, `max_step`: the first and the largest step.
- `max_steps`: accepted steps before the trace fails as `NoConvergence`.
- `logit_tolerance`, `settle_ulps`: the corrector converges on a correction that moves no logit more than
  `logit_tolerance` past `settle_ulps` ulps of its own size (`LogitSettled`).
- `reach`: refuse a corrected point whose logits move further from the predictor's than `reach` times the
  predictor's own largest logit move (`WithinReach`). `Inf32` switches the test off.
- `newton_iterations`, `contraction_limit`: CurveTracing's `Newton`, its corrections in logits (`logit_change`).
- `nominal_contraction`, `nominal_distance`, `nominal_turn`, `growth_limit`: CurveTracing's `StepControl`. The
  distance is the first correction's largest logit move; `nothing` drops a term.
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
    max_refusals::Int
    resolution_ulps::Int
    linear_solver::L
end

# Defaults: the faster of two step-control arms on one game per LogitNash benchmark size (nominal distance 1 and turn
# 0.3 rad, growth 4, against 0.4, 0.2 and 2); `benchmark/logitnash_speed.jl` compares the engines.
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
        nominal_distance::Union{Nothing, Real} = 1.0f0,
        nominal_turn::Union{Nothing, Real} = 0.3f0,
        growth_limit::Real = 4.0f0,
        max_refusals::Integer = 80,
        resolution_ulps::Integer = 4,
        linear_solver::L = LinearSolve.LUFactorization(),
    )::CurveSettings{E, L} where {E <: Real, L}
    return CurveSettings{E, L}(
        E(first_step), E(max_step), Int(max_steps), E(logit_tolerance), Int(settle_ulps), E(reach), Int(newton_iterations),
        E(contraction_limit), optional_limit(E, nominal_contraction), optional_limit(E, nominal_distance),
        optional_limit(E, nominal_turn), E(growth_limit), Int(max_refusals), Int(resolution_ulps), linear_solver,
    )
end

"""`limit` in `E`, or `nothing` where it is off."""
optional_limit(::Type{<:Real}, ::Nothing)::Nothing = nothing
function optional_limit(::Type{E}, limit::Real)::E where {E <: Real}
    return E(limit)
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

"""CurveTracing's `PseudoArclength` for `settings` over `width` payoff gaps."""
function curve_algorithm(settings::CurveSettings{E}, width::Integer)::CurveTracing.PseudoArclength where {E <: Real}
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
        metric = CurveTracing.FixedWeights(ones(E, width + 1)),
        settings.linear_solver,
        accept = WithinReach(settings.reach),
        settings.resolution_ulps,
        settings.max_refusals,
    )
end

"""
CurveTracing's `init` of the principal branch of `gains` in `[w; λ]`: a `Tracer` from the flat start toward rising
`λ`, or the `StepFailure` there (`start_failure`).
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
        curve_algorithm(settings, width);
        settings.first_step,
    )
end

"""The `TraceFailure` of CurveTracing's start `failure` over `width` logits: `NotFinite` or `Singular`, with `context`."""
function start_failure(failure::CurveTracing.StepFailure{E}, width::Integer, context::NamedTuple)::TraceFailure where {E <: Real}
    cause = ifelse(failure.cause == CurveTracing.NotFiniteStart, NotFinite, Singular)
    return TraceFailure(cause, (; context..., reason = :start, precision = zero(E), logits = zeros(E, width), steps = 0))
end

"""
The `TraceFailure` of a trace that stopped at `point = [w; λ]` after `steps` steps for `reason`: `cause`, with
`context`, the precision and logits there, the steps and `details`.
"""
function curve_failure(
        cause::TraceCause,
        reason::Symbol,
        point::AbstractVector{E},
        steps::Integer,
        context::NamedTuple,
        details::NamedTuple,
    )::TraceFailure where {E <: Real}
    @debug "trace stopped" _group = :trace reason
    return TraceFailure(
        cause,
        (; context..., reason, precision = last(point), logits = collect(logits_at(point)), steps, details...),
    )
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
        return Err(curve_failure(NoConvergence, :max_steps, tracer.point, tracer.steps, context, (;)))
    end
    stepped = CommonSolve.step!(tracer)
    if is_error(stepped)
        failure = unwrap_error(stepped)
        reason = ifelse(failure.cause == CurveTracing.Unresolved, :unresolved_step, :refusals)
        details = (; failure.last_refusal, failure.step_length, failure.refusals)
        return Err(curve_failure(TraceStuck, reason, tracer.point, tracer.steps, context, details))
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
        return Err(curve_failure(TraceStuck, :landing, tracer.point, tracer.steps, context, (; target = precision, refusal = unwrap_error(landed))))
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
        return Err(start_failure(unwrap_error(started), logit_count(gains), context))
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
    stepped = curve_step!(tracer, max_steps, context)
    while !is_error(stepped)
        point = tracer.point
        if last(point) >= maximum_precision
            landed = landed_point(tracer, maximum_precision, context)
            if is_error(landed)
                return Err(unwrap_error(landed))
            end
            point = unwrap(landed)
        end
        if !recorder(logits_at(point), last(point))
            return Ok((; recorder.precisions, recorder.logits))
        end
        if last(point) >= maximum_precision
            return Err(curve_failure(NoConvergence, :maximum_precision, point, tracer.steps, context, (; maximum_precision)))
        end
        stepped = curve_step!(tracer, max_steps, context)
    end
    return Err(unwrap_error(stepped))
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
        return Err(start_failure(unwrap_error(started), logit_count(gains), (;)))
    end
    return landed_lambdas(unwrap(started), precisions, settings.max_steps)
end

"""`engine_lambdas` on a started `tracer`."""
function landed_lambdas(
        tracer::CurveTracing.Tracer{E},
        precisions::AbstractVector{E},
        max_steps::Integer,
    )::Result{Vector{Vector{E}}, TraceFailure} where {E <: Real}
    logits = Vector{Vector{E}}(undef, length(precisions))
    for index in sortperm(precisions)
        context = (; lam = precisions[index])
        while last(tracer.point) < precisions[index]
            stepped = curve_step!(tracer, max_steps, context)
            if is_error(stepped)
                return Err(unwrap_error(stepped))
            end
        end
        landed = landed_point(tracer, precisions[index], context)
        if is_error(landed)
            return Err(unwrap_error(landed))
        end
        logits[index] = collect(logits_at(unwrap(landed)))
    end
    return Ok(logits)
end
