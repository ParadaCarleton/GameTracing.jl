using ErrorTypes: Result
using LinearAlgebra: LAPACKException, SingularException
using Moshi.Data: @data
using Moshi.Derive: @derive
using PathFollowing: PathFollowing

"""Which limit a trace that did not converge spent: its `max_steps`, or `logit_solve`'s `maximum_precision`, unsettled."""
@enum StopLimit MaxSteps MaximumPrecision

"""
Why a trace stuck short of its target: a predictor below the float type's resolution (`UnresolvedStep`), more refusals
in one PathFollowing step than `max_refusals` (`StepRefusals`), a BifurcationKit point equal to the last
(`UnmovedPoint`), a BifurcationKit start that would not correct (`UncorrectedStart`), or a `keep_going` or a refusal
that stopped it (`StoppedShort`).
"""
@enum StuckReason UnresolvedStep StepRefusals UnmovedPoint UncorrectedStart StoppedShort

"""Where a trace failed: at its flat start (`AtStart`), on its way to the target (`MidTrace`) or at the endpoint check (`AtEndpoint`)."""
@enum TraceStage AtStart MidTrace AtEndpoint

"""
    StalledCorrection(precision, correction, iterate, pin, arclength)

A Newton correction that stalled at rounding noise: its record's `precision`, its last `correction` in Gambit's norm, its
iterate `[w; λ]`, and the `pin` and `arclength` of its step. The endpoint check's correction has its pin at the
endpoint's precision and arclength zero.
"""
# No `E <: Real` bound: Moshi's `@derive Eq` of `TraceCause` builds this field's type at `E = Any`.
struct StalledCorrection{E}
    precision::E
    correction::E
    iterate::Vector{E}
    pin::E
    arclength::E
end

@derive StalledCorrection[Eq, Hash]

"""
Why a trace stopped short of its target precision, a Moshi sum in the trace's element type `E`, one variant per cause.
A variant with no field of type `E` takes it explicitly: `TraceCause.NotFinite{E}(AtStart)`.

- `NoConvergence(limit)`: the trace spent its `StopLimit`.
- `TraceStuck(reason)`: the trace stuck for its `StuckReason`.
- `NotFinite(stage)`: the game was not finite `AtStart` or `AtEndpoint` (`TraceStage`).
- `Singular(stage)`: a linear solve broke `AtStart`, `MidTrace` or `AtEndpoint`.
- `PrecisionExhausted(stage, stall)`: Newton's corrections stopped contracting at the float type's rounding noise
  (`StalledCorrection`), `MidTrace` (`CorrectorReach`) or `AtEndpoint`; a wider float type goes further.
"""
@data TraceCause{E <: Real} begin
    struct NoConvergence
        limit::StopLimit
    end
    struct TraceStuck
        reason::StuckReason
    end
    struct NotFinite
        stage::TraceStage
    end
    struct Singular
        stage::TraceStage
    end
    struct PrecisionExhausted
        stage::TraceStage
        stall::StalledCorrection{E}
    end
end

@derive TraceCause[Show, Eq, Hash]

# Moshi declares each variant `public`, but not `Type`, the sum's own name.
@eval TraceCause $(Expr(:public, :Type))

"""
    TraceStop(logits, gaps, precision, steps, step_length)

Where a failed trace stopped: its last accepted point's `logits`, payoff `gaps` `w` and `precision`, its accepted
`steps`, and the step it held (`step_length`), zero when it held none. A non-finite start holds zero logits at
precision zero beside its non-finite gaps.
"""
struct TraceStop{E <: Real}
    logits::Vector{E}
    gaps::Vector{E}
    precision::E
    steps::Int
    step_length::E
end

@derive TraceStop[Eq, Hash]

"""
The trace engine's own record where a trace failed, a Moshi sum in the trace's element type `E`. A variant with no
field of type `E` takes it explicitly: `EngineRecord.NoRecord{E}()`.

- `NoRecord()`: no engine record, as no engine ran (a non-finite start), the engine ran to its end (a `logit_solve`
  cap reached unsettled), or GameTracing's own check failed a PathFollowing trace.
- `BifurcationKitTrace(accepted_step, precision_min, precision_max, logit_tangent, gap_tangent, tangent_precision,
  logit_accuracy, message)`: the arclength step of the last accepted point (zero when none was accepted), the
  precision's range, the tangent there in logits (`logit_tangent`, `dz = λ dw + w dλ`) and in payoff gaps
  (`gap_tangent`, `tangent_precision`), the largest last correction in Gambit's norm of an accepted point, and the caught
  exception's `showerror` text (empty when none ended the trace).
- `PathFollowingStep(failure)`: the `PathFollowing.StepFailure` that ended the trace.
"""
@data EngineRecord{E <: Real} begin
    NoRecord()
    struct BifurcationKitTrace
        accepted_step::E
        precision_min::E
        precision_max::E
        logit_tangent::Vector{E}
        gap_tangent::Vector{E}
        tangent_precision::E
        logit_accuracy::E
        message::String
    end
    struct PathFollowingStep
        failure::PathFollowing.StepFailure.Type{E}
    end
end

@derive EngineRecord[Show, Eq, Hash]

# As for `TraceCause`.
@eval EngineRecord $(Expr(:public, :Type))

"""
    TraceFailure(cause, stop, engine, context)

A trace that did not reach its target: why (`cause`, a `TraceCause`), where (`stop`, a `TraceStop`), the engine's
record there (`engine`, an `EngineRecord`) and the caller's own `context`, a `NamedTuple` kept apart from the trace's
fields.
"""
struct TraceFailure{E <: Real, C <: NamedTuple}
    cause::TraceCause.Type{E}
    stop::TraceStop{E}
    engine::EngineRecord.Type{E}
    context::C
end

@derive TraceFailure[Eq, Hash]

"""
    Traced(logits, steps, logit_accuracy)

The `logits` of a trace at its target precision, a root of the game in effect there, the continuation `steps` it took
over every segment, and `logit_accuracy`, a move in Gambit's norm of the log probabilities (`LogProbabilityChange`).
On BifurcationKit that is the largest last Newton move of any accepted point and of the endpoint's check
(`verified_endpoint`), at most the corrector's `logit_tolerance` past each strategy's `settle_ulps`
(`log_probability_corrections`). On PathFollowing it is the endpoint check's correction (`verified_landing`), at most
`endpoint_tolerance` past each strategy's `endpoint_ulps`.
"""
struct Traced{E <: Real}
    logits::Vector{E}
    steps::Int
    logit_accuracy::E
end

"""
A trace's outcome, an ErrorTypes `Result`: `Ok(Traced)`, or `Err(TraceFailure)` with the caller's context of type `C`.
A caller reaches the logits through `unwrap`, `@?` or a check of `is_error`, never by field access on the result;
`unwrap` on a failure throws at the caller.
"""
const TraceResult{E <: Real, C <: NamedTuple} = Result{Traced{E}, TraceFailure{E, C}}

"""
    uncorrected_start(exception)

Whether `exception` is BifurcationKit's report that a start point would not correct: a bare
`"Stopping continuation."` at the first point, an `ErrorException` at the second.
"""
uncorrected_start(::Any)::Bool = false
uncorrected_start(message::String)::Bool = message == "Stopping continuation."
# `msg` is an `AbstractString` field; the `String` test and assertion let `startswith` and `contains` compile against it.
uncorrected_start(exception::ErrorException)::Bool = exception.msg isa String &&
    startswith(exception.msg::String, "Newton failed to converge")

"""
    solver_blowup(exception)

Whether `exception` is a broken linear solve or non-finite factorization: `SingularException`,
`LAPACKException`, or LAPACK's `ArgumentError` for a NaN or infinite matrix. A bare `ArgumentError`
is a caller's contract violation and propagates.
"""
solver_blowup(::Any)::Bool = false
solver_blowup(::SingularException)::Bool = true
solver_blowup(::LAPACKException)::Bool = true
function solver_blowup(exception::ArgumentError)::Bool
    return exception.msg isa String &&
        (contains(exception.msg::String, "LAPACK") || contains(exception.msg::String, "matrix contains Infs or NaNs"))
end
