using ErrorTypes: Result
using LinearAlgebra: LAPACKException, SingularException
"""
Why a trace stopped short of its target precision. `PrecisionExhausted`: Newton's corrections stopped contracting at
the float type's rounding noise, before the target (`CorrectorReach`); a wider float type goes further.
"""
@enum TraceCause NoConvergence TraceStuck NotFinite Singular PrecisionExhausted

"""
    TraceFailure(cause, context)

A trace that did not reach its target: `cause` names why and `context` carries the last accepted `logits`
and precision. A `trace_endpoint` trace that left its start also carries that point's payoff `gaps` `w`, the tangent
there in logits (`logit_tangent`, `dz = λ dw + w dλ`) and in payoff gaps (`gap_tangent`, `tangent_precision`), and the
step it held; a non-finite start has none of these. A `PrecisionExhausted` failure also carries the Newton iterate
`stalled_iterate = [w; λ]` whose `correction` in logits stalled, at precision `lambda`. The caller's own `context`
fields sit beside these (`failure_context`).
"""
struct TraceFailure{C <: NamedTuple}
    cause::TraceCause
    context::C
end

"""
    failure_context(context, fields)

A `TraceFailure` context: the caller's `context` beside the trace's own `fields`. A caller field named as a trace field
throws, so neither hides the other.
"""
function failure_context(context::NamedTuple, fields::NamedTuple)::NamedTuple
    shared = intersect(keys(context), keys(fields))
    if !isempty(shared)
        throw(ArgumentError("the caller's context names the trace's own fields $(join(shared, ", "))"))
    end
    return merge(context, fields)
end

"""
    Traced(logits, steps, logit_accuracy)

The `logits` of a trace at its target precision, a root of the game in effect there, the continuation `steps` it took
over every segment, and `logit_accuracy`: the largest last Newton correction in logits of any accepted point, at most
the corrector's `logit_tolerance` plus `settle_ulps` ulps of the largest logit it moved.
"""
struct Traced{E <: Real}
    logits::Vector{E}
    steps::Int
    logit_accuracy::E
end

"""
A trace's outcome, an ErrorTypes `Result`: `Ok(Traced)`, or `Err(TraceFailure)` whose context holds the last
accepted `logits`. A caller reaches the logits through `unwrap`, `@?` or a check of `is_error`, never by field
access on the result; `unwrap` on a failure throws at the caller.
"""
const TraceResult{E <: Real} = Result{Traced{E}, TraceFailure}

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
