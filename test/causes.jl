# Readers of a `TraceFailure`'s Moshi cause, for the tests and the benchmarks.

using Moshi.Data: isa_variant, variant_name

"""Whether `failure`'s cause is one of the `TraceCause` variants `variants`."""
function failed_as(failure::TraceFailure, variants::Type...)::Bool
    return any(variant -> isa_variant(failure.cause, variant), variants)
end

"""The enum `cause` holds: a `NoConvergence` limit, a `TraceStuck` reason, or another variant's stage."""
function cause_detail(cause::TraceCause.Type)::Enum
    isa_variant(cause, TraceCause.NoConvergence) && return cause.limit
    isa_variant(cause, TraceCause.TraceStuck) && return cause.reason
    return cause.stage
end

"""`cause` as its variant and `cause_detail`, such as `TraceStuck(UnmovedPoint)`. It holds no stall, so it fits a CSV field."""
cause_label(cause::TraceCause.Type)::String = "$(variant_name(cause))($(cause_detail(cause)))"
