# The trace-geometry arms in payoff gaps: Fisher gaps beside a step in λ or in log(1 + λ), with the shipped reach acceptance,
# each at three state weights.
const ARMS = (
    fisher_reach = (; distance = FisherDistance),
    fisher_log_precision_reach = (; distance = FisherLogPrecisionDistance),
)
const ARM_WEIGHTS = (0.01f0, 0.03f0, 0.1f0)

"""The `BifurcationKitSettings` of `arm` (an `ARMS` entry) at state weight `state_weight`, with `keywords` passed on."""
function arm_settings(arm::NamedTuple, state_weight::E; keywords...)::BifurcationKitSettings{E} where {E <: Real}
    geometry = TraceGeometry{E}(; arm.distance, state_weight)
    return BifurcationKitSettings{E}(; step_metric = geometry, keywords...)
end

"""Every arm at every weight, as `label => settings` pairs."""
function arm_variants()::Vector{Pair{String, BifurcationKitSettings{Float32}}}
    return [string(name, " state_weight ", weight) => arm_settings(arm, weight) for (name, arm) in pairs(ARMS) for weight in ARM_WEIGHTS]
end
