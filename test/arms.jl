# The trace-geometry arms in payoff gaps: the Fisher distance with each precision term, a step in λ or in log(1 + λ),
# with the shipped reach acceptance.
const ARMS = (
    fisher_reach = (; precision_term = TargetPrecisionTerm),
    fisher_log_precision_reach = (; precision_term = LogPrecisionTerm),
)
# The BifurcationKit arms' state weights in the tests and the full benchmark, about the default 0.1. Below 0.03 the log
# term exhausts at the folding 3x3 game and at `UNFOLDED_PITCHFORK_GAME` near λ 0.808, a known failure that
# `logitnash_tests.jl` pins at 0.01.
const ARM_WEIGHTS = (0.03f0, 0.1f0, 0.3f0)
# The PathFollowing arms' state weights in the full benchmark, about the log term's fewest steps at 0.1.
const PATH_WEIGHTS = (0.03f0, 0.1f0, 0.3f0)

"""The `TraceGeometry` of `arm` (an `ARMS` entry) at state weight `state_weight`."""
function arm_geometry(arm::NamedTuple, state_weight::E)::TraceGeometry{E} where {E <: Real}
    return TraceGeometry{E}(; distance = FisherDistance, arm.precision_term, state_weight)
end

"""The `BifurcationKitSettings` of `arm` at state weight `state_weight`, with `keywords` passed on."""
function arm_settings(arm::NamedTuple, state_weight::E; keywords...)::BifurcationKitSettings{E} where {E <: Real}
    return BifurcationKitSettings{E}(; step_metric = arm_geometry(arm, state_weight), keywords...)
end

"""Every arm at each state weight of `weights`, as `label => settings` pairs."""
function arm_variants(weights::Tuple)::Vector{Pair{String, BifurcationKitSettings{Float32}}}
    return [string(name, " state_weight ", weight) => arm_settings(arm, weight) for (name, arm) in pairs(ARMS) for weight in weights]
end

"""PathFollowing's shipped settings and every arm at each state weight of `weights`, as `label => settings` pairs."""
function path_variants(weights::Tuple)::Vector{Pair{String, PathFollowingSettings{Float32}}}
    arms = [
        string("path ", name, " state_weight ", weight) => PathFollowingSettings{Float32}(; step_metric = arm_geometry(arm, weight))
            for (name, arm) in pairs(ARMS) for weight in weights
    ]
    return ["path shipped" => PathFollowingSettings{Float32}(); arms]
end
