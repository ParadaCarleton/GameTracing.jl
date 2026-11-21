using ErrorTypes: Err, Ok, Result, is_error, unwrap, unwrap_error
using LinearAlgebra: dot

"""
    LogitQREProfile(gains, lam, logits)

One point of a normal-form game's logit QRE branch, as pygambit's `LogitQREMixedStrategyProfile`: the precision `lam`
(Gambit's `λ`, `π ∝ exp(λ u)`, the trace's precision) and `profile`, each player's mixed strategy (`mixed_strategies`).
`logits` are the trace's log odds against each player's first strategy; they stay finite where a probability underflows.
"""
struct LogitQREProfile{E <: Real, S <: Tuple, L <: AbstractVector{E}}
    lam::E
    profile::S
    logits::L
    # The only constructor: `profile` is computed from `logits`, so the two cannot disagree.
    function LogitQREProfile(gains::NormalFormGains, lam::E, logits::AbstractVector{E})::LogitQREProfile{E} where {E <: Real}
        stored = collect(logits)
        profile = mixed_strategies(gains, stored)
        return new{E, typeof(profile), typeof(stored)}(lam, profile, stored)
    end
end

# The results of `logit_solve` and of the branch searches, over `N` players in element type `E`, their failures with
# context `C`: `LambdaContext` for `logit_solve_lambda`, `RegretContext` for `logit_solve` and `logit_solve_branch`.
const QREProfile{E, N} = LogitQREProfile{E, NTuple{N, Vector{E}}, Vector{E}}
const ProfileResult{E, N, C} = Result{QREProfile{E, N}, TraceFailure{E, C}}
const BranchResult{E, N, C} = Result{Vector{QREProfile{E, N}}, TraceFailure{E, C}}
# A `logit_solve_lambda` failure's context: the precision its trace was heading to.
const LambdaContext{E} = @NamedTuple{lam::E}
# A `logit_solve` or `logit_solve_branch` failure's context: the regret tolerance, in the trace's element type.
const RegretContext{E} = @NamedTuple{maxregret::E}

"""
    gambit_settings(E, max_accel)

`BifurcationKitSettings` in `E` for pygambit's `max_accel`, which caps each step's growth (`StepControl`'s
`max_step_growth`, and `aggressiveness = max_accel - 1`). BifurcationKit's growth limit must exceed 1, so at
`max_accel = 1` step control only refuses steps. Zero aggressiveness then holds an accepted step's length, where Gambit
also shrinks it after a poor corrector. The first step comes from the game (`opening_step`), as in `PathFollowingSettings`.
The trace steps in `default_step_metric`, which `PathFollowingSettings` shares.
"""
function gambit_settings(::Type{E}, max_accel::Real)::BifurcationKitSettings{E} where {E <: Real}
    if !(max_accel >= 1)
        throw(ArgumentError("max_accel must be at least 1; got $max_accel"))
    end
    growth = E(max_accel)
    return BifurcationKitSettings{E}(;
        aggressiveness = growth - one(E),
        step_metric = default_step_metric(E),
        step_control = StepControl{E}(; max_step_growth = growth > 1 ? growth : nothing),
    )
end

"""The regret of a player whose gains against its first strategy are `gain` and whose mixed strategy is `strategy`."""
function player_regret(gain::AbstractVector{E}, strategy::AbstractVector{E})::E where {E <: Real}
    return maximum(gain; init = zero(E)) - dot(view(strategy, (firstindex(strategy) + 1):lastindex(strategy)), gain)
end

"""The work arrays of `max_regret!` on `gains` at logits like `logits`: a pass's arrays (`pass_workspace`) and the gains."""
function regret_workspace(gains::NormalFormGains, logits::AbstractVector{E})::NamedTuple where {E <: Real}
    return (; pass = pass_workspace(gains, logits), gain = zeros(E, length(logits)))
end

"""
The largest regret of any player of `gains` at `logits`: its best strategy's expected payoff less its mixed strategy's.
The gains and strategies go into `work` (`regret_workspace`).
"""
function max_regret!(work::NamedTuple, gains::NormalFormGains, logits::AbstractVector{E})::E where {E <: Real}
    pass, gain = write_gains!!(work.pass, work.gain, gains, logits)
    return maximum(player_regret.(view.((gain,), logit_ranges(gains)), pass.strategies))
end

"""`max_regret!` in work arrays of its own."""
function max_regret(gains::NormalFormGains, logits::AbstractVector{E})::E where {E <: Real}
    return max_regret!(regret_workspace(gains, logits), gains, logits)
end

"""Refuse `settings` whose element type is not `float` of the payoffs' of `gains`: the trace and its profiles run in the settings' type."""
function require_payoff_type(gains::NormalFormGains, ::EngineSettings{E})::Nothing where {E <: Real}
    if E != float(eltype(gains))
        throw(ArgumentError("settings trace in $E, but the payoffs are $(eltype(gains)); build the settings in $(float(eltype(gains)))"))
    end
    return nothing
end

"""
`high - low` in floating point, so integer payoffs cannot wrap: `Int8` -128 and 127 give 255. Finite extremes whose
difference overflows throw an `ArgumentError`. A non-finite extreme passes through, and its trace fails `NotFinite`.
"""
function payoff_span(low::E, high::E)::float(E) where {E <: Real}
    span = float(high) - float(low)
    if isfinite(low) && isfinite(high) && !isfinite(span)
        throw(ArgumentError("the payoffs from $low to $high span past $(float(E)); use a wider element type"))
    end
    return span
end

"""The largest payoff of `gains` less its smallest, over every player (`payoff_span`): Gambit's scale for `maxregret`."""
function payoff_range(gains::NormalFormGains{N, A})::float(eltype(A)) where {N, A}
    return payoff_span(minimum(minimum, gains.payoffs), maximum(maximum, gains.payoffs))
end

"""
A gap is one payoff less another, so the payoff range bounds it. A constant added to one player's payoffs widens the
range, though the gaps stay; a per-player span is tighter but moves every step sequence (issue #136).
"""
function gap_bound(gains::NormalFormGains{N, A})::float(eltype(A)) where {N, A}
    return payoff_range(gains)
end

"""
    BranchRecorder(gains, tolerance)

A `keep_going` that appends each accepted point of a trace of `gains` to `precisions` and
`logits`, and stops the trace at the first point whose largest regret (`max_regret!`) is at most `tolerance`.
"""
struct BranchRecorder{G <: NormalFormGains, E <: Real, W <: NamedTuple}
    gains::G
    tolerance::E
    precisions::Vector{E}
    logits::Vector{Vector{E}}
    work::W
end

function BranchRecorder(gains::NormalFormGains, tolerance::E)::BranchRecorder{<:NormalFormGains, E} where {E <: Real}
    return BranchRecorder(gains, tolerance, E[], Vector{E}[], regret_workspace(gains, zeros(E, logit_count(gains))))
end

function (recorder::BranchRecorder{G, E})(logits::AbstractVector{E}, precision::Real)::Bool where {G, E <: Real}
    push!(recorder.precisions, precision)
    push!(recorder.logits, collect(logits))
    return max_regret!(recorder.work, recorder.gains, logits) > recorder.tolerance
end

"""The logits at precision `lam` on the principal branch of `gains`, traced with `settings`."""
function lambda_logits(gains::NormalFormGains, lam::E, settings::BifurcationKitSettings{E})::Result{Vector{E}, TraceFailure{E, LambdaContext{E}}} where {E <: Real}
    result = trace_endpoint(LogitGame(gains, lam), logit_count(gains), settings; context = (; lam))
    if is_error(result)
        return Err(unwrap_error(result))
    end
    return Ok(unwrap(result).logits)
end

"""
    logit_solve_lambda(gains, lam; max_accel, settings)

pygambit's `logit_solve_lambda`: the `LogitQREProfile` on the principal branch of the normal-form game `gains`
(`NormalFormGains`) at each precision in `lam`, a number or a vector, in `lam`'s order, or the first trace's failure.
`max_accel` caps the trace's step growth (`gambit_settings`); `settings`, an `EngineSettings`, replaces it. pygambit's
`first_step` has no counterpart: every trace's first step comes from the game (`opening_step`).
`PathFollowingSettings` trace once, their steps stopping at each precision in turn. Every precision must be
non-negative and finite in the trace's element type (`ArgumentError` otherwise). Profiles take `float` of the payoffs'
element type (Float64 for integer payoffs), and `settings` must share it (`ArgumentError` otherwise). A failure's
context is the precision its trace was heading to (`LambdaContext`).
"""
function logit_solve_lambda(
        gains::NormalFormGains{N},
        lam::AbstractVector{<:Real};
        max_accel::Real = GAMBIT_MAX_ACCEL,
        settings::EngineSettings{E} = gambit_settings(float(eltype(gains)), max_accel),
    )::BranchResult{E, N, LambdaContext{E}} where {N, E <: Real}
    require_payoff_type(gains, settings)
    precisions = E.(lam)
    # Checked before and after conversion to `E`, which rounds a tiny negative precision to -0 and a huge one to Inf.
    if !(all(reachable, lam) && all(reachable, precisions))
        throw(ArgumentError("every precision must be non-negative and finite in $E; got $lam"))
    end
    logits = engine_lambdas(gains, precisions, settings)
    if is_error(logits)
        return Err(unwrap_error(logits))
    end
    return Ok(LogitQREProfile.((gains,), precisions, unwrap(logits)))
end

"""
`logit_solve_lambda` on BifurcationKit: the logits of `gains`' principal branch at each precision of `precisions`, one
trace each (`lambda_logits`), or the first failure.
"""
function engine_lambdas(
        gains::NormalFormGains,
        precisions::AbstractVector{E},
        settings::BifurcationKitSettings{E},
    )::Result{Vector{Vector{E}}, TraceFailure{E, LambdaContext{E}}} where {E <: Real}
    # ponytail: one trace per precision; `PathFollowingSettings` land on each precision in one trace instead.
    results = lambda_logits.((gains,), precisions, (settings,))
    for result in results
        if is_error(result)
            return Err(unwrap_error(result))
        end
    end
    return Ok(unwrap.(results))
end

# The vector method's keywords, passed on whole.
function logit_solve_lambda(gains::NormalFormGains{N}, lam::Real; keywords...)::BranchResult{<:Real, N, <:LambdaContext} where {N}
    return logit_solve_lambda(gains, [lam]; keywords...)
end

# The precisions and logits of a branch's accepted points, in trace order.
const BranchPoints{E} = @NamedTuple{precisions::Vector{E}, logits::Vector{Vector{E}}}

"""
The accepted points of `logit_solve_branch`'s trace of `gains`, as precisions and logits, or its failure. Takes
`logit_solve_branch`'s keywords and defaults; the profiles are left to the caller, which may want only the last.
"""
function settled_points(
        gains::NormalFormGains;
        max_accel::Real = GAMBIT_MAX_ACCEL,
        maximum_precision::Real = 1.0f6,
        settings::EngineSettings{E} = gambit_settings(float(eltype(gains)), max_accel),
        maxregret::Real = sqrt(eps(element_type(settings))),
    )::Result{BranchPoints{E}, TraceFailure{E, RegretContext{E}}} where {E <: Real}
    if !(maxregret > 0)
        throw(ArgumentError("maxregret must be positive; got $maxregret"))
    end
    # Checked in `E`, where a tiny or huge cap rounds to 0 or Inf. A cap of 0 ends the trace at its start.
    if !(reachable(E(maximum_precision)) && E(maximum_precision) > 0)
        throw(ArgumentError("maximum_precision must be positive and finite in $E; got $maximum_precision"))
    end
    require_payoff_type(gains, settings)
    return engine_points(gains, settings, E(maximum_precision), E(maxregret) * E(payoff_range(gains)), (; maxregret = E(maxregret)))
end

"""
`settled_points` on BifurcationKit: the accepted points of a trace of `gains` to `maximum_precision`, to the first
whose largest regret is at most `tolerance`, or the failure with `context`. A trace that reaches `maximum_precision`
unsettled fails `NoConvergence` (`MaximumPrecision`) there.
"""
function engine_points(
        gains::NormalFormGains,
        settings::BifurcationKitSettings{E},
        maximum_precision::E,
        tolerance::E,
        context::C,
    )::Result{BranchPoints{E}, TraceFailure{E, C}} where {E <: Real, C <: NamedTuple}
    recorder = BranchRecorder(gains, tolerance)
    result = trace_endpoint(LogitGame(gains, maximum_precision), logit_count(gains), settings; keep_going = recorder, context)
    settled = !isempty(recorder.logits) && max_regret!(recorder.work, gains, last(recorder.logits)) <= recorder.tolerance
    settled && return Ok((; recorder.precisions, recorder.logits))
    is_error(result) && return Err(unwrap_error(result))
    return Err(TraceFailure(unwrap(result), maximum_precision, context))
end

"""
    TraceFailure(traced::Traced, maximum_precision, context)

The failure of a trace that reached `maximum_precision` as `traced` but did not settle there: `NoConvergence`
(`MaximumPrecision`) at its endpoint. The engine ran to its end, so it holds no step and no record.
"""
function TraceFailure(traced::Traced{E}, maximum_precision::E, context::C)::TraceFailure{E, C} where {E <: Real, C <: NamedTuple}
    stop = TraceStop(traced.logits, traced.logits ./ maximum_precision, maximum_precision, traced.steps, zero(E))
    return TraceFailure(TraceCause.NoConvergence{E}(MaximumPrecision), stop, EngineRecord.NoRecord{E}(), context)
end

"""
    logit_solve_branch(gains; maxregret, max_accel, maximum_precision, settings)

pygambit's `logit_solve_branch`: every accepted point of the principal branch of the normal-form game `gains`, from
the first step to the first point where no player's regret passes `maxregret` times the payoff range. The trace runs
to `maximum_precision` at most, and fails `NoConvergence` (`MaximumPrecision`) if it ends there unsettled; any other
failure of the trace is returned as is. A failure's context holds `maxregret` in the trace's element type `E`
(`RegretContext`). `maxregret` defaults to `sqrt(eps(E))`. `max_accel` and `settings` are `logit_solve_lambda`'s.
"""
function logit_solve_branch(gains::NormalFormGains{N}; keywords...)::BranchResult{<:Real, N, <:RegretContext} where {N}
    return branch_profiles(gains, settled_points(gains; keywords...))
end

"""Every point of `points` (`settled_points`) as a profile of `gains`, or the failure."""
function branch_profiles(gains::NormalFormGains{N}, points::Result{BranchPoints{E}, TraceFailure{E, C}})::BranchResult{E, N, C} where {N, E <: Real, C <: NamedTuple}
    if is_error(points)
        return Err(unwrap_error(points))
    end
    return Ok(LogitQREProfile.((gains,), unwrap(points).precisions, unwrap(points).logits))
end

"""
    logit_solve(gains; maxregret, max_accel, maximum_precision, settings)

pygambit's `logit_solve`: the last point of `logit_solve_branch`, an approximate Nash equilibrium of the normal-form
game `gains` at which no player's regret passes `maxregret` times the payoff range. Only that point's profile is built.
"""
function logit_solve(gains::NormalFormGains{N}; keywords...)::ProfileResult{<:Real, N, <:RegretContext} where {N}
    return last_profile(gains, settled_points(gains; keywords...))
end

"""The last point of `points` (`settled_points`) as a profile of `gains`, or the failure."""
function last_profile(gains::NormalFormGains{N}, points::Result{BranchPoints{E}, TraceFailure{E, C}})::ProfileResult{E, N, C} where {N, E <: Real, C <: NamedTuple}
    if is_error(points)
        return Err(unwrap_error(points))
    end
    return Ok(LogitQREProfile(gains, last(unwrap(points).precisions), last(unwrap(points).logits)))
end
