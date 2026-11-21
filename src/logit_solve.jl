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
`logit_solve`'s default `stall_window` (`regret_stalled`). On 450 games (GAMUT, maxregret_reach, the fold set) in
Float32 and Float64 on both engines, traced past the default `maxregret` with no cap, a trace that settled ran at most
100 rising points with no new least regret before it settled; 200 is twice that. At the floor, regret takes a few
rounded values: of 152 traces run to 20,000 points whose least regret did not halve over their second half, the window
ended 87. The others still set a new least regret every few dozen points, as `λ` creeps up.
"""
const DEFAULT_STALL_WINDOW = 200

"""
    BranchRecorder(gains, tolerance, window)

A `keep_going` that appends each accepted point of a trace of `gains` to `precisions` and `logits`, its largest regret
(`max_regret!`) to `regrets` and the least regret so far to `least`. It stops the trace at the first point whose regret
is at most `tolerance`, or where the regret stalls at its rounding floor over `window` points (`regret_stalled`).
"""
struct BranchRecorder{G <: NormalFormGains, E <: Real, W <: NamedTuple}
    gains::G
    tolerance::E
    window::Int
    precisions::Vector{E}
    logits::Vector{Vector{E}}
    regrets::Vector{E}
    least::Vector{E}
    work::W
end

function BranchRecorder(gains::NormalFormGains, tolerance::E, window::Integer)::BranchRecorder{<:NormalFormGains, E} where {E <: Real}
    window > 0 || throw(ArgumentError("stall_window must be positive; got $window"))
    return BranchRecorder(gains, tolerance, Int(window), E[], Vector{E}[], E[], E[], regret_workspace(gains, zeros(E, logit_count(gains))))
end

function (recorder::BranchRecorder{G, E})(logits::AbstractVector{E}, precision::Real)::Bool where {G, E <: Real}
    regret = max_regret!(recorder.work, recorder.gains, logits)
    push!(recorder.precisions, precision)
    push!(recorder.logits, collect(logits))
    push!(recorder.regrets, regret)
    push!(recorder.least, min(regret, isempty(recorder.least) ? regret : last(recorder.least)))
    return !settled_regret(recorder) && !regret_stalled(recorder)
end

"""Whether `recorder`'s last point settled: its largest regret is at most the tolerance."""
function settled_regret(recorder::BranchRecorder)::Bool
    return !isempty(recorder.regrets) && last(recorder.regrets) <= recorder.tolerance
end

"""
Whether `recorder`'s regret stalled at its rounding floor: its last `window` points each rose in precision and none
set a new least regret. Near an equilibrium the regret falls as `λ` rises, like `1 / λ` at a mixed one, so a regret
that holds over the window has met the float type's rounding. A fold, where `λ` falls, restarts the window.
"""
function regret_stalled(recorder::BranchRecorder)::Bool
    (; precisions, least, window) = recorder
    count = length(precisions)
    count > window || return false
    before = count - window
    # `lt = <=` makes `issorted` strict: each precision lies above the one before.
    return least[count] == least[before] && issorted(view(precisions, before:count); lt = <=)
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
        maximum_precision::Union{Nothing, Real} = nothing,
        settings::EngineSettings{E} = gambit_settings(float(eltype(gains)), max_accel),
        maxregret::Real = default_maxregret(element_type(settings)),
        stall_window::Integer = DEFAULT_STALL_WINDOW,
    )::Result{BranchPoints{E}, TraceFailure{E, RegretContext{E}}} where {E <: Real}
    if !(maxregret > 0)
        throw(ArgumentError("maxregret must be positive; got $maxregret"))
    end
    require_payoff_type(gains, settings)
    target = trace_ceiling(settings, maximum_precision)
    recorder = BranchRecorder(gains, E(maxregret) * E(payoff_range(gains)), stall_window)
    context = (; maxregret = E(maxregret))
    return branch_outcome(recorder, engine_points(gains, settings, target, recorder, context), !isnothing(maximum_precision), context)
end

"""
The precision a `logit_solve` trace under `settings` runs to: `maximum_precision`, checked positive and finite in `E`
(`ArgumentError`), or `precision_ceiling(E)` for none.
"""
function trace_ceiling(::EngineSettings{E}, maximum_precision::Real)::E where {E <: Real}
    # Checked in `E`, where a tiny or huge cap rounds to 0 or Inf. A cap of 0 ends the trace at its start.
    if !(reachable(E(maximum_precision)) && E(maximum_precision) > 0)
        throw(ArgumentError("maximum_precision must be positive and finite in $E; got $maximum_precision"))
    end
    return E(maximum_precision)
end

function trace_ceiling(settings::EngineSettings{E}, ::Nothing)::E where {E <: Real}
    require_target_free(settings)
    return precision_ceiling(E)
end

"""
    precision_ceiling(E)

The precision of a `logit_solve` trace with no `maximum_precision`: `√floatmax(E)`, the largest whose square is finite,
as a plain norm at the target squares it (`plain_weights`). The regret stalls long before it (`regret_stalled`).
"""
function precision_ceiling(::Type{E})::E where {E <: AbstractFloat}
    return sqrt(floatmax(E))
end

"""
Throw an `ArgumentError` unless `settings` step in a norm free of the target precision: a BifurcationKit trace measures
its steps at the target (`StepDistance`), so only `FisherDistance` with `LogPrecisionTerm` runs to `precision_ceiling`.
PathFollowing measures at target one and runs to any.
"""
function require_target_free(settings::BifurcationKitSettings)::Nothing
    (; distance, precision_term) = settings.step_metric
    if !(distance == FisherDistance && precision_term == LogPrecisionTerm)
        throw(ArgumentError("$distance with $precision_term scales its steps by the target precision; pass a maximum_precision"))
    end
    return nothing
end

"""
`settled_points` on BifurcationKit: a trace of `gains` to `target`, each accepted point into `recorder`, which stops
it once it settles or stalls. `Ok(nothing)` when it ended so or at the target, else its failure with `context`.
"""
function engine_points(
        gains::NormalFormGains,
        settings::BifurcationKitSettings{E},
        target::E,
        recorder::BranchRecorder,
        context::C,
    )::Result{Nothing, TraceFailure{E, C}} where {E <: Real, C <: NamedTuple}
    result = trace_endpoint(LogitGame(gains, target), logit_count(gains), settings; keep_going = recorder, context)
    is_error(result) && return Err(unwrap_error(result))
    return Ok(nothing)
end

"""
The points of `recorder` once its trace `ended` (`engine_points`, `Ok(nothing)` or the engine's failure), with
`context`, checked the same on both engines. A settled last point returns every point. A regret stall fails
`PrecisionExhausted` (`regret_stall`); so does a trace that reached `precision_ceiling` unsettled. Else the engine's
failure stands, and a trace that reached a set `maximum_precision` (`capped`) unsettled fails `NoConvergence`
(`MaximumPrecision`).
"""
function branch_outcome(
        recorder::BranchRecorder{G, E},
        ended::Result{Nothing, TraceFailure{E, C}},
        capped::Bool,
        context::C,
    )::Result{BranchPoints{E}, TraceFailure{E, C}} where {G, E <: Real, C <: NamedTuple}
    settled_regret(recorder) && return Ok((; recorder.precisions, recorder.logits))
    if regret_stalled(recorder) || (!is_error(ended) && !capped)
        return Err(TraceFailure(regret_stall(recorder), recorded_stop(recorder), EngineRecord.NoRecord{E}(), context))
    end
    is_error(ended) && return Err(unwrap_error(ended))
    return Err(TraceFailure(TraceCause.NoConvergence{E}(MaximumPrecision), recorded_stop(recorder), EngineRecord.NoRecord{E}(), context))
end

"""The `TraceStop` at `recorder`'s last point, after one step per recorded point, holding no step."""
function recorded_stop(recorder::BranchRecorder{G, E})::TraceStop{E} where {G, E <: Real}
    logits, precision = last(recorder.logits), last(recorder.precisions)
    return TraceStop(logits, logits ./ precision, precision, length(recorder.precisions), zero(E))
end

"""
The cause of a regret stall at `recorder`'s last point `[w; λ]`: `PrecisionExhausted` (`AtEndpoint`), its
`StalledCorrection` the endpoint check's Newton correction there with `λ` pinned (`pinned_correction`), its move in
Gambit's norm with no rounding allowance. A correction that fails gives its own cause (`NotFinite`, `Singular`).
"""
function regret_stall(recorder::BranchRecorder{G, E})::TraceCause.Type{E} where {G, E <: Real}
    (; gains) = recorder
    logits, precision = last(recorder.logits), last(recorder.precisions)
    point = [logits ./ precision; precision]
    pinned = pinned_correction(LogitGame(gains, precision), point, pass_workspace(gains, logits))
    is_error(pinned) && return unwrap_error(pinned)
    (; corrected) = unwrap(pinned)
    correction, _ = log_probability_corrections(point, corrected, checked_ranges(logit_ranges(gains), length(logits)), zero(E), one(E))
    return TraceCause.PrecisionExhausted(AtEndpoint, StalledCorrection(precision, correction, corrected, precision, zero(E)))
end

"""
    default_maxregret(E)

`logit_solve`'s default `maxregret` in element type `E`: pygambit's 1e-8, or `eps(E)^(2/3)` where `E` cannot resolve
it. That is 1e-8 in Float64 and 2.4e-5 in Float32.
"""
function default_maxregret(::Type{E})::E where {E <: AbstractFloat}
    return max(E(1 // 10^8), cbrt(eps(E))^2)
end

"""
    logit_solve_branch(gains; maxregret, max_accel, maximum_precision, stall_window, settings)

pygambit's `logit_solve_branch`: every accepted point of the principal branch of the normal-form game `gains`, from
the first step to the first point where no player's regret passes `maxregret` times the payoff range. The trace has no
cap on the precision, as Gambit's has none. It ends where the regret stalls at its rounding floor for `stall_window`
points (`regret_stalled`, `DEFAULT_STALL_WINDOW`), and fails `PrecisionExhausted` (`AtEndpoint`) there. An optional
`maximum_precision` caps it, and a trace that ends there unsettled fails `NoConvergence` (`MaximumPrecision`). Any
other failure of the trace is returned as is. A failure's context holds `maxregret` in the trace's element type `E`
(`RegretContext`). `maxregret` defaults to `default_maxregret(E)`. `max_accel` and `settings` are `logit_solve_lambda`'s.
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
    logit_solve(gains; maxregret, max_accel, maximum_precision, stall_window, settings)

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
