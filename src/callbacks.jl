using BifurcationKit: BifurcationKit
using Logging: @debug
"""
    CorrectorSettings{E}(; reach, max_kl, logit_tolerance, settle_ulps, pin_tolerance, contraction_limit,
                         contraction_offset, noise_ratio, noise_limit, newton_iterations)

The corrector's controls (`CorrectorReach`); `nothing` switches a limit off. `reach`: a corrector that moves a logit
further from the predictor than this times the predictor's own largest logit move is refused. `shift_share`: the share
of that reach a redraw's predicted root shift may use (`RootShift`). `max_kl`: a corrected point that moves any player's mixed
strategy further from the predictor's than this KL is refused (`largest_kl`); the game's pass must define `logit_ranges`.
A step's turn is limited by `StepControl`'s `turn_limit`.

Convergence is scale-free, as in Gambit's tracer (path.cc:372, `c_tol` in log probabilities): a Newton correction that
moves no logit more than `logit_tolerance`, at a precision within the relative `pin_tolerance` of its pin.
Each logit also gets `settle_ulps` ulps of its own size (`logit_corrections`). A Float32 logit near 1e3 rounds by
1.2e-4, past the default tolerance, so without this allowance a Float32 trace stalls near λ 300 on 20x20 games.
A correction that moves the logits more than `contraction_limit` times the one before it (plus `logit_tolerance`
times `contraction_offset`) fails to contract; BifurcationKit's step growth reads the same rate (`corrector_quality`).
Suspected noise is confirmed when the halved step's stalled correction is at least `noise_ratio` of the suspect's
(`corrected!`). Halving keeps noise's corrections alike (ratio 1); it halves those of a predictor off the branch's
tangent (ratio 1/2) and quarters those of its curvature (1/4). The default, `1/√2`, splits 1/2 from 1.
A stall counts as noise only within `noise_limit` times the noise floor (`noise_floor`): the larger of `logit_tolerance`
and the largest settled correction so far. A corrector past it is far from any root, so the step halves at once.
`newton_iterations`: Newton iterations per corrector (BifurcationKit's default, 25; Gambit's is 100).
"""
Base.@kwdef struct CorrectorSettings{E <: Real}
    reach::Union{Nothing, E} = 2.0f0
    # Measured shares at the swap fixtures (test/games.jl): NEAR 0.075; PITCHFORK 4.0, FAR 77, FOLD 1057.
    # Half the reach leaves the corrector the other half for the branch's curvature, and splits 0.075 and 4 evenly.
    shift_share::E = 0.5f0
    max_kl::Union{Nothing, E} = nothing
    logit_tolerance::E = 1.0f-4
    # 16 ulps: Float32 solved 102 of 110 LogitNash benchmark games at maxregret 1e-4 (4 ulps: 98; none: 80). Float64 unchanged.
    settle_ulps::Int = 16
    pin_tolerance::E = 1.0f-5
    contraction_limit::Union{Nothing, E} = 0.6f0
    contraction_offset::E = 0.1f0
    # At 1/2 the 3x3x3 benchmark game of seed 5 confirmed noise at a ratio of 0.5007, at λ 3.24 in Float64 too.
    noise_ratio::E = inv(sqrt(2.0f0))
    # The noisy-gain test (opening_tests.jl) stalls within 200 tolerances. A corrector diverging from a long step on an
    # 11-logit IRVModel stay game stalled at 4,490 and was confirmed as noise; two more halvings converge there.
    noise_limit::Union{Nothing, E} = 1.0f3
    newton_iterations::Int = 25
end

"""The largest change of a logit from `from` to `to`, two points `[w; λ]` (`logits_at`), one logit at a time."""
function logit_change(from::AbstractVector{E}, to::AbstractVector{E})::E where {E <: Real}
    largest = zero(E)
    for index in firstindex(to):(lastindex(to) - 1)
        largest = max(largest, abs(last(to) * to[index] - last(from) * from[index]))
    end
    return largest
end

"""
`(correction, excess)` of a Newton step from `from` to `to`, two points `[w; λ]`: the largest change of a logit
(`logit_change`), and the largest change times `scale` past `relative` times that logit's own size, which the settle
test reads. `scale` turns a change into an estimate of the error it leaves (PathFollowing's `error_estimate`).
"""
function logit_corrections(from::AbstractVector{E}, to::AbstractVector{E}, relative::E, scale::E)::NTuple{2, E} where {E <: Real}
    largest, excess = zero(E), zero(E)
    for index in firstindex(to):(lastindex(to) - 1)
        later, earlier = last(to) * to[index], last(from) * from[index]
        change = abs(later - earlier)
        largest = max(largest, change)
        excess = max(excess, scale * change - relative * max(abs(later), abs(earlier)))
    end
    return (largest, excess)
end

"""
A trace callback that always lets the trace go on: `keep_going(logits, precision)` is `true`, and
`corrected_accepted` accepts every closing point without building its logits.
"""
struct AcceptAll end

(::AcceptAll)(::AbstractVector, ::Real)::Bool = true

"""Whether `accept_corrected` accepts the closing point `point = [w; λ]` at precision `precision`, from its logits."""
function corrected_accepted(accept_corrected::Any, point::AbstractVector, precision::Real, converged::Bool)::Bool
    return accept_corrected(logits_at(point), precision, converged)
end

corrected_accepted(::AcceptAll, ::AbstractVector, ::Real, ::Bool)::Bool = true

"""Whether `measured` is within `limit`; every measure is within an absent (`nothing`) limit."""
within_limit(::Nothing, ::Real)::Bool = true
within_limit(limit::Real, measured::Real)::Bool = measured <= limit

"""
    NoiseRecord(accuracy, suspect)

A corrector's record over its trace. `accuracy`: the largest last correction of a closing point the corrector
accepted. `suspect`: the last suspected stall at rounding noise, or `nothing`; `confirmed` means it exhausted the
precision.
"""
struct NoiseRecord{E <: Real}
    accuracy::E
    suspect::Union{
        Nothing,
        @NamedTuple{precision::E, correction::E, confirmed::Bool},
    }
end

"""
    CorrectorState(step_length, predictor_move, correction, stalls, settled_here, noise, unresolved, reach_failure,
        stalled_step)

The scalars a `CorrectorReach` carries between BifurcationKit's calls: the current step's `step_length` and its
predictor's largest logit change `predictor_move`; the current solve's last `correction` in logits, its
non-contracting corrections in a row (`stalls`) and whether it settled (`settled_here`); and the trace's `noise`
(`NoiseRecord`). `stalled_step` holds the `pin` and `arclength` of the PALC step whose iterate `stalled` holds
(`record_stall!`), NaN before any stall.
"""
struct CorrectorState{E <: Real}
    step_length::E
    predictor_move::E
    correction::E
    stalls::Int
    settled_here::Bool
    noise::NoiseRecord{E}
    unresolved::Bool
    reach_failure::@NamedTuple{precision::E, correction::E}
    stalled_step::@NamedTuple{pin::E, arclength::E}
end

"""`state` with the given fields replaced."""
function restated(
        state::CorrectorState{E};
        step_length::Real = state.step_length,
        predictor_move::Real = state.predictor_move,
        correction::Real = state.correction,
        stalls::Integer = state.stalls,
        settled_here::Bool = state.settled_here,
        noise::NoiseRecord{E} = state.noise,
        unresolved::Bool = state.unresolved,
        reach_failure::@NamedTuple{precision::E, correction::E} = state.reach_failure,
        stalled_step::@NamedTuple{pin::E, arclength::E} = state.stalled_step,
    )::CorrectorState{E} where {E <: Real}
    return CorrectorState{E}(step_length, predictor_move, correction, stalls, settled_here, noise, unresolved, reach_failure, stalled_step)
end

"""
    CorrectorReach(accept_corrected, settings, geometry, target_precision, start, predicted, step_length, ranges)

The trace's BifurcationKit `callback_newton`, which also decides the corrector's convergence: a `settled` iterate
ends Newton's method converged (`NewtonConverged`), and a correction that fails to contract stops it (`corrected!`).
Two such corrections in a row, each larger than the predictor's own move in logits and within `noise_limit` of the
noise floor (`noise_floor`), are suspected rounding noise (`noise`), confirmed at the next halved step; the check then refuses every corrector (`noise_confirmed`).
A non-finite residual stops the corrector,
and so does any iterate whose logits are further from the predictor's than `settings.reach` times the predictor's own
largest logit move (`within_reach`); the step then halves. A predictor within `RESOLUTION_ULPS` ulps of each of the
point's coordinates (`step_unresolved`) refuses that corrector and every later one (`unresolved`): no smaller step can
add a point, so the trace ends as `TraceStuck`, or as `PrecisionExhausted` when the corrector before it stopped past its
reach (`unresolved_noise`). On a smooth branch that distance is the branch's curvature
times `ds² / 2`, so halving brings any step under it, while an iterate that lands on another branch is far from it. A converged closing point is refused the same way past `reach`, and past a
player's KL `KL(corrected ‖ predicted)` of `settings.max_kl` over the players of `ranges` (`largest_kl`, `kl_ranges`). A
predictor that did not move has no reach to compare.
`accept_corrected(logits, λ, converged)` then sees every other closing point. `predicted` holds the current step's
predictor, `iterate` the current solve's last iterate and `stalled` the iterate of the last stall or reach failure on
record (`noise`, `reach_failure`, `record_stall!`), all updated in place, mutable copies of `predicted` and `start` in
`similar(predicted)`; `state` (`CorrectorState`) holds the scalars that change between calls. `geometry` measures the
step length at `target_precision`.
"""
struct CorrectorReach{C, E <: Real, V <: AbstractVector{E}, R}
    accept_corrected::C
    settings::CorrectorSettings{E}
    geometry::TraceGeometry{E}
    target_precision::E
    predicted::V
    iterate::V
    stalled::V
    state::Base.RefValue{CorrectorState{E}}  # lint: ref # lint: non-public-access  by design: callback_newton returns a verdict, and its state between calls is one immutable value here (RefValue is a Ref's concrete type)
    ranges::R
end

"""`check`'s `CorrectorState` after replacing the fields named in `changes`."""
function restate!(check::CorrectorReach{C, E}; changes...)::CorrectorState{E} where {C, E <: Real}
    check.state[] = restated(check.state[]; changes...)
    return check.state[]
end

# A predictor within this many ulps of a coordinate leaves it unchanged at the float type's resolution.
const RESOLUTION_ULPS = 4

"""
Whether the predictor `predicted` moves no coordinate of `start` by more than `RESOLUTION_ULPS · eps(E)` of its own
size: the ODE solvers' per-coordinate error test, `|Δz_i| ≤ abstol + reltol |z_i|`, with abstol zero. Coordinates
of different scales (payoff gaps near 1, precision up to 1e6) each count, and any move of a zero coordinate resolves.
"""
function step_unresolved(start::AbstractVector{E}, predicted::AbstractVector{E})::Bool where {E <: Real}
    for index in eachindex(start, predicted)
        # Negated so a NaN move counts as resolved, as the comparison is false for it.
        if !(abs(predicted[index] - start[index]) <= RESOLUTION_ULPS * eps(E) * abs(start[index]))
            return false
        end
    end
    return true
end

"""A step's reach-failure record before any corrector failed its reach: the correction is NaN, as at a solve's start."""
function no_reach_failure(::Type{E})::@NamedTuple{precision::E, correction::E} where {E <: Real}
    return (; precision = zero(E), correction = E(NaN32))
end

"""
`check`'s noise record once its step is unresolved. When the last corrector stopped past its reach, whatever its
correction (one under `logit_tolerance` also fails to settle once the predictor is below the rounding noise), halving reached the float type's resolution without closing a point: the corrections are
rounding noise, and the record confirms the precision exhausted there (`PrecisionExhausted`). `reach_failure` holds
the last reach failure of the current step, with a NaN correction when there was none; that failure ended the last
corrector, so `check.stalled` still holds its iterate.
"""
function unresolved_noise(check::CorrectorReach{C, E})::NoiseRecord{E} where {C, E <: Real}
    (; reach_failure, noise) = check.state[]
    if isnan(reach_failure.correction)
        return noise
    end
    return NoiseRecord(noise.accuracy, (; reach_failure..., confirmed = true))
end

"""
`check`'s noise floor in logits: the larger of `logit_tolerance` and the noise record's `accuracy`, the largest last
correction of a settled closing point. Rounding or evaluation noise that stalls a corrector starts near it.
"""
function noise_floor(check::CorrectorReach{C, E})::E where {C, E <: Real}
    return max(check.settings.logit_tolerance, check.state[].noise.accuracy)
end

"""
`check` with a PALC `iterate` of a stall or reach failure in `stalled`, and in `stalled_step` the `pin` and
`arclength` of its step: with the last accepted point and its tangent, the constraint its corrector held it to. Only
PALC stalls: a plain Newton iterate has no step and throws.
"""
function record_stall!(check::CorrectorReach{C, E}, iterate::NamedTuple)::CorrectorReach{C, E} where {C, E <: Real}
    if !haskey(iterate, :ds)
        throw(ArgumentError("a plain Newton iterate has no PALC step to record as a stall"))
    end
    copyto!(check.stalled, iterate.x)
    restate!(check; stalled_step = (; pin = iterate.p, arclength = iterate.ds))
    return check
end

"""Whether `check`'s corrector met rounding noise, confirmed at a halved step (`corrected!`)."""
function noise_confirmed(check::CorrectorReach)::Bool
    return !isnothing(check.state[].noise.suspect) && check.state[].noise.suspect.confirmed
end

function CorrectorReach(
        accept_corrected::C,
        settings::CorrectorSettings{E},
        geometry::TraceGeometry{E},
        target_precision::E,
        start::AbstractVector{E},
        predicted::AbstractVector{E},
        step_length::E,
        ranges::R,
    )::CorrectorReach{C, E} where {C, E <: Real, R}
    # `similar` gives mutable storage like `predicted`'s; `copy` keeps a range or static vector immutable.
    predicted_buffer, iterate_buffer = copyto!(similar(predicted), predicted), copyto!(similar(predicted), start)
    stalled_buffer = copy(iterate_buffer)
    # A plain Newton solve has no predictor move: NaN fails the noise test, so its stalls refuse.
    initial = CorrectorState{E}(
        step_length, E(NaN32), E(NaN32), 0, false, NoiseRecord(zero(E), nothing), false, no_reach_failure(E),
        (; pin = E(NaN32), arclength = E(NaN32)),
    )
    return CorrectorReach{C, E, typeof(predicted_buffer), R}(
        accept_corrected,
        settings,
        geometry,
        target_precision,
        predicted_buffer,
        iterate_buffer,
        stalled_buffer,
        Ref(initial),  # lint: ref  see `state`
        ranges,
    )
end

"""
The players' `logit_ranges` that `CorrectorReach`'s KL limit `limit` reads, over `width` logits of `pass!!`'s game:
`nothing` when the limit is off, so a pass without `logit_ranges` traces under it.
"""
kl_ranges(::Nothing, ::Any, ::Integer)::Nothing = nothing
kl_ranges(::Real, pass::Any, width::Integer)::Union{Tuple, AbstractVector} = checked_ranges(logit_ranges(pass), width)

"""Whether no player's KL from `predicted` to `corrected` over `ranges` passes `limit` (`largest_kl`); none passes an absent limit."""
kl_within(::Nothing, ::Any, ::AbstractVector, ::AbstractVector)::Bool = true
@inline function kl_within(limit::Real, ranges::Any, corrected::AbstractVector{E}, predicted::AbstractVector{E})::Bool where {E <: Real}
    return largest_kl(ranges, corrected, predicted) <= limit
end

"""Whether `check` accepts its corrector's converged closing point `corrected` (reach and KL)."""
function closing_accepted(
        check::CorrectorReach{C, E},
        corrected::AbstractVector{E},
    )::Bool where {C, E <: Real}
    (; settings) = check
    if !within_reach(check, corrected)
        @debug "corrector refused" _group = :trace reason = :reach predictor_move = check.state[].predictor_move
        return false
    end
    if !kl_within(settings.max_kl, check.ranges, corrected, check.predicted)
        @debug "corrector refused" _group = :trace reason = :kl
        return false
    end
    return true
end

"""The verdict that ends a solve: converged (`NewtonConverged`) when it `settled_here`, else `proceed`."""
function closing_verdict(check::CorrectorReach, proceed::Bool)::Union{
        Bool,
        BifurcationKit.NewtonConverged,
    }
    if check.state[].settled_here
        return BifurcationKit.NewtonConverged()
    end
    return proceed
end

function (check::CorrectorReach{C, E})(
        iterate::I;
        fromNewton::Bool = true,
        # fatou-ignore unused-argument: BifurcationKit passes keyword arguments this callback ignores
        keywords...,
    )::Union{Bool, BifurcationKit.NewtonConverged} where {C, E <: Real, I <: NamedTuple}
    if check.state[].unresolved
        return false
    end
    if fromNewton
        # Newton's closing call is its only one without `J`; an exact root converges on BifurcationKit's own test.
        if !haskey(iterate, :J)
            return closing_verdict(check, isfinite(iterate.residual))
        end
        if iterate.step == 0
            restate!(check; predictor_move = E(NaN32))
        end
        return corrected!(check, iterate, isfinite(iterate.residual))
    end
    # PALC's closing call, marked `closing`, ends every corrector.
    if !haskey(iterate, :closing)
        @debug "corrector iterate" _group = :trace step = iterate.step residual = iterate.residual precision = last(
            iterate.x,
        )
        if iterate.step == 0
            copyto!(check.predicted, iterate.x)
            restate!(
                check;
                step_length = arclength_distance(iterate.z0.u, iterate.x, check.geometry, check.target_precision),
                predictor_move = logit_change(iterate.z0.u, iterate.x),
            )
            # The unresolved test reads the previous step's reach failure, so the record resets only after it.
            if step_unresolved(iterate.z0.u, iterate.x)
                @debug "corrector stopped" _group = :trace reason = :unresolved step_length = check.state[].step_length
                restate!(check; unresolved = true, noise = unresolved_noise(check))
                return false
            end
            restate!(check; reach_failure = no_reach_failure(E))
        end
        if !isfinite(iterate.residual)
            @debug "corrector stopped" _group = :trace reason = :nonfinite
            return corrected!(check, iterate, false)
        end
        reached = within_reach(check, iterate.x)
        verdict = corrected!(check, iterate, reached)
        if !reached
            correction = check.state[].correction
            @debug "corrector stopped" _group = :trace reason = :reach correction
            record_stall!(check, iterate)
            restate!(check; reach_failure = (; precision = last(iterate.x), correction))
        end
        return verdict
    end
    # An exact root converges unsettled on BifurcationKit's own test (`residual <= tol`, tol 0), so it meets the same tests.
    converged = isfinite(iterate.residual) &&
        (check.state[].settled_here || iszero(iterate.residual))
    @debug "corrector closed" _group = :trace step = iterate.step residual = iterate.residual converged step_length = check.state[].step_length
    if converged && !closing_accepted(check, iterate.x)
        return false
    end
    if !corrected_accepted(check.accept_corrected, iterate.x, iterate.p, converged)
        return false
    end
    # A solve that settled at its start has no correction (NaN). An unsettled exact root's next correction is 0.
    closing_correction = ifelse(check.state[].settled_here, check.state[].correction, zero(E))
    if converged && !isnan(closing_correction)
        restate!(
            check;
            noise = NoiseRecord(max(check.state[].noise.accuracy, closing_correction), check.state[].noise.suspect),
        )
    end
    return closing_verdict(check, true)
end

"""
    corrected!(check, iterate, proceed)

`check`'s verdict on a Newton `iterate` that its other tests let `proceed`, in logits `z = λ w`. The first iterate
is the solve's start and never settles. Each later one is a correction `Δz` from the iterate before: it settles
(`NewtonConverged`) when `settled` by the part of `Δz` past each logit's `settle_ulps`. A correction past `contraction_limit` times the one before (plus
`logit_tolerance · contraction_offset`) fails to contract, Gambit's test (path.cc), and stops the corrector, so the
step halves. Two in a row, each larger than the predictor's own move and within `noise_limit` of the noise floor
(`noise_floor`), are suspected noise (`noise`), the iterate in `check.stalled`; the
step still halves. A second such solve before any settles, its correction at least `noise_ratio` of the suspect's,
confirms it (`noise_confirmed`): halving shrinks the corrections of a predictor past a fold, not noise. Every later
solve refuses.
"""
function corrected!(check::CorrectorReach{C, E}, iterate::NamedTuple, proceed::Bool)::Union{
        Bool,
        BifurcationKit.NewtonConverged,
    } where {C, E <: Real}
    if noise_confirmed(check)
        return false
    end
    if iterate.step == 0
        copyto!(check.iterate, iterate.x)
        restate!(check; correction = E(NaN32), stalls = 0, settled_here = false)
        return proceed
    end
    before = check.state[].correction
    correction, excess = logit_corrections(check.iterate, iterate.x, check.settings.settle_ulps * eps(E), one(E))
    copyto!(check.iterate, iterate.x)
    restate!(check; correction)
    if !proceed
        return false
    end
    if settled(check, iterate, excess)
        restate!(check; settled_here = true, noise = NoiseRecord(check.state[].noise.accuracy, nothing))
        return BifurcationKit.NewtonConverged()
    end
    (; contraction_limit, contraction_offset, logit_tolerance) = check.settings
    # The first correction has no rate (`before` is NaN), so it contracts.
    if isnan(before) ||
            within_limit(
            scaled_limit(
                contraction_limit,
                before + logit_tolerance * contraction_offset,
            ),
            correction,
        )
        restate!(check; stalls = 0)
        return true
    end
    # A stall under the predictor's move halves with the step; one far past the noise floor is no rounding noise.
    if !(check.state[].predictor_move <= correction) || !within_limit(scaled_limit(check.settings.noise_limit, noise_floor(check)), correction)
        @debug "corrector stopped" _group = :trace reason = :contraction correction before noise_floor = noise_floor(check)
        return false
    end
    if restate!(check; stalls = check.state[].stalls + 1).stalls < 2
        return true
    end
    suspect = check.state[].noise.suspect
    confirmed = !isnothing(suspect) && correction >= check.settings.noise_ratio * suspect.correction
    @debug "corrector stopped" _group = :trace reason = :noise confirmed correction predictor_move = check.state[].predictor_move
    record_stall!(check, iterate)
    restate!(
        check;
        noise = NoiseRecord(check.state[].noise.accuracy, (; precision = last(iterate.x), correction, confirmed)),
    )
    return false
end

"""The pin row `λ - pin` of a corrector `iterate`'s residual: the last entry of Newton's `fx` or of PALC's `res_f`."""
function pin_residual(iterate::NamedTuple)::eltype(iterate.x)
    if haskey(iterate, :fx)
        return last(iterate.fx)
    end
    return last(iterate.res_f)
end

"""
Whether `check`'s corrector has settled `iterate`, reached by a Newton step whose largest logit move past that
logit's `settle_ulps` is `correction` (`logit_corrections`): the step moved no logit more than `logit_tolerance` past
its rounding, and the precision `λ` is at its pin to the relative `pin_tolerance`. The pin
needs its own test: a far target's logits barely move with `λ`, so they would pass a `λ` that never left its start.
"""
function settled(check::CorrectorReach, iterate::I, correction::Real)::Bool where {
        I <:
        NamedTuple,
    }
    precision = last(iterate.x)
    offset = pin_residual(iterate)
    pin_scale = max(abs(precision), abs(precision - offset))
    return correction <= check.settings.logit_tolerance &&
        abs(offset) <= check.settings.pin_tolerance * pin_scale
end

"""
Whether `point` moved no logit further from the predictor's than `check.settings.reach` times the predictor's own
largest logit move (`logit_change`). Logits are target-free, as the convergence test is; a weighted norm can shrink a
branch jump (a frozen Fisher metric) or swell a settled point (the norm's `P²` on the gaps at a far target).
"""
function within_reach(check::CorrectorReach, point::AbstractVector)::Bool
    if !(check.state[].predictor_move > 0)
        return true
    end
    return within_limit(
        scaled_limit(check.settings.reach, check.state[].predictor_move),
        logit_change(check.predicted, point),
    )
end

"""
    TraceProgress(gaps, precision, steps, precision_min, precision_max, gap_tangent, tangent_precision, unmoved)

The last accepted point `(gaps, precision)` of a trace, its step count, the range of the precision
over accepted points and the tangent there (both in payoff gaps), read from BifurcationKit's branch (`branch_progress`).
`unmoved` marks a trace that stopped on an accepted point bitwise equal to the one before it (`TraceFinalise`).
"""
struct TraceProgress{E <: Real}
    gaps::Vector{E}
    precision::E
    steps::Int
    precision_min::E
    precision_max::E
    gap_tangent::Vector{E}
    tangent_precision::E
    unmoved::Bool
end

"""
BifurcationKit's branch record of the trace's `point = [w; λ]` and the state's tangent: the gaps `w`, their tangent
and the tangent's precision part. `branch_progress` reads the trace's progress back from these records.
"""
function trace_record_point(
        point::AbstractVector{E}, ::Real;
        state::Any,
        # fatou-ignore unused-argument: BifurcationKit passes keyword arguments this record ignores
        keywords...,
    )::NamedTuple where {E <: Real}
    return (;
        gaps = point[begin:(end - 1)], gap_tangent = state.τ.u[
            begin:(end - 1),
        ], tangent_precision = state.τ.p,
    )
end

"""A branch record's precision."""
function record_precision(record::R)::fieldtype(R, :param) where {R <: NamedTuple}
    return record.param
end

"""Whether two branch records hold the same point `(w, λ)`, bitwise: a step below the point's rounding."""
same_point(record::NamedTuple, other::NamedTuple)::Bool = record.param == other.param &&
    record.gaps == other.gaps

"""
    branch_progress(branch, opening)

The `TraceProgress` after a segment whose BifurcationKit `branch` of `trace_record_point` records starts at the point of
`opening`, the progress before the segment: its last record, its steps counted on from `opening.steps`, the precision
range of both, and `unmoved` when its last two records hold the same point. A branch of its start alone accepted no
point and leaves `opening` as it was, tangent included.
"""
function branch_progress(branch::AbstractVector, opening::TraceProgress{E})::TraceProgress{
        E,
    } where {E <: Real}
    if length(branch) == 1
        return opening
    end
    record = last(branch)
    unmoved = same_point(record, branch[end - 1])
    return TraceProgress{E}(
        copy(record.gaps),
        record.param,
        opening.steps + record.step,
        min(opening.precision_min, minimum(record_precision, branch)),
        max(opening.precision_max, maximum(record_precision, branch)),
        copy(record.gap_tangent),
        record.tangent_precision,
        unmoved,
    )
end

"""
    TraceFinalise(keep_going, redraw, slot, settings)

BifurcationKit's `finalise_solution` of one trace segment. An accepted point bitwise equal to the branch's last one (a
step below the point's rounding) stops the trace, a stall. The trace goes on while `keep_going(logits,
precision)` holds. Then `redraw(logits, precision, shift)` may return a new game, where `shift` is
the point's `RootShift` check under `settings`: `slot`, a `Ref` to the current `RedrawSlot` (the constructor
wraps the one it is given), takes it and the
segment stops, for `trace_endpoint` to correct the point onto it and restart. A `redraw` that returns `nothing` or the
current game swaps nothing. `JULIA_DEBUG=trace` lists every point.
"""
struct TraceFinalise{A, D, S, C <: CorrectorSettings}
    keep_going::A
    redraw::D
    slot::Base.RefValue{S}  # lint: ref # lint: non-public-access  by design: finalise_solution returns a Bool, and a redraw's swap reaches the trace as one immutable value here (RefValue is a Ref's concrete type)
    settings::C
end

function TraceFinalise(keep_going::A, redraw::D, slot::S, settings::C)::TraceFinalise{A, D, S, C} where {A, D, S, C <: CorrectorSettings}
    return TraceFinalise{A, D, S, C}(keep_going, redraw, Ref(slot), settings)  # lint: ref  see `slot`
end

function (finalise::TraceFinalise)(
        point::P, tangent::T, step::S, result::Any;
        state::Any,
        iter::Any,
        # fatou-ignore unused-argument: BifurcationKit passes keyword arguments this callback ignores
        keywords...,
    )::Bool where {P, T, S <: Integer}
    @debug "trace point" _group = :trace step precision = point.p state = copy(
        point.u,
    ) tangent = copy(tangent.u)
    # BifurcationKit saves the point after this call, so the branch's last record is the point before it.
    if same_point(
            (;
                param = point.p, gaps = view(
                    point.u,
                    firstindex(point.u):(lastindex(point.u) - 1),
                ),
            ),
            last(result.branch),
        )
        return false
    end
    logits = logits_at(point.u)
    if !finalise.keep_going(logits, point.p)
        return false
    end
    slot = restated(finalise.slot[]; step_length = abs(state.ds))
    finalise.slot[] = slot
    # A redraw swaps the map for the steps after this point, and the endpoint has none.
    if point.p == finalise.slot[].game.target_precision
        return true
    end
    # The slot is pending only after a swap here: a restart clears it before the segment runs.
    finalise.slot[] = redrawn_slot(finalise.redraw, slot, logits, point, tangent, state.ds, iter.alg, finalise.settings)
    return !finalise.slot[].pending
end
