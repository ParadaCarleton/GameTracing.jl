using BifurcationKit: BifurcationKit
using ErrorTypes: Err, Ok
using LinearAlgebra: LU, issuccess, lu!
"""
    EngineSettings{E}

The settings of a trace engine in element type `E`: `BifurcationKitSettings` or `PathFollowingSettings`.
`logit_solve` and its kin take either as `settings`.
"""
abstract type EngineSettings{E <: Real} end

"""The element type `E` of `settings`, an `EngineSettings{E}`."""
function element_type(::EngineSettings{E})::Type{E} where {E <: Real}
    return E
end

"""
    StepControl{E}(; distance_limit, max_step_growth, turn_limit)

BifurcationKit's `CorrectorQuality` thresholds, which shrink a step whose corrector looks poor. `distance_limit` is the
Euclidean length of a Newton step's logit part (Gambit's 0.4 in log probabilities); `max_step_growth` bounds a step's
growth (Gambit's `max_accel` 1.1), and the step then shrinks as the distance or contraction nears its limit.
`turn_limit` refuses a step whose tangent turns this many radians from the last one, Allgower and Georg's
test (6.1.10) for a step across a fold onto another branch; 1 is under Gambit's quarter turn (`t·newT < 0`, path.cc).
Without `BifurcationKitSettings`' `orientation_check` each tangent keeps the last one's direction, so a turn `α` reads
`min(α, π - α)`: the test catches turns near a right angle and misses a reversal.
The corrector's contraction rate (`CorrectorSettings`) also slows growth. `nothing` switches a threshold off.
"""
Base.@kwdef struct StepControl{E <: Real}
    distance_limit::Union{Nothing, E} = 0.4f0
    max_step_growth::Union{Nothing, E} = GAMBIT_MAX_ACCEL
    turn_limit::Union{Nothing, E} = 1.0f0
end

"""
    BifurcationKitSettings{E}(; max_step, aggressiveness, step_metric,
                     corrector, max_steps, tangent, bordered_solver, step_control, orientation_check, palc_corrector,
                     palc_options, continuation_options)

BifurcationKit PALC controls for `trace_endpoint`, in `(w, λ)` with `w = logits / λ` the payoff gaps and `λ` the
precision. Step sizes are arclength in `step_metric`'s norm (`StepDistance`). The defaults are
`gambit_settings(E, GAMBIT_MAX_ACCEL)`: Gambit's growth limit. The first step comes from the game (`opening_step`).

- `max_step`: the largest arclength step, which also caps the first. A step whose corrector fails halves; the trace ends as
  `TraceStuck` once its predictor moves the point no more than a few ulps of its size (`step_unresolved`).
  BifurcationKit's `dsmin` is zero: its halving stops at a zero step. After the predictor stops moving, BifurcationKit
  still halves to zero through the subnormals, about 150 halvings of one refused residual evaluation each.
- `aggressiveness` (BifurcationKit's `a`): without a `step_control` growth limit, a step grows by
  `1 + a ((N - n) / N)²` after a corrector of `n` of its `N = corrector.newton_iterations` iterations. The default
  is `GAMBIT_MAX_ACCEL - 1`, the growth limit's largest growth.
- `corrector`: a `CorrectorSettings`.
- `step_metric`: a `TraceGeometry`, the norm a trace steps in and its state weight. The default is
  `default_step_metric`, both engines' default. The first tangent is the exact tangent at the flat start (`opening_tangent`).
- `max_steps`: continuation steps before the trace fails as `NoConvergence`.
- `tangent`: BifurcationKit's tangent algorithm, passed to PALC as is. `Bordered()` (the default) solves the bordered
  Jacobian system. `Secant()` extrapolates the last two points and saves that solve. Do not use it where a small payoff
  perturbation unfolds a pitchfork: there it crosses onto another arm (the test "a pitchfork unfolded by a payoff
  perturbation"). Over 40 random unfoldings at payoff scale 1e-4 it kept the branch in none, `Bordered()` in all 40.
- `bordered_solver`: a `BorderedSolver`.
- `step_control`: a `StepControl`, or `nothing` for none.
- `orientation_check` orients a bordered tangent by the determinant of the bordered matrix. Only `MatrixSolve` has a
  determinant, so `BorderingSolve` needs `orientation_check = false`; the constructor throws otherwise.
- `palc_corrector`: PALC's `corrector` keyword, `nothing` by default, which is BifurcationKit's own Newton method
  (`newton_palc`; it stops on leaving `[0, P]`, `P` the target precision). `BifurcationKit.FreeParameterNewton()` leaves `λ` free there instead,
  the trace's choice before this field. `CorrectorReach` reads any corrector's Newton callbacks.
- `palc_options`: a `NamedTuple` of further keywords for `BifurcationKit.PALC`. A keyword the trace sets itself (`tangent`,
  `θ`, `dotθ`, `bls`, `step_control`, `orientation_check`, `corrector`, `orientation_dot`) throws an `ArgumentError` at the
  trace: the step shrinking, tangent orientation and geometry rest on them. Set them through the fields above.
- `continuation_options`: a `NamedTuple` of further keywords for `BifurcationKit.ContinuationPar`. A keyword the trace sets
  itself (`continuation_options` lists them) throws the same way: `CorrectorReach` needs `tol = 0`, and the trace ends at `p_max`.
"""
struct BifurcationKitSettings{
        E <: Real,
        T <: BifurcationKit.AbstractTangentComputation,
        C,
        P <: NamedTuple,
        O <: NamedTuple,
    } <: EngineSettings{E}
    max_step::E
    aggressiveness::E
    step_metric::TraceGeometry{E}
    corrector::CorrectorSettings{E}
    max_steps::Int
    tangent::T
    bordered_solver::BorderedSolver
    step_control::Union{Nothing, StepControl{E}}
    orientation_check::Bool
    palc_corrector::C
    palc_options::P
    continuation_options::O
end

# Each keyword is named, not splatted: a splatted `keywords...` leaves the field conversions to runtime dispatch.
function BifurcationKitSettings{E}(;
        max_step::Real = 1000.0f0,
        aggressiveness::Real = E(GAMBIT_MAX_ACCEL) - one(E),
        step_metric::TraceGeometry{E} = default_step_metric(E),
        corrector::CorrectorSettings{E} = CorrectorSettings{E}(),
        max_steps::Integer = 2_000,
        tangent::T = BifurcationKit.Bordered(),
        bordered_solver::BorderedSolver = MatrixSolve,
        step_control::Union{Nothing, StepControl{E}} = StepControl{E}(),
        orientation_check::Bool = true,
        palc_corrector::C = nothing,
        palc_options::P = (;),
        continuation_options::O = (;),
    )::BifurcationKitSettings{E, T, C, P, O} where {
        E <: Real, T <: BifurcationKit.AbstractTangentComputation, C, P <: NamedTuple, O <: NamedTuple,
    }
    return BifurcationKitSettings{E, T, C, P, O}(
        E(max_step), E(aggressiveness), step_metric, corrector, Int(max_steps), tangent, bordered_solver,
        step_control, checked_orientation(orientation_check, bordered_solver), palc_corrector, palc_options,
        continuation_options,
    )
end

"""`orientation_check`, or a throw where it asks for a determinant that `bordered_solver` does not compute (`BorderingSolve`)."""
function checked_orientation(orientation_check::Bool, bordered_solver::BorderedSolver)::Bool
    orientation_check && bordered_solver == BorderingSolve &&
        throw(ArgumentError("BorderingSolve has no bordered determinant for orientation_check; pass orientation_check = false"))
    return orientation_check
end

"""`limit * scale`, or `nothing` where the threshold is off (`limit` is `nothing`)."""
scaled_limit(::Nothing, ::Real)::Nothing = nothing
scaled_limit(limit::Real, scale::Real)::typeof(limit * scale) = limit * scale

"""
PALC's `step_control` for `control` over `width` gaps in `geometry`: `nothing` when it is off, else BifurcationKit's
`CorrectorQuality`, its distance limit scaled from the gaps' Euclidean length into the norm. Its contraction rate,
`corrector`'s limit and offset, slows growth only: `CorrectorReach` refuses on the rate in Gambit's norm.
"""
corrector_quality(
    ::Nothing,
    ::CorrectorSettings,
    ::TraceGeometry,
    ::Integer,
)::Nothing = nothing

function corrector_quality(
        control::StepControl{E},
        corrector::CorrectorSettings{E},
        geometry::TraceGeometry{E},
        width::Integer,
    )::BifurcationKit.CorrectorQuality{E} where {E <: Real}
    return BifurcationKit.CorrectorQuality{E}(
        scaled_limit(control.distance_limit, sqrt(coordinate_weight(geometry, width))),
        scaled_limit(corrector.contraction_limit, one(E)),
        scaled_limit(control.max_step_growth, one(E)),
        corrector.contraction_offset,
        scaled_limit(control.turn_limit, one(E)),
        false,
    )
end

"""
PALC's `orientation_dot`: the plain product `trace_dot` of two branch vectors `[w; λ]` in `geometry` at
`target_precision`; `λ` is in the vectors, so the continuation parameters add nothing. Without the bordered
determinant a new tangent takes the sign that keeps this product with the old one positive; the trace's Fisher
metric, frozen at saturated logits, would let `λ` alone decide that sign past a fold and turn the trace back.
"""
struct PlainOrientation{E <: Real}
    geometry::TraceGeometry{E}
    target_precision::E
end

function (orientation::PlainOrientation{E})(
        left::AbstractVector{E},
        right::AbstractVector{E},
        # PALC passes the continuation parameters and its state weight θ too.
        ::Real,
        ::Real,
        ::Real,
    )::E where {E <: Real}
    return trace_dot(left, right, orientation.geometry, orientation.target_precision)
end

"""
BifurcationKit's PALC for `settings` over a trace of `width` logits of `game` to its target precision, its tangent
oriented by the bordered determinant where `orientation_check` holds (`settings.orientation_check` by default). A
Fisher distance reads the game's `logit_ranges` (`palc_geometry`).
"""
function palc_algorithm(
        settings::BifurcationKitSettings{E},
        game::LogitGame,
        width::Integer;
        orientation_check::Bool = settings.orientation_check,
    )::BifurcationKit.AbstractContinuationAlgorithm where {E <: Real}
    return palc_with_geometry(
        settings,
        width,
        palc_geometry(settings.step_metric, game.pass!!, width, game.target_precision),
        orientation_check,
        PlainOrientation(settings.step_metric, E(game.target_precision)),
    )
end

"""
BifurcationKit's PALC for `settings` over `width` logits in `geometry` (`palc_geometry`), whose type varies by
distance, with the bordered determinant's `orientation_check` (`MatrixSolve` only; it throws otherwise) and `orientation`.
"""
function palc_with_geometry(
        settings::BifurcationKitSettings{E},
        width::Integer,
        geometry::NamedTuple,
        orientation_check::Bool,
        orientation::PlainOrientation{E},
    )::BifurcationKit.AbstractContinuationAlgorithm where {E <: Real}
    # No named local: the tuple's Union-typed solver fields would make it an abstract local.
    return BifurcationKit.PALC(;
        added_options(
            (;
                tangent = settings.tangent,
                θ = geometry.palc_state_weight,
                dotθ = geometry.dot,
                bls = bordered_linear_solver(settings.bordered_solver, width + 1, E),
                step_control = corrector_quality(
                    settings.step_control,
                    settings.corrector,
                    settings.step_metric,
                    width,
                ),
                orientation_check = checked_orientation(orientation_check, settings.bordered_solver),
                corrector = settings.palc_corrector,
                orientation_dot = orientation,
            ), settings.palc_options
        )...
    )
end


"""
BifurcationKit's continuation parameters for `settings`, from precision zero to `target_precision`, with first step
`step_length` and at most `max_steps` steps. Eigenvalues, fold and bifurcation detection serve bifurcation diagrams; the trace reads
only its endpoint.
"""
function continuation_options(
        settings::BifurcationKitSettings{E},
        step_length::E,
        max_steps::Integer,
        target_precision::Real,
    )::BifurcationKit.ContinuationPar where {E <: Real}
    own = (;
        ds = step_length,
        dsmin = zero(E),
        dsmax = settings.max_step,
        a = settings.aggressiveness,
        p_min = zero(E),
        p_max = E(target_precision),
        max_steps,
        newton_options = BifurcationKit.NewtonPar(;
            # Only `CorrectorReach` converges a corrector (`settled`), and an exact root.
            tol = zero(E),
            max_iterations = settings.corrector.newton_iterations,
            α = one(E),
            αmin = one(E) / 1000,
        ),
        save_sol_every_step = 0,
        detect_fold = false,
        detect_bifurcation = 0,
        tol_stability = zero(E),
        dsmin_bisection = zero(E),
        tol_bisection_eigenvalue = zero(E),
        tol_param_bisection_event = zero(E),
    )
    return BifurcationKit.ContinuationPar(; added_options(own, settings.continuation_options)...)
end

"""`merge(own, options)`, or an `ArgumentError` where `options` names a keyword of `own`, which the trace sets itself."""
function added_options(own::NamedTuple, options::NamedTuple)::NamedTuple
    overridden = intersect(keys(own), keys(options))
    if !isempty(overridden)
        throw(ArgumentError("the trace sets $(join(overridden, ", ")) itself; set it through a BifurcationKitSettings field"))
    end
    return merge(own, options)
end

"""
    trace_iterator(game, pass_work, start, settings, finalise, corrector, step_length, max_steps)

BifurcationKit's PALC iterator over `game`'s branch from `start = [w; λ]` toward `game.target_precision`, on the
game's pass work `pass_work`, with `finalise` at each accepted point, `corrector` (`CorrectorReach`) at each Newton
iterate and a branch of `trace_record_point` records. BifurcationKit's constructors are not inferable, so each is
built in its own function and passed on without a named local; it runs once per segment.
"""
function trace_iterator(
        game::LogitGame,
        pass_work::Any,
        start::AbstractVector{E},
        settings::BifurcationKitSettings{E},
        finalise::TraceFinalise,
        corrector::CorrectorReach,
        step_length::E,
        max_steps::Integer,
    )::BifurcationKit.ContIterable where {E <: Real}
    return continuation_iterator(
        branch_problem(game, start, last(start), pass_work; record_from_solution = trace_record_point),
        palc_algorithm(settings, game, length(start) - 1),
        continuation_options(
            settings,
            step_length,
            max_steps,
            game.target_precision,
        ),
        finalise,
        corrector,
    )
end

function continuation_iterator(
        problem::BifurcationKit.AbstractBifurcationProblem,
        algorithm::BifurcationKit.AbstractContinuationAlgorithm,
        options::BifurcationKit.ContinuationPar,
        finalise::TraceFinalise,
        corrector::CorrectorReach,
    )::BifurcationKit.ContIterable
    return BifurcationKit.ContIterable(
        problem,
        algorithm,
        options;
        # The default name formats the clock at every segment; the trace saves no file.
        filename = "branch",
        normC = BifurcationKit.norminf,
        verbosity = 0,
        finalise_solution = finalise,
        callback_newton = corrector,
    )
end

"""
    opened_state(iterator, start, opening)

BifurcationKit's first continuation state of `iterator` from `start`: its two-point start toward the point
`opening`, whose secant is the opening or restart tangent.
"""
function opened_state(
        iterator::BifurcationKit.ContIterable,
        start::AbstractVector,
        opening::AbstractVector,
    )::BifurcationKit.ContState
    return first(
        BifurcationKit.iterate_from_two_points(
            iterator,
            start,
            last(start),
            opening,
            last(opening),
        )
    )
end

"""
    restart_opening(iterator, start, factorization, old_tangent)

`(; opening, gap_tangent, tangent_precision)` of a restart of `iterator` after a redraw: the point `opening` `ds`
along `restart_tangent` from the swapped game's `factorization`, oriented by the trace's `old_tangent`, and that
tangent of unit length in the trace's norm, split into its gap and precision parts.
"""
function restart_opening(
        iterator::BifurcationKit.ContIterable,
        start::AbstractVector{E},
        factorization::LU{E},
        old_tangent::AbstractVector{E},
    )::NamedTuple where {E <: Real}
    algorithm = iterator.alg
    tangent = restart_tangent(factorization, old_tangent, E(algorithm.θ), algorithm.orientation_dot)
    return opening_along(iterator, start, tangent)
end

"""`(; opening, gap_tangent, tangent_precision)` of `iterator` from `start` along `tangent`, scaled to unit length in the trace's norm."""
function opening_along(
        iterator::BifurcationKit.ContIterable,
        start::AbstractVector{E},
        tangent::AbstractVector{E},
    )::NamedTuple where {E <: Real}
    algorithm = iterator.alg
    dot, palc_state_weight = algorithm.dotθ, E(algorithm.θ)
    # A Fisher trace's metric is frozen at the point it restarts from.
    BifurcationKit.reweight!(dot, BifurcationKit.BorderedArray(start, last(start)))
    length_in_norm = sqrt(palc_product(dot, palc_state_weight, tangent, tangent))
    return (;
        opening = start .+ (iterator.contparams.ds / length_in_norm) .* tangent,
        gap_tangent = tangent[begin:(end - 1)] ./ length_in_norm,
        tangent_precision = last(tangent) / length_in_norm,
    )
end

"""
    OpeningTangent(tangent)

The tangent of a branch's opening at the flat start, `[-∂F/∂λ; 1]` in `(w, λ)`, as `opening_tangent` returns it.
"""
struct OpeningTangent{E <: Real}
    tangent::Vector{E}
end

"""
    opening_tangent(game, start, pass_work)

The branch's `OpeningTangent` at the flat `start = [w; 0]`. At precision zero play is uniform whatever the gaps, so
`∂F/∂w` is the identity and the tangent is `(-∂F/∂λ, 1)`, read from one evaluation of the game's Jacobian on its pass
work `pass_work`. A secant start instead turns by radians at a near crossing, where rounding in its second solve is
large against the branch's bend.
"""
function opening_tangent(game::LogitGame, start::AbstractVector{E}, pass_work::Any)::OpeningTangent{E} where {E <: Real}
    width = length(start) - 1
    system = branch_system!(BranchWorkspace(start, pass_work), game, start, last(start))
    return OpeningTangent([-system.jacobian[1:width, end]; one(E)])
end

"""
    trace_endpoint(game, width, settings; keep_going, accept_corrected, redraw, context)

Follow `game`'s branch of `width` logits from the flat start (zero logits, the only root at precision zero) to
`game.target_precision` on the engine of `settings` (`engine_endpoint`). Return `Ok(Traced)` logits there or
`Err(TraceFailure)` for the failure that stopped short (a `TraceResult`).

`keep_going(logits, λ)` runs at each accepted point before the next step's corrector, and `false` stops the trace.
`accept_corrected(logits, λ, converged)` runs as each corrector ends, and `false` refuses a converged point, which
shrinks the step. `redraw(logits, λ, shift)` swaps the game mid-trace; only `BifurcationKitSettings` runs it
(`check_redraw`). `context`, a `NamedTuple`, rides on every failure as its own field (`TraceFailure`).

A non-finite start fails `NotFinite` (`AtStart`), and a zero target returns the flat start with no step. A trace that
stops short fails `NoConvergence` (`MaxSteps`) when it spends `settings.max_steps`, and `TraceStuck`, `Singular` or
`PrecisionExhausted` otherwise (`TraceCause`). The point at the target passes one more Newton correction there,
settled in Gambit's norm, or fails.
"""
function trace_endpoint(
        game::LogitGame,
        width::Integer,
        settings::EngineSettings{E};
        keep_going::Any = AcceptAll(),
        accept_corrected::Any = AcceptAll(),
        redraw::Any = NoRedraw(),
        context::C = (;),
    )::TraceResult{E, C} where {E <: Real, C <: NamedTuple}
    check_redraw(settings, redraw)
    flat = zeros(E, width)
    # One set of pass work arrays serves every evaluation of the game; a redraw's swap brings the new game's.
    pass_work = pass_workspace(game.pass!!, flat)
    start = trace_start(game, flat, pass_work)
    # The start is a root for every finite game; a non-finite gain shows in its gaps.
    if !all(isfinite, start)
        stop = TraceStop(flat, collect(payoff_gaps(start)), zero(E), 0, zero(E))
        return Err(TraceFailure(TraceCause.NotFinite{E}(AtStart), stop, EngineRecord.NoRecord{E}(), context))
    end
    # The flat start is the only root at precision zero, so a zero target takes no step. Checked in `E`, where a tiny
    # target rounds to 0.
    if iszero(E(game.target_precision))
        return Ok(Traced(flat, 0, zero(E)))
    end
    return engine_endpoint(game, start, pass_work, settings, (; keep_going, accept_corrected, redraw), context)
end

"""Throw an `ArgumentError` unless the engine of `settings` runs `redraw`: only BifurcationKit redraws a game mid-trace."""
check_redraw(::BifurcationKitSettings, ::Any)::Nothing = nothing

"""
    engine_endpoint(game, start, pass_work, settings::BifurcationKitSettings, callbacks, context)

`trace_endpoint` on BifurcationKit from the finite `start = [w; 0]` (`trace_start`) on `game`'s pass work `pass_work`,
to a nonzero target, with `callbacks = (; keep_going, accept_corrected, redraw)`. A refusal by `accept_corrected`
halves the step. A stop short of the target fails `TraceStuck` for a step below the float type's resolution, an
unchanged accepted point or a refusal, and `Singular` for a broken linear solve. The endpoint passes `verified_endpoint`.

`redraw(logits, λ, shift)` runs at each accepted point short of the target precision that `keep_going` passes, and returns
`nothing`, a new game of the same type and target precision, or the `ShiftedGame` that `shift(game)` (`RootShift`)
returned for one. That check tells whether a candidate game's root lies within the corrector's reach, so a caller can
refine a draw that does not before it returns; the restart reuses its factorization of the new game's Jacobian. A new game replaces the map for the steps after that point: the
trace restarts there along the new map's tangent, oriented by the old one, with the step it held, and its next
corrector lands on the new map under the same step control. The endpoint, with no step after it, is never redrawn.
"""
function engine_endpoint(
        game::LogitGame,
        start::AbstractVector{E},
        pass_work::Any,
        settings::BifurcationKitSettings{E},
        callbacks::NamedTuple,
        context::C,
    )::TraceResult{E, C} where {E <: Real, C <: NamedTuple}
    (; keep_going, accept_corrected, redraw) = callbacks
    gaps = payoff_gaps(start)[:]
    opening = TraceProgress(gaps, zero(E), 0, zero(E), zero(E), zero(gaps), zero(E), false)
    finalise = TraceFinalise(
        keep_going,
        redraw,
        RedrawSlot(game, false, opening_step(game, settings.step_metric, settings.max_step), pass_work),
        settings.corrector,
    )
    return redrawn_trace(
        guarded_trace(start, opening_tangent(game, start, pass_work), settings, finalise, accept_corrected, opening),
        finalise,
        settings,
        accept_corrected,
        context,
    )
end

"""
    redrawn_trace(outcome, finalise, settings, accept_corrected, context)

The `TraceResult` of a trace segment that ended with `outcome`. When `redraw` swapped the game there, the trace
restarts on the new game from the last accepted point (`outcome.progress`), a root of the old game, moved by the swap's
root shift onto the new game's root, as a new segment counted on from its steps.
"""
function redrawn_trace(
        outcome::NamedTuple,
        finalise::TraceFinalise,
        settings::BifurcationKitSettings{E},
        accept_corrected::Any,
        context::C,
    )::TraceResult{E, C} where {E <: Real, C <: NamedTuple}
    (; progress) = outcome
    slot = finalise.slot[]
    if !isnothing(outcome.exception) || !slot.pending
        return reached_state(progress, outcome, slot.game, slot.pass_work, settings, context)
    end
    finalise.slot[] = restated(slot; pending = false)
    if progress.steps >= settings.max_steps
        return reached_state(progress, outcome, slot.game, slot.pass_work, settings, context)
    end
    return redrawn_trace(
        with_earlier_accuracy(
            guarded_trace(
                [progress.gaps; progress.precision] .+ slot.shift,
                [progress.gap_tangent; progress.tangent_precision],
                settings,
                finalise,
                accept_corrected,
                progress,
            ),
            outcome.logit_accuracy,
        ),
        finalise,
        settings,
        accept_corrected,
        context,
    )
end

"""A segment's `outcome` whose `logit_accuracy` also spans the earlier segments' `earlier` accuracy."""
function with_earlier_accuracy(outcome::NamedTuple, earlier::Real)::NamedTuple
    return (; outcome..., logit_accuracy = max(outcome.logit_accuracy, earlier))
end

"""
    guarded_trace(start, old_tangent, settings, finalise, accept_corrected, opening)

`(; progress, step_length, accepted_step, exhausted, logit_accuracy, unresolved, exception)` of a PALC segment on `finalise.slot`'s
game from `start = [w; λ]`, the point of `opening` (the `TraceProgress` before it), for the steps `settings.max_steps`
leaves: the progress after it (`branch_progress`), the step length its last corrector tried, the step of the last
accepted point (the slot's, carried from an earlier segment when this one accepted none), the corrector's confirmed stall
at rounding noise (`exhausted_record`, or `nothing`), the largest last correction of an accepted point, whether its predictor fell below
the float type's resolution (`step_unresolved`), and `nothing` for a segment that
ran, else the third-party failure it ended on. The segment opens along `restart_tangent`, oriented by `old_tangent`
(`opening_tangent` from the flat start).
BifurcationKit throws a `String` when Newton misses its first point and an `ErrorException` at its second; a
non-finite Jacobian reaches LAPACK, and so does a singular restart tangent. Any other exception propagates.
"""
function guarded_trace(
        start::AbstractVector{E},
        old_tangent::Union{AbstractVector{E}, OpeningTangent{E}},
        settings::BifurcationKitSettings{E},
        finalise::TraceFinalise,
        accept_corrected::Any,
        opening::TraceProgress{E},
    )::NamedTuple where {E <: Real}
    corrector = CorrectorReach(
        accept_corrected,
        settings.corrector,
        settings.step_metric,
        E(finalise.slot[].game.target_precision),
        similar(start),
        similar(start),
        zero(E),
        checked_ranges(logit_ranges(finalise.slot[].game.pass!!), length(start) - 1),
    )
    return segment_outcome(
        segment_branch(
            trace_iterator(
                finalise.slot[].game,
                finalise.slot[].pass_work,
                start,
                settings,
                finalise,
                corrector,
                finalise.slot[].step_length,
                settings.max_steps - opening.steps,
            ),
            start,
            finalise.slot[].factorization,
            old_tangent,
            opening,
        ),
        corrector,
        finalise,
    )
end

"""
`guarded_trace`'s outcome of a segment that `ran` (`segment_branch`) under `corrector`, with the step of the last
accepted point, which `finalise` sets in its slot at each one.
"""
function segment_outcome(
        ran::NamedTuple,
        corrector::CorrectorReach{C, E},
        finalise::TraceFinalise,
    )::NamedTuple where {C, E <: Real}
    return (;
        ran.progress,
        step_length = ifelse(isnothing(ran.exception), corrector.state[].step_length, zero(E)),
        accepted_step = E(finalise.slot[].step_length),
        exhausted = isnothing(ran.exception) && noise_confirmed(corrector) ? exhausted_record(corrector) : nothing,
        logit_accuracy = corrector.state[].noise.accuracy,
        unresolved = isnothing(ran.exception) && corrector.state[].unresolved,
        ran.exception,
    )
end

"""
The confirmed stall at rounding noise of `corrector` (`noise_confirmed`), a `StalledCorrection`: the noise record's
`precision` and `correction`, its stalled Newton iterate `[w; λ]`, and its step's `pin` and `arclength`
(`record_stall!`). A corrector without a confirmed stall throws.
"""
function exhausted_record(corrector::CorrectorReach{C, E})::StalledCorrection{E} where {C, E <: Real}
    if !noise_confirmed(corrector)
        throw(ArgumentError("the corrector has no confirmed stall at rounding noise"))
    end
    (; precision, correction) = corrector.state[].noise.suspect
    (; pin, arclength) = corrector.state[].stalled_step
    return StalledCorrection(precision, correction, collect(corrector.stalled), pin, arclength)
end

"""How a guarded BifurcationKit call ended other than with its result: `Thrown` or `Ended`."""
abstract type GuardedEnd end

"""An exception BifurcationKit threw that `expected_failure` names, held as a value."""
struct Thrown{X} <: GuardedEnd
    exception::X
end

"""BifurcationKit's `continuation!` ran to its end."""
struct Ended <: GuardedEnd end

"""
    segment_branch(iterator, factorization, old_tangent, opening)

`(; progress, exception)` of BifurcationKit's continuation of `iterator` from the point of `opening`: the progress
after its branch (`branch_progress`), and the exception it threw or `nothing`. A segment that throws before its
branch exists, or accepts no point, stays at `opening` with the restart tangent once that is computed. Only an
exception `expected_failure` names is caught (`Thrown`); any other propagates. Each step is its own method, as
BifurcationKit's results are not inferable.
"""
function segment_branch(
        iterator::BifurcationKit.ContIterable,
        start::AbstractVector{E},
        factorization::Union{Nothing, LU{E}},
        old_tangent::Union{AbstractVector{E}, OpeningTangent{E}},
        opening::TraceProgress{E},
    )::NamedTuple where {E <: Real}
    return restarted_segment(
        iterator,
        start,
        guarded_restart(iterator, start, factorization, old_tangent, opening),
        opening,
    )
end

restarted_segment(
    ::BifurcationKit.ContIterable,
    ::AbstractVector,
    thrown::Thrown,
    opening::TraceProgress,
)::NamedTuple = (; progress = opening, thrown.exception)

function restarted_segment(
        iterator::BifurcationKit.ContIterable,
        start::AbstractVector{E},
        restart::NamedTuple,
        opening::TraceProgress{E},
    )::NamedTuple where {E <: Real}
    opened = TraceProgress{E}(
        opening.gaps,
        opening.precision,
        opening.steps,
        opening.precision_min,
        opening.precision_max,
        restart.gap_tangent,
        restart.tangent_precision,
        opening.unmoved,
    )
    return opened_segment(iterator, opened_state(iterator, start, restart.opening), opened)
end

function opened_segment(
        iterator::BifurcationKit.ContIterable,
        state::BifurcationKit.ContState,
        opened::TraceProgress,
    )::NamedTuple
    return continued_segment(
        iterator,
        state,
        BifurcationKit.ContResult(iterator, state),
        opened,
    )
end

"""The segment from `state`, its records saved in `branch` as BifurcationKit continues it."""
function continued_segment(
        iterator::BifurcationKit.ContIterable,
        state::BifurcationKit.ContState,
        branch::BifurcationKit.ContResult,
        opened::TraceProgress,
    )::NamedTuple
    # `guarded_continuation!` runs before `ended_segment` reads the branch it filled.
    return ended_segment(guarded_continuation!(iterator, state, branch), branch, opened)
end

ended_segment(
    ::Ended,
    branch::BifurcationKit.ContResult,
    opened::TraceProgress,
)::NamedTuple = (; progress = branch_progress(branch.branch, opened), exception = nothing)

ended_segment(
    thrown::Thrown,
    branch::BifurcationKit.ContResult,
    opened::TraceProgress,
)::NamedTuple = (; progress = branch_progress(branch.branch, opened), thrown.exception)

"""`restart_opening`, or the `Thrown` failure that `expected_failure` names; any other exception propagates."""
function guarded_restart(
        iterator::BifurcationKit.ContIterable,
        start::AbstractVector,
        factorization::Any,
        old_tangent::Any,
        opening::TraceProgress,
    )::Union{NamedTuple, GuardedEnd}
    try
        return restart_opening(iterator, start, factorization, old_tangent, opening)
    catch exception
        return thrown_failure(exception)
    end
end

"""BifurcationKit's `continuation!`: `Ended` once it ends, or the `Thrown` failure that `expected_failure` names."""
function guarded_continuation!(
        iterator::BifurcationKit.ContIterable,
        state::BifurcationKit.ContState,
        branch::BifurcationKit.ContResult,
    )::GuardedEnd
    try
        BifurcationKit.continuation!(iterator, state, branch)
        return Ended()
    catch exception
        return thrown_failure(exception)
    end
end

"""The `Thrown` failure holding `exception` if `expected_failure` names it; any other exception is rethrown."""
function thrown_failure(exception::X)::Thrown{X} where {X}
    if !expected_failure(exception)
        throw(exception)
    end
    return Thrown(exception)
end

"""Whether `exception` is a failure BifurcationKit reports by throwing: an `uncorrected_start` or a `solver_blowup`."""
expected_failure(exception::Any)::Bool = uncorrected_start(exception) ||
    solver_blowup(exception)

"""
    restart_opening(iterator, start, factorization, old_tangent, opening)

The restart of `iterator` along `restart_tangent` from `factorization` (see the four-argument method); `opening`'s
progress is not read.
"""
function restart_opening(
        iterator::BifurcationKit.ContIterable,
        start::AbstractVector{E},
        factorization::LU{E},
        old_tangent::AbstractVector{E},
        ::TraceProgress{E},
    )::NamedTuple where {E <: Real}
    return restart_opening(iterator, start, factorization, old_tangent)
end

function restart_opening(
        iterator::BifurcationKit.ContIterable,
        start::AbstractVector{E},
        ::Nothing,
        flat::OpeningTangent{E},
        ::TraceProgress{E},
    )::NamedTuple where {E <: Real}
    return opening_along(iterator, start, flat.tangent)
end

function restart_opening(
        ::BifurcationKit.ContIterable,
        ::AbstractVector,
        ::Nothing,
        ::AbstractVector,
        ::TraceProgress,
    )::NamedTuple
    # The swapped game was non-finite at the swap point (`ShiftedGame`); LAPACK's own check reports it so.
    throw(ArgumentError("matrix contains Infs or NaNs"))
end

"""
    reached_state(progress, outcome, game, pass_work, settings, context)

The `TraceResult` (`Ok(Traced)` or `Err(TraceFailure)`) of a trace whose `progress` holds the last accepted point and
whose `outcome` (`guarded_trace`) is the last corrector's step length, its `exhausted` record, or the BifurcationKit
exception that ended it. A corrector at rounding noise fails `PrecisionExhausted` (`MidTrace`) with that record, a
`StalledCorrection`. A point at the target passes one more Newton correction (`verified_endpoint`) on `game`'s pass
work `pass_work`. Any other stop fails for its `stop_cause`.
"""
function reached_state(
        progress::TraceProgress{E},
        outcome::NamedTuple,
        game::LogitGame,
        pass_work::Any,
        settings::BifurcationKitSettings{E},
        context::C,
    )::TraceResult{E, C} where {E <: Real, C <: NamedTuple}
    if !isnothing(outcome.exhausted)
        return Err(TraceFailure(TraceCause.PrecisionExhausted(MidTrace, outcome.exhausted), progress, outcome, context))
    end
    if isnothing(outcome.exception) && progress.precision == E(game.target_precision)
        return verified_endpoint(progress, outcome, game, pass_work, settings.corrector, context)
    end
    return Err(TraceFailure(stop_cause(progress, outcome, settings.max_steps), progress, outcome, context))
end

"""
Why a BifurcationKit trace whose last accepted point `progress` holds stopped short after `outcome`, in order: a start
that would not correct (`UncorrectedStart`), a broken linear solve (`Singular`, `MidTrace`), a step below resolution
(`UnresolvedStep`), a point that did not move (`UnmovedPoint`), `max_steps` spent (`MaxSteps`), or else `keep_going` or
a refusal (`StoppedShort`).
"""
function stop_cause(progress::TraceProgress{E}, outcome::NamedTuple, max_steps::Integer)::TraceCause.Type{E} where {E <: Real}
    uncorrected_start(outcome.exception) && return TraceCause.TraceStuck{E}(UncorrectedStart)
    isnothing(outcome.exception) || return TraceCause.Singular{E}(MidTrace)
    outcome.unresolved && return TraceCause.TraceStuck{E}(UnresolvedStep)
    progress.unmoved && return TraceCause.TraceStuck{E}(UnmovedPoint)
    progress.steps >= max_steps && return TraceCause.NoConvergence{E}(MaxSteps)
    return TraceCause.TraceStuck{E}(StoppedShort)
end

"""`pinned_correction`'s result: the corrected point and the residual it corrected, or the `TraceCause` that stopped it."""
const PinnedCorrection{E} = Result{@NamedTuple{corrected::Vector{E}, residual::Vector{E}}, TraceCause.Type{E}}

"""
    pinned_correction(game, point, pass_work)

The endpoint check's Newton correction of `point = [w; λ]` on `game`, with `λ` pinned (`branch_system!` on the pass
work `pass_work`): `Ok((; corrected, residual))`, the point itself at an exact root, which needs none. A non-finite
residual or Jacobian fails `NotFinite` (`AtEndpoint`), at an exact root too, and a singular Jacobian off one fails
`Singular` (`AtEndpoint`).
"""
function pinned_correction(game::LogitGame, point::AbstractVector{E}, pass_work::Any)::PinnedCorrection{E} where {E <: Real}
    system = branch_system!(BranchWorkspace(point, pass_work), game, point, last(point))
    (all(isfinite, system.residual) && all(isfinite, system.jacobian)) || return Err(TraceCause.NotFinite{E}(AtEndpoint))
    all(iszero, system.residual) && return Ok((; corrected = collect(point), system.residual))
    factorization = lu!(system.jacobian; check = false)
    issuccess(factorization) || return Err(TraceCause.Singular{E}(AtEndpoint))
    return Ok((; corrected = point .- factorization \ system.residual, system.residual))
end

"""
    verified_endpoint(progress, outcome, game, pass_work, corrector, context)

The `TraceResult` at `progress`, the last accepted point, on the target. One more iteration of the corrector that
landed it checks it. BifurcationKit corrects a step past `p_max` by Newton's method with `λ` pinned at the bound
(`_corrector_at_bound`), so the check is that step: a dense LU solve of `game`'s branch Jacobian on its pass work
`pass_work`, as `DefaultLS`. That
corrector can stop on a lucky draw of evaluation noise, a correction under the tolerance by chance (issue #134).

The correction is `pinned_correction`'s, which fails `NotFinite` or `Singular`. An unsettled correction (as `settled`:
past `corrector.logit_tolerance` beyond each strategy's `settle_ulps`, `log_probability_corrections`) fails `PrecisionExhausted` (`AtEndpoint`).
Its `StalledCorrection` runs from the endpoint (`gaps`, `precision`) to a plain Newton iterate pinned at the
endpoint's precision, arclength zero. Else the point is `Ok(Traced)`, its `logit_accuracy` at least the correction.
"""
function verified_endpoint(
        progress::TraceProgress{E},
        outcome::NamedTuple,
        game::LogitGame,
        pass_work::Any,
        corrector::CorrectorSettings{E},
        context::C,
    )::TraceResult{E, C} where {E <: Real, C <: NamedTuple}
    (; gaps, precision) = progress
    point = [gaps; precision]
    pinned = pinned_correction(game, point, pass_work)
    is_error(pinned) && return Err(TraceFailure(unwrap_error(pinned), progress, outcome, context))
    (; corrected) = unwrap(pinned)
    (; logit_tolerance, settle_ulps) = corrector
    ranges = checked_ranges(logit_ranges(game.pass!!), length(point) - 1)
    correction, excess = log_probability_corrections(point, corrected, ranges, settle_ulps * eps(E), one(E))
    if !(excess <= logit_tolerance)
        stall = StalledCorrection(precision, correction, corrected, precision, zero(E))
        return Err(TraceFailure(TraceCause.PrecisionExhausted(AtEndpoint, stall), progress, outcome, context))
    end
    return Ok(Traced(logits_at(point), progress.steps, max(E(outcome.logit_accuracy), correction)))
end

"""
    TraceFailure(cause, progress, outcome, context)

The failure for `cause` of a BifurcationKit trace whose last accepted point `progress` holds, after the segment
`outcome` (`guarded_trace`), with the caller's `context`. Its `TraceStop` holds that point and the step its last
corrector held, and its `BifurcationKitTrace` record the rest of `progress` and `outcome`.
"""
function TraceFailure(
        cause::TraceCause.Type{E},
        progress::TraceProgress{E},
        outcome::NamedTuple,
        context::C,
    )::TraceFailure{E, C} where {E <: Real, C <: NamedTuple}
    @debug "trace stopped" _group = :trace cause
    stop = TraceStop(progress.precision .* progress.gaps, progress.gaps, progress.precision, progress.steps, E(outcome.step_length))
    # The caught exception's text, empty when none ended the segment.
    message = isnothing(outcome.exception) ? "" : sprint(showerror, outcome.exception)
    record = EngineRecord.BifurcationKitTrace(
        ifelse(progress.steps > 0, E(outcome.accepted_step), zero(E)), progress.precision_min, progress.precision_max,
        progress.precision .* progress.gap_tangent .+ progress.tangent_precision .* progress.gaps, progress.gap_tangent,
        progress.tangent_precision, E(outcome.logit_accuracy), message,
    )
    return TraceFailure(cause, stop, record, context)
end
