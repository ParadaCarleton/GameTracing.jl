using ErrorTypes: is_error, unwrap, unwrap_error
using Logging: Logging
using LinearAlgebra: LAPACKException, SingularException

const FAILURE_SETTINGS = BifurcationKitSettings{Float32}(max_step = 0.05f0)

@testset "a non-finite response past a precision stops the trace short of it" begin
    result = trace_endpoint(
        LogitGame(PoisonedPass(GAME, 1.0f0), TARGET_PRECISION),
        WIDTH,
        FAILURE_SETTINGS,
    )
    @test is_error(result)
    @test unwrap_error(result).cause == GameTracing.TraceStuck
    @test unwrap_error(result).context.precision <= 1.0f0
end

@testset "a non-finite Jacobian is a failure value" begin
    result = trace_endpoint(
        LogitGame(NaNSlopePass(GAME, 0), TARGET_PRECISION),
        WIDTH,
        FAILURE_SETTINGS,
    )
    @test is_error(result)
    @test unwrap_error(result).cause in (GameTracing.Singular, GameTracing.TraceStuck)
end

@testset "a caller's exception propagates" begin
    @test_throws ArgumentError trace_endpoint(
        LogitGame(ThrowingPass(GAME), TARGET_PRECISION),
        WIDTH,
        FAILURE_SETTINGS,
    )
end

struct RefuseConverged end

(::RefuseConverged)(::AbstractVector, ::Real, ::Bool)::Bool = false

@testset "refusing every corrector ends the trace" begin
    result = trace_endpoint(
        LogitGame(GAME, TARGET_PRECISION),
        WIDTH,
        FAILURE_SETTINGS;
        accept_corrected = RefuseConverged(),
    )
    @test is_error(result)
    @test unwrap_error(result).cause == GameTracing.TraceStuck
end

"""An `accept_corrected` that refuses every corrector past precision `level`."""
struct RefusePast{E <: Real}
    level::E
end

(refuse::RefusePast)(::AbstractVector, precision::Real, ::Bool)::Bool = precision <= refuse.level

# Refused from the flat start, where λ = 0, and mid-branch.
@testset "halving ends once the predictor cannot move the point ($(typeof(refusal)))" for refusal in (RefuseConverged(), RefusePast(0.25f0 * TARGET_PRECISION))
    logs, result = Test.collect_test_logs(min_level = Logging.Debug) do
        trace_endpoint(LogitGame(GAME, TARGET_PRECISION), WIDTH, FAILURE_SETTINGS; accept_corrected = refusal)
    end
    messages = [(log.message, get(log.kwargs, :reason, nothing)) for log in logs]
    @test unwrap_error(result).cause == GameTracing.TraceStuck
    # At λ = 0 any move of λ resolves: halving ends at BifurcationKit's zero step, never at an unresolved predictor.
    @test count(==(("corrector stopped", :unresolved)), messages) == ifelse(refusal isa RefuseConverged, 0, 1)
    # Halving 0.05 to Float32's smallest subnormal 2^-149 takes log2(0.05) + 149 ≈ 145 steps (measured: 143 from
    # the flat start, 41 mid-branch); 15 more allow accepted steps before the refusals.
    @test count(message -> first(message) == "corrector closed", messages) < 160
end

@testset "a predictor that leaves the point unchanged is refused, even after a settled solve" begin
    point = Float32[1, -2, 0.5f0]
    check = GameTracing.CorrectorReach(
        GameTracing.AcceptAll(),
        CorrectorSettings{Float32}(),
        TraceGeometry{Float32}(),
        TARGET_PRECISION,
        copy(point),
        copy(point),
        0.0f0,
        nothing,
    )
    GameTracing.restate!(check; settled_here = true)
    @test !check((; x = point .+ eps.(point), z0 = (; u = point), step = 0, residual = 0.0f0); fromNewton = false)
    @test check.state[].unresolved
    @test !check((; x = point, residual = 0.0f0, step = 1, p = 1.0f0, closing = true); fromNewton = false)
    # Each coordinate's move is measured against its own size: any move of a zero coordinate resolves.
    @test !GameTracing.step_unresolved(Float32[0, 1], Float32[1.0f-30, 1])
    @test GameTracing.step_unresolved(Float32[0, 1], Float32[0, 1 + eps(1.0f0)])
    # A large coordinate does not hide a small one's move.
    @test !GameTracing.step_unresolved(Float32[-5.0f0, 13.0f0], Float32[-5.0f0, 14.0f0])
    @test !GameTracing.step_unresolved(point, point .+ Float32[0, 0, 1.0f-3])
end

@testset "a corrector reach too short to hold any step ends the trace" begin
    result = trace_endpoint(
        LogitGame(FOLD_GAME, 2.0f0), 1, BifurcationKitSettings{Float32}(
            corrector = CorrectorSettings{Float32}(reach = 1.0f-6),
            max_step = 0.1f0,
        )
    )
    @test is_error(result)
    # A step that halves to the float type's resolution after reach refusals is labeled PrecisionExhausted
    # (`unresolved_noise`): a refused correction under `logit_tolerance` is rounding noise too. The trace still fails.
    @test unwrap_error(result).cause in (GameTracing.TraceStuck, GameTracing.NoConvergence, GameTracing.PrecisionExhausted)
    @test isfinite(unwrap_error(result).context.precision)
end

@testset "a reach failure that halves to an unresolved step exhausts the precision at the failed iterate and its step ($element)" for element in (Float32, widen(Float32))
    start, predicted, far = element[1, -2, 1 // 2], element[1, -2, 3 // 4], element[3, -2, 3 // 4]
    check = GameTracing.CorrectorReach(
        GameTracing.AcceptAll(), CorrectorSettings{element}(), TraceGeometry{element}(), element(TARGET_PRECISION), start, predicted, one(element), nothing,
    )
    payload = (; p = element(3 // 4), ds = element(1 // 4), residual = zero(element))
    @test check((; payload..., x = predicted, z0 = (; u = start), step = 0); fromNewton = false)
    # The predictor moved a logit by 0.5; this iterate moves one 1.5 further, past the reach of twice that.
    @test !check((; payload..., x = far, step = 1); fromNewton = false)
    @test check.stalled == far
    @test check.state[].stalled_step == (; pin = payload.p, arclength = payload.ds)
    # The halved step's predictor is the start: no smaller step resolves, so the reach failure confirms noise.
    @test !check((; payload..., x = start, z0 = (; u = start), step = 0, ds = payload.ds / 2); fromNewton = false)
    @test GameTracing.noise_confirmed(check)
    @test GameTracing.exhausted_record(check) ==
        (; precision = payload.p, correction = element(3 // 2), stalled_iterate = far, stalled_pin = payload.p, stalled_arclength = payload.ds)
    # Only PALC's corrector stalls on a step; a plain Newton iterate has none to record.
    @test_throws ArgumentError GameTracing.record_stall!(check, (; x = far, step = 1, residual = zero(element)))
end

@testset "exceptions BifurcationKit and LAPACK throw are told apart from a caller's" begin
    @test GameTracing.uncorrected_start("Stopping continuation.")
    @test GameTracing.uncorrected_start(
        ErrorException(
            "Newton failed to converge at the start",
        )
    )
    @test !GameTracing.uncorrected_start(ErrorException("another"))
    @test !GameTracing.uncorrected_start("another")
    @test !GameTracing.uncorrected_start(ArgumentError("bad game"))
    @test GameTracing.solver_blowup(SingularException(2))
    @test GameTracing.solver_blowup(LAPACKException(2))
    @test GameTracing.solver_blowup(ArgumentError("matrix contains Infs or NaNs"))
    @test GameTracing.solver_blowup(ArgumentError("LAPACK failed"))
    @test !GameTracing.solver_blowup(ArgumentError("bad game"))
    @test !GameTracing.solver_blowup(DimensionMismatch("bad game"))
end

@testset "a bordered matrix of the wrong size is rejected" begin
    @test_throws DimensionMismatch GameTracing.bordered_matrix!(
        zeros(Float32, 2, 2),
        ones(Float32, 2, 2),
        ones(Float32, 2),
        ones(Float32, 2),
        1.0f0,
        1.0f0,
        1.0f0,
        nothing,
    )
end

@testset "a triangular bordered system solves like the dense one" begin
    solver = GameTracing.bordered_linear_solver(GameTracing.MatrixSolve, 2, Float32)
    jacobian = Float32[2 0; 0 3]
    border = Float32[0, 0]
    solution, parameter, _, _ = solver(
        jacobian,
        border,
        Float32[0, 0],
        1.0f0,
        Float32[2, 3],
        1.0f0,
    )
    @test collect(solution) ≈ Float32[1, 1]
    @test parameter ≈ 1.0f0
end

@testset "a predictor that did not move has no reach to exceed" begin
    check = GameTracing.CorrectorReach(
        GameTracing.AcceptAll(),
        CorrectorSettings{Float32}(),
        TraceGeometry{Float32}(),
        TARGET_PRECISION,
        zeros(Float32, 3),
        zeros(Float32, 3),
        0.0f0,
        nothing,
    )
    @test GameTracing.within_reach(check, ones(Float32, 3))
end

@testset "reach measures logits: a far target's gap rounding is within it, a logit jump is not" begin
    # At λ = 10 a gap move of 1e-8 moves the logit 1e-7.
    previous, predicted = Float32[0, 10.0f0], Float32[1.0f-6, 10.0f0]
    check = GameTracing.CorrectorReach(
        GameTracing.AcceptAll(),
        CorrectorSettings{Float32}(),
        TraceGeometry{Float32}(),
        1.0f6,
        previous,
        predicted,
        1.0f0,
        nothing,
    )
    GameTracing.restate!(check; predictor_move = GameTracing.logit_change(previous, predicted))
    @test GameTracing.within_reach(check, predicted .+ Float32[1.0f-8, 0])
    @test !GameTracing.within_reach(check, predicted .+ Float32[3.0f-6, 0])
end

@testset "a game that is non-finite at every positive precision fails at the first step" begin
    result = trace_endpoint(
        LogitGame(PoisonedPass(GAME, 0.0f0), TARGET_PRECISION),
        WIDTH,
        FAILURE_SETTINGS,
    )
    @test is_error(result)
    @test unwrap_error(result).cause == GameTracing.TraceStuck
    # The two-point start counts as BifurcationKit's step 1; no point was accepted off the flat start.
    @test unwrap_error(result).context.precision == 0
end

@testset "BifurcationKit's report of an uncorrectable start is a stuck trace" begin
    progress = GameTracing.TraceProgress(
        zeros(Float32, 2),
        0.0f0,
        0,
        0.0f0,
        0.0f0,
        zeros(Float32, 2),
        0.0f0,
        false,
    )
    result = GameTracing.reached_state(
        progress,
        (; step_length = 0.0f0, accepted_step = 0.0f0, exhausted = nothing, logit_accuracy = 0.0f0, unresolved = false, exception = "Stopping continuation."),
        LogitGame(GAME, TARGET_PRECISION),
        pass_workspace(GAME, progress.gaps),
        BifurcationKitSettings{Float32}(; max_steps = 10),
        (;),
    )
    @test is_error(result)
    @test unwrap_error(result).cause == GameTracing.TraceStuck
end

"""The `guarded_trace` outcome of a segment that ran to its target, its accepted points settled within `accuracy`."""
function ran_outcome(accuracy::E)::NamedTuple where {E <: Real}
    return (; step_length = zero(E), accepted_step = zero(E), exhausted = nothing, logit_accuracy = accuracy, unresolved = false, exception = nothing)
end

"""`reached_state` of a trace of `game` to `gaps` on its target, its accepted points settled within `accuracy`."""
function reached_at(game::LogitGame, gaps::AbstractVector{E}, accuracy::E)::GameTracing.TraceResult{E} where {E <: Real}
    target = E(game.target_precision)
    progress = GameTracing.TraceProgress(collect(gaps), target, 1, zero(E), target, zero(gaps), one(E), false)
    return GameTracing.reached_state(progress, ran_outcome(accuracy), game, pass_workspace(game.pass!!, progress.gaps), BifurcationKitSettings{E}(), (;))
end

@testset "the endpoint passes one more pinned Newton correction, or the trace fails ($element)" for element in (Float32, widen(Float32))
    tolerance = BifurcationKitSettings{element}().corrector.logit_tolerance
    # Under its onset the noise is off: the gain is `level`, the root `w = level`, and `δ / λ` off it a correction `δ`.
    flat = LogitGame(NoisyFlatPass(element(1 // 2), element(1.0f-7), element(10), UInt(1)), one(element))
    settled = unwrap(reached_at(flat, element[1 // 2 + tolerance / 2], zero(element)))
    @test settled.logits == element[1 // 2 + tolerance / 2]
    @test isapprox(settled.logit_accuracy, tolerance / 2; atol = 4 * eps(element))
    # The corrector's own accuracy stands when it is the larger.
    @test unwrap(reached_at(flat, element[1 // 2 + tolerance / 2], tolerance)).logit_accuracy == tolerance
    unsettled = unwrap_error(reached_at(flat, element[1 // 2 + 2 * tolerance], zero(element)))
    @test unsettled.cause == GameTracing.PrecisionExhausted
    # The correction runs from the endpoint to a plain Newton iterate at the target.
    @test unsettled.context.lambda == unsettled.context.stalled_pin == last(unsettled.context.stalled_iterate) == 1
    @test unsettled.context.stalled_iterate[1] ≈ 1 // 2
    @test unsettled.context.gaps == element[1 // 2 + 2 * tolerance]
    @test isapprox(unsettled.context.correction, 2 * tolerance; atol = 4 * eps(element))
    # A corrector can stop on a lucky draw of noise (issue #134). At the target `λ η` is 0.01, a hundred tolerances.
    target = element(1.0f5)
    noisy = LogitGame(NoisyFlatPass(element(1.0f-4), element(1.0f-7), element(10), UInt(3)), target)
    lucky = unwrap_error(reached_at(noisy, element[1.0f-4], zero(element)))
    @test lucky.cause == GameTracing.PrecisionExhausted
    @test lucky.context.lambda == target
    # One draw of at most `λ η`, plus the rounding of the logit `λ level`.
    @test tolerance < lucky.context.correction <= target * noisy.pass!!.noise + 2 * eps(target * noisy.pass!!.level)
    # `G = logistic(z) + b` has Jacobian `1 - λ/4` at `w = 0`, singular at λ = 4. With `b = -1/2`, `w = 0` is an exact
    # root there and needs no correction; with `b = 0` it is no root, and Newton's method has no step.
    exact = unwrap(reached_at(LogitGame(LinearGainPass(ones(element, 1, 1), element[-1 // 2]), element(4)), zeros(element, 1), tolerance))
    @test exact.logits == zeros(element, 1)
    @test exact.logit_accuracy == tolerance
    singular = unwrap_error(reached_at(LogitGame(LinearGainPass(ones(element, 1, 1), zeros(element, 1)), element(4)), zeros(element, 1), zero(element)))
    @test singular.cause == GameTracing.Singular
end

"""A `trace_record_point` record of BifurcationKit's branch at `step`: the point `(gaps, param)` and a unit tangent."""
function branch_record(
        gaps::AbstractVector{E},
        param::E,
        step::Integer,
    )::NamedTuple where {E <: Real}
    return (; gaps, gap_tangent = zero(gaps), tangent_precision = one(E), param, step)
end

"""`finalise`'s verdict on the accepted `point` (a `branch_record`) of a branch that so far holds `branch`."""
function finalised(
        finalise::GameTracing.TraceFinalise,
        point::NamedTuple,
        branch::AbstractVector,
    )::Bool
    # BifurcationKit calls `finalise` before it saves the point to the branch, with the point's tangent.
    tangent = (; u = [point.gap_tangent; point.tangent_precision], p = point.tangent_precision)
    # Only a redraw reads the algorithm, so a flat game of the records' width serves.
    width = length(point.gaps)
    flat = LogitGame(LinearGainPass(zeros(typeof(point.param), width, width), zeros(typeof(point.param), width)), TARGET_PRECISION)
    return finalise(
        (; u = [point.gaps; point.param], p = point.param),
        tangent,
        point.step,
        (; branch);
        state = (; ds = one(point.param) / 10),
        iter = (; alg = GameTracing.palc_algorithm(BifurcationKitSettings{typeof(point.param)}(), flat, width)),
    )
end

@testset "an accepted point equal to the last one stops the trace as a stall" begin
    game = LogitGame(GAME, TARGET_PRECISION)
    finalise = GameTracing.TraceFinalise(
        GameTracing.AcceptAll(),
        GameTracing.NoRedraw(),
        GameTracing.RedrawSlot(game, false, 0.1f0, pass_workspace(game.pass!!, zeros(Float32, WIDTH))),
        GameTracing.CorrectorSettings{Float32}(),
    )
    start, moved = branch_record(Float32[0, 0], 0.0f0, 0),
        branch_record(Float32[0.1f0, 0.2f0], 0.5f0, 1)
    @test finalised(finalise, moved, [start])
    @test !finalised(finalise, branch_record(moved.gaps, moved.param, 2), [start, moved])
    # A change in the precision alone, or in the logits alone, is progress.
    @test finalised(finalise, branch_record(moved.gaps, 0.6f0, 2), [start, moved])
    @test finalised(
        finalise,
        branch_record(Float32[0.3f0, 0.2f0], moved.param, 2),
        [start, moved],
    )

    opening = GameTracing.TraceProgress(
        zeros(Float32, 2),
        0.0f0,
        0,
        0.0f0,
        0.0f0,
        zeros(Float32, 2),
        0.0f0,
        false,
    )
    @test GameTracing.branch_progress([start], opening) === opening
    progress = GameTracing.branch_progress([start, moved], opening)
    @test !progress.unmoved
    @test (progress.steps, progress.precision_max) == (1, moved.param)
    @test progress.gaps == moved.gaps && progress.precision == moved.param
    outcome = (; step_length = 0.0f0, accepted_step = 0.0f0, exhausted = nothing, logit_accuracy = 0.0f0, unresolved = false, exception = nothing)
    settings = BifurcationKitSettings{Float32}(; max_steps = 2)
    @test unwrap_error(
        GameTracing.reached_state(
            progress,
            outcome,
            game,
            pass_workspace(game.pass!!, progress.gaps),
            settings,
            (;),
        )
    ).cause ==
        GameTracing.TraceStuck
    stuck = GameTracing.branch_progress(
        [start, moved, branch_record(moved.gaps, moved.param, 2)],
        opening,
    )
    @test stuck.unmoved
    @test stuck.steps == 2
    # A stall that spends the step budget is still a stall; moving points that spend it are not.
    stalled = unwrap_error(GameTracing.reached_state(stuck, outcome, game, pass_workspace(game.pass!!, stuck.gaps), settings, (;)))
    @test stalled.cause == GameTracing.TraceStuck
    @test stalled.context.unmoved
    spent = GameTracing.branch_progress(
        [start, moved, branch_record(moved.gaps, 0.6f0, 2)],
        opening,
    )
    @test !spent.unmoved
    @test unwrap_error(
        GameTracing.reached_state(
            spent,
            outcome,
            game,
            pass_workspace(game.pass!!, spent.gaps),
            settings,
            (;),
        )
    ).cause ==
        GameTracing.NoConvergence
end

"""
The records of a switchback in the (gap, param) plane: up a sheet with a wiggle, a left U-turn, down a sheet, a right
U-turn and up a third sheet. Sheets take steps of 0.3 and U-turns of 0.05, so adjacent sheets lie closer than a step.
"""
function switchback_branch(::Type{E})::AbstractVector where {E <: Real}
    turn = [E(pi) * index / 5 for index in 1:4]
    headings = [zeros(E, 4); fill(E(1) / 5, 2); zeros(E, 6); turn; fill(E(pi), 20); reverse(turn); zeros(E, 24)]
    lengths = [fill(E(3) / 10, 12); fill(E(1) / 20, 4); fill(E(3) / 10, 20); fill(E(1) / 20, 4); fill(E(3) / 10, 24)]
    # Record `k` holds the point reached by the steps before it, heading `headings[k]`; the branch starts at (0, 3).
    gaps = cumsum([zero(E); (lengths .* sin.(headings))[1:(end - 1)]])
    params = E(3) .+ cumsum([zero(E); (lengths .* cos.(headings))[1:(end - 1)]])
    return [
        (; gaps = [gaps[index]], gap_tangent = [sin(headings[index])], tangent_precision = cos(headings[index]), param = params[index], step = index)
            for index in eachindex(headings)
    ]
end

@testset "a switchback whose sheets lie closer than a step keeps the trace going ($element)" for element in (Float32, widen(Float32))
    branch = switchback_branch(element)
    distance(left::NamedTuple, right::NamedTuple)::element = hypot(only(left.gaps) - only(right.gaps), left.param - right.param)
    # Records on later sheets lie closer to a record on an earlier sheet than to the record before them.
    near = [index for index in 3:length(branch) if any(earlier -> distance(branch[index], branch[earlier]) < distance(branch[index], branch[index - 1]), 1:(index - 2))]
    @test length(near) > 20
    finalise = GameTracing.TraceFinalise(
        GameTracing.AcceptAll(),
        GameTracing.NoRedraw(),
        GameTracing.RedrawSlot(LogitGame(GAME, one(element)), false, element(1) / 10, pass_workspace(GAME, zeros(element, 1))),
        GameTracing.CorrectorSettings{typeof(element(1))}(),
    )
    @test all(index -> finalised(finalise, branch[index], branch[1:(index - 1)]), 2:length(branch))
end
