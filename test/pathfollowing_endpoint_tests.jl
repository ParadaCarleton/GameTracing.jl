using ErrorTypes: is_error, unwrap, unwrap_error
using Random: Xoshiro

"""
The passes of games.jl that both engines trace in `element`: `(; name, pass, width, target)`. Every pass shape the
protocol allows: in-place passes, a pass with work arrays (`NormalFormGains`, `CountingPass`), vector and tuple
`logit_ranges`, players of three strategies, gains that move with the precision, and a branch that folds.
"""
function endpoint_cases(element::Type{<:Real})::Vector{NamedTuple}
    linear = LinearGainPass(element[0 1; 1 0], element[1 // 2, -1 // 4])
    rng = Xoshiro(29)
    random_passes = [(; name = Symbol(:normal_form_pass_, players), pass = NormalFormPass(map(table -> element.(table), random_payoffs(rng, players))), width = players, target = element(4)) for players in 2:5]
    three_by_three = NormalFormGains(ntuple(_ -> element.(3 .* randn(rng, 3, 3)), 2))
    return [
        (; name = :linear, pass = linear, width = 2, target = element(2)),
        # Past a fold at precision about 1.41 (fold_tests.jl).
        (; name = :fold, pass = ThresholdPass(element(6), element(3)), width = 1, target = element(2)),
        (; name = :damped, pass = DampedGainPass(linear, element(1 // 2)), width = 2, target = element(2)),
        (; name = :precision_gain, pass = PrecisionGainPass(), width = 1, target = element(3)),
        (; name = :window_noise, pass = WindowNoisePass(linear, element(1 // 10), element(1 // 1000)), width = 2, target = element(2)),
        random_passes...,
        (; name = :three_prisoners, pass = NormalFormGains(map(table -> element.(table), LOGITNASH_GAMES.three_prisoners)), width = 3, target = element(4)),
        (; name = :three_by_three, pass = three_by_three, width = 4, target = element(4)),
        (; name = :counting, pass = CountingPass(NormalFormGains(map(table -> element.(table), LOGITNASH_GAMES.stag_hunt)), Ref(0)), width = 2, target = element(2)),
    ]
end

@testset "both engines' trace_endpoint land on one root of every test pass, within their correctors' settle bounds ($element)" for element in (Float32, widen(Float32))
    bifurcation_kit, path_following = BifurcationKitSettings{element}(), PathFollowingSettings{element}()
    for (; name, pass, width, target) in endpoint_cases(element)
        reference, traced = (trace_endpoint(LogitGame(pass, target), width, settings) for settings in (bifurcation_kit, path_following))
        @test !is_error(reference) && !is_error(traced)
        (is_error(reference) || is_error(traced)) && continue
        reference_logits, logits = unwrap(reference).logits, unwrap(traced).logits
        # Each endpoint lies within its corrector's settle bound of the root, to first order.
        bound = settle_bound(bifurcation_kit.corrector, reference_logits) + settle_bound(path_following.distance, pass, logits)
        agree = maximum(abs, logits .- reference_logits) <= bound
        agree || @info "the engines' endpoints differ" name reference_logits logits bound
        @test agree
        @test unwrap(traced).logit_accuracy <= settle_bound(path_following.distance, pass, logits)
        @test unwrap(traced).steps > 0
    end
end

@testset "a PathFollowing trace builds its game's pass work once ($element)" for element in (Float32, widen(Float32))
    builds = Ref(0)
    stag_hunt = CountingPass(NormalFormGains(map(table -> element.(table), LOGITNASH_GAMES.stag_hunt)), builds)
    @test unwrap(trace_endpoint(LogitGame(stag_hunt, element(2)), 2, PathFollowingSettings{element}())) isa Traced
    @test builds[] == 1
end

"""A `keep_going` that records each precision it sees and stops past `limit`."""
struct RecordedStop{E <: Real}
    limit::E
    precisions::Vector{E}
end

function (stop::RecordedStop)(::AbstractVector, precision::Real)::Bool
    push!(stop.precisions, precision)
    return precision < stop.limit
end

"""An `accept_corrected` that records each `converged` flag it sees and refuses every point past `level`."""
struct RecordedRefusal{E <: Real}
    level::E
    flags::Vector{Bool}
end

function (refusal::RecordedRefusal)(::AbstractVector, precision::Real, converged::Bool)::Bool
    push!(refusal.flags, converged)
    return precision <= refusal.level
end

@testset "both engines fail alike: cause and shared context fields ($element)" for element in (Float32, widen(Float32))
    linear = LinearGainPass(element[0 1; 1 0], element[1 // 2, -1 // 4])
    target = element(2)
    for engine in (BifurcationKitSettings, PathFollowingSettings)
        settings = engine{element}(; max_step = element(1 // 20))
        # A non-finite start, with the caller's context beside the flat start.
        broken = LinearGainPass(element[NaN32 1; 1 0], element[1 // 2, -1 // 4])
        start = unwrap_error(trace_endpoint(LogitGame(broken, target), 2, settings; context = (; label = :broken)))
        @test (start.cause, start.context.label, start.stop.logits, start.stop.precision) == (TraceCause.NotFinite{element}(GameTracing.AtStart), :broken, zeros(element, 2), 0)
        # A zero target takes no step.
        flat = unwrap(trace_endpoint(LogitGame(linear, zero(element)), 2, settings))
        @test (flat.logits, flat.steps, flat.logit_accuracy) == (zeros(element, 2), 0, 0)
        # A spent step budget.
        budget = unwrap_error(trace_endpoint(LogitGame(linear, target), 2, engine{element}(; max_step = element(1 // 100), max_steps = 3)))
        @test failed_as(budget, TraceCause.NoConvergence)
        @test 0 < budget.stop.precision < target
        # A refusing `keep_going` stops the trace on the branch, short of the target.
        stopped = unwrap_error(trace_endpoint(LogitGame(linear, target), 2, settings; keep_going = RecordedStop(target / 2, element[])))
        @test failed_as(stopped, TraceCause.TraceStuck)
        (; logits, gaps, precision, steps, step_length) = stopped.stop
        @test 0 < precision < target && steps > 0 && step_length > 0
        @test logits ≈ precision .* gaps
        @test residual_norm(linear, logits, precision) < 1.0f-4
        # Refusing every corrector past a level stops the trace there.
        refused = unwrap_error(trace_endpoint(LogitGame(linear, target), 2, settings; accept_corrected = RecordedRefusal(target / 4, Bool[])))
        @test failed_as(refused, TraceCause.TraceStuck)
        @test refused.stop.precision <= target / 4
        # A game non-finite past a precision stops short of it; one non-finite at every positive precision, at the start.
        for poison in (one(element), zero(element))
            poisoned = unwrap_error(trace_endpoint(LogitGame(PoisonedPass(linear, poison), target), 2, settings))
            @test failed_as(poisoned, TraceCause.TraceStuck)
            @test poisoned.stop.precision <= poison
        end
        # A caller's exception and a pass without a gap bound propagate.
        @test_throws ArgumentError trace_endpoint(LogitGame(ThrowingPass(linear), target), 2, settings)
        @test_throws r"define GameTracing\.gap_bound" trace_endpoint(LogitGame(UnboundedPass(linear), target), 2, settings)
    end
end

@testset "PathFollowing's trace_endpoint runs keep_going and accept_corrected at each point ($element)" for element in (Float32, widen(Float32))
    linear = LinearGainPass(element[0 1; 1 0], element[1 // 2, -1 // 4])
    target = element(2)
    settings = PathFollowingSettings{element}()
    # Every accepted point, the endpoint included, rising to it; a `false` at the endpoint lands.
    stop = RecordedStop(target, element[])
    traced = unwrap(trace_endpoint(LogitGame(linear, target), 2, settings; keep_going = stop))
    @test length(stop.precisions) == traced.steps
    @test issorted(stop.precisions) && last(stop.precisions) == target && first(stop.precisions) > 0
    # Only converged points reach `accept_corrected`, and a refusal past a level fails as a refused step.
    refusal = RecordedRefusal(target / 4, Bool[])
    refused = unwrap_error(trace_endpoint(LogitGame(linear, target), 2, settings; accept_corrected = refusal))
    @test all(refusal.flags) && !isempty(refusal.flags)
    @test refused.engine.failure.last_refusal == PathFollowing.AcceptRefused
    @test refused.cause in (TraceCause.TraceStuck{element}(GameTracing.StepRefusals), TraceCause.TraceStuck{element}(GameTracing.UnresolvedStep))
    # A non-finite Jacobian at a finite start fails there as `NotFinite`; BifurcationKit's fails `Singular` or `TraceStuck`.
    nan_slope = unwrap_error(trace_endpoint(LogitGame(NaNSlopePass(linear, 0), target), 2, settings; context = (; label = :slope)))
    @test (nan_slope.cause, nan_slope.context.label) == (TraceCause.NotFinite{element}(GameTracing.AtStart), :slope)
    @test (nan_slope.stop.precision, nan_slope.stop.logits, nan_slope.stop.steps) == (0, zeros(element, 2), 0)
end

@testset "PathFollowing's trace_endpoint throws on a redraw; NoRedraw passes ($element)" for element in (Float32, widen(Float32))
    game = LogitGame(LinearGainPass(element[0 1; 1 0], element[1 // 2, -1 // 4]), element(2))
    settings = PathFollowingSettings{element}()
    @test_throws r"BifurcationKitSettings to redraw" trace_endpoint(game, 2, settings; redraw = SwapAt(game, one(element)))
    @test !is_error(trace_endpoint(game, 2, settings; redraw = GameTracing.NoRedraw()))
end

"""`pass`'s game under a type with a gap bound and no `logit_ranges`."""
struct RangelessGains{P}
    pass::P
end

(wrapper::RangelessGains)(arguments::Any...)::Tuple = wrapper.pass(arguments...)
GameTracing.gap_bound(wrapper::RangelessGains)::Real = GameTracing.gap_bound(wrapper.pass)

"""`pass`'s game of two logits under a type whose one logit range leaves the second logit out."""
struct MisrangedGains{P}
    pass::P
end

(wrapper::MisrangedGains)(arguments::Any...)::Tuple = wrapper.pass(arguments...)
GameTracing.gap_bound(wrapper::MisrangedGains)::Real = GameTracing.gap_bound(wrapper.pass)
GameTracing.logit_ranges(::MisrangedGains)::Tuple{UnitRange{Int}} = (1:1,)

@testset "a pass without logit_ranges cannot trace on PathFollowing, as every corrector measure reads them ($element)" for element in (Float32, widen(Float32))
    linear = LinearGainPass(element[0 1; 1 0], element[1 // 2, -1 // 4])
    plain = PathFollowingSettings{element}(; step_metric = TraceGeometry{element}(; distance = SummedDistance))
    @test_throws MethodError trace_endpoint(LogitGame(RangelessGains(linear), element(2)), 2, plain)
    # The default Fisher metric reads the ranges too, and the corrector checks that they tile the logits.
    @test_throws MethodError trace_endpoint(LogitGame(RangelessGains(linear), element(2)), 2, PathFollowingSettings{element}())
    @test_throws r"do not tile" trace_endpoint(LogitGame(MisrangedGains(linear), element(2)), 2, PathFollowingSettings{element}())
end

"""PathFollowing's endpoint check (`verified_landing`) of `game` at the point `[gaps; target]`, from a started tracer."""
function landed_at(game::LogitGame, gaps::AbstractVector{E}, settings::PathFollowingSettings{E})::TraceResult{E} where {E <: Real}
    problem = GameTracing.path_problem(game.pass!!, zero(gaps))
    tracer = unwrap(GameTracing.path_init(problem, settings, GameTracing.PositivePrecision(GameTracing.AcceptAll())))
    tracer.point .= [gaps; E(game.target_precision)]
    return GameTracing.verified_landing(tracer, settings.endpoint_tolerance, settings.endpoint_ulps, (; label = :landed))
end

@testset "the PathFollowing endpoint passes one more pinned Newton correction, or the trace fails ($element)" for element in (Float32, widen(Float32))
    settings = PathFollowingSettings{element}()
    tolerance = settings.distance.tolerance
    # Under its onset the noise is off: the gain is `level`, the root `w = level`, and `δ / λ` off it a correction `δ`.
    # At logit 1/2 the profile moves about 0.72 δ in Gambit's norm, so δ = tolerance settles and δ = 4 tolerance does not.
    flat = LogitGame(NoisyFlatPass(element(1 // 2), element(1.0f-7), element(10), UInt(1)), one(element))
    settled = unwrap(landed_at(flat, element[1 // 2 + tolerance], settings))
    @test settled.logits == element[1 // 2 + tolerance]
    # The accuracy is Gambit's norm of the correction, about 0.72 δ at logit 1/2.
    @test isapprox(settled.logit_accuracy, flat_gambit(tolerance); atol = 4 * eps(element))
    unsettled = unwrap_error(landed_at(flat, element[1 // 2 + 4 * tolerance], settings))
    @test (failed_as(unsettled, TraceCause.PrecisionExhausted), unsettled.cause.stage, unsettled.context.label) == (true, GameTracing.AtEndpoint, :landed)
    @test unsettled.cause.stall.precision == unsettled.cause.stall.pin == last(unsettled.cause.stall.iterate) == 1
    @test unsettled.cause.stall.iterate[1] ≈ 1 // 2
    @test unsettled.stop.gaps == element[1 // 2 + 4 * tolerance]
    @test isapprox(unsettled.cause.stall.correction, flat_gambit(4 * tolerance); atol = 4 * eps(element))
    # A strategy off the support moves almost no probability for a large logit move. `G = 10` puts the root at `w = 10`,
    # logit 100 at λ = 10, so a point at logit 90 corrects the reference's log probability by 10: both tests refuse it.
    saturated = LogitGame(LinearGainPass(zeros(element, 1, 1), element[10]), element(10))
    far = unwrap_error(landed_at(saturated, element[9], settings))
    @test (failed_as(far, TraceCause.PrecisionExhausted), far.cause.stage) == (true, GameTracing.AtEndpoint)
    @test isapprox(far.cause.stall.correction, 10; rtol = 4 * eps(element))
    # A loose endpoint passes it only beside a loose settle test in Gambit's norm.
    loose_distance = CorrectorDistance{element}(; tolerance = 20, target_correction = nothing)
    loose = PathFollowingSettings{element}(; distance = loose_distance, endpoint_tolerance = 20, split_check = nothing)
    @test (loose.endpoint_tolerance, loose.endpoint_ulps) == (element(20), 16)
    @test isapprox(unwrap(landed_at(saturated, element[9], loose)).logit_accuracy, 10; rtol = 4 * eps(element))
    # `G = logistic(z) + b` has Jacobian `1 - λ/4` at `w = 0`, singular at λ = 4. With `b = -1/2`, `w = 0` is an exact
    # root there and needs no correction; with `b = 0` it is no root, and Newton's method has no step.
    exact = unwrap(landed_at(LogitGame(LinearGainPass(ones(element, 1, 1), element[-1 // 2]), element(4)), zeros(element, 1), settings))
    @test (exact.logits, exact.steps, exact.logit_accuracy) == (zeros(element, 1), 0, 0)
    singular = unwrap_error(landed_at(LogitGame(LinearGainPass(ones(element, 1, 1), zeros(element, 1)), element(4)), zeros(element, 1), settings))
    @test singular.cause == TraceCause.Singular{element}(GameTracing.AtEndpoint)
    # A Jacobian non-finite only at the target fails `NotFinite`; LAPACK would throw on it.
    late_nan = LogitGame(NaNSlopePass(LinearGainPass(ones(element, 1, 1), zeros(element, 1)), 1), element(2))
    not_finite = unwrap_error(landed_at(late_nan, element[1 // 2], settings))
    @test (not_finite.cause, not_finite.context.label) == (TraceCause.NotFinite{element}(GameTracing.AtEndpoint), :landed)
    # At an exact root as well: `G = 1/2` has its root at `w = 1/2`.
    exact_nan = LogitGame(NaNSlopePass(LinearGainPass(zeros(element, 1, 1), element[1 // 2]), 1), element(2))
    @test unwrap_error(landed_at(exact_nan, element[1 // 2], settings)).cause == TraceCause.NotFinite{element}(GameTracing.AtEndpoint)
end

@testset "BifurcationKit's endpoint check fails NotFinite on a non-finite Jacobian ($element)" for element in (Float32, widen(Float32))
    # `G = 1/2` puts the root at `w = 1/2`; the check at `w = 1/4` needs a correction, and so the Jacobian.
    constant = LinearGainPass(zeros(element, 1, 1), element[1 // 2])
    progress(gap::Real) = GameTracing.TraceProgress(element[gap], element(2), 3, zero(element), element(2), zeros(element, 1), one(element), false)
    outcome = (; unresolved = false, step_length = one(element), accepted_step = one(element), logit_accuracy = zero(element), exception = nothing)
    work = GameTracing.pass_workspace(constant, zeros(element, 1))
    corrector = BifurcationKitSettings{element}().corrector
    late_nan = LogitGame(NaNSlopePass(constant, 1), element(2))
    failure = unwrap_error(GameTracing.verified_endpoint(progress(1 // 4), outcome, late_nan, work, corrector, (; label = :bk)))
    @test (failure.cause, failure.stop.steps, failure.context.label) == (TraceCause.NotFinite{element}(GameTracing.AtEndpoint), 3, :bk)
    @test unwrap_error(GameTracing.verified_endpoint(progress(1 // 2), outcome, late_nan, work, corrector, (;))).cause ==
        TraceCause.NotFinite{element}(GameTracing.AtEndpoint)
    # Under `from` the slope stays finite, and a point near the root passes.
    finite = LogitGame(NaNSlopePass(constant, 3), element(2))
    near = unwrap(GameTracing.verified_endpoint(progress(1 // 2 + 1 // 2^20), outcome, finite, work, corrector, (;)))
    # The correction moves the gap by 2⁻²⁰, the logit at λ = 2 by twice that: its Gambit's norm is the accuracy.
    accuracy = GameTracing.log_probability_change(element[1 // 2 + 1 // 2^20, 2], element[1 // 2, 2], (1:1,))
    @test near.steps == 3 && isapprox(near.logit_accuracy, accuracy; rtol = sqrt(eps(element)))
end
