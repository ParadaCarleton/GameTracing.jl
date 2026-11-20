using ErrorTypes: is_error, unwrap, unwrap_error
using LinearAlgebra: I, lu
using Random: AbstractRNG, Xoshiro

# Random normal-form games whose trace to precision 1e6 once opened at λ ≈ 42426 and stalled off the map.
const FAR_OPENING_SEED = 0xb0bf341028e24ba6
# The seed whose (2, 2, 2) game stalled at λ ≈ 22.9 on its way to 1e6, under a residual tolerance below P G's rounding.
const MIXED_STALL_SEED = 0x233ce38d667063a1
# The seed whose (3, 3) game's corrector made two zero-length Newton steps at λ ≈ 34: with tolerance 0 the step control's
# contraction rate was 0 / 0, so the step length went NaN and the step halved forever.
const ZERO_STEP_SEED = 0x4e6997513e7cba6d
# The seed whose (3, 3) game folds at λ ≈ 18.02, where the per-logit Bernoulli weights exhausted Float32.
const FAR_FOLD_SEED = 0xe291aba5864127e0
# Game 1's Float32 trace exhausts its precision at λ ≈ 26.1 by rounding, 2e-6 from Gambit; Float64 reaches 50.
const ROUNDING_EXHAUSTED_SEED = 0x3ff2026d53896c44

"""Normal-form games with strategy counts `(3, 3)` and `(2, 2, 2)`, payoffs `3 randn`, drawn from `rng` in that order."""
function far_opening_games(rng::AbstractRNG)::AbstractVector
    return [ntuple(_ -> 3 .* randn(rng, Float32, counts), length(counts)) for counts in ((3, 3), (2, 2, 2))]
end

"""A `keep_going` that stops a trace at the first accepted point at or past precision `precision`."""
struct StopAtPrecision{E <: Real}
    precision::E
end

(stop::StopAtPrecision)(::AbstractVector, precision::Real)::Bool = precision < stop.precision

@testset "a frozen Fisher metric in Rounded numbers computes the engine's products ($element)" for element in (Float32, widen(Float32))
    frozen = PathFollowing.frozen_metric(GameTracing.FisherMetric((1:2, 3:3), one(element), LogPrecisionTerm, element(10), element(127 // 128)), element[1, -1, 2, 3])
    left, right = element[1, 2, -1, 3], element[-1, 1, 2, 2]
    number = Rounded{element, widen(element)}
    rounded = PathFollowing.metric_inner(rounded_frozen(frozen, number), number.(left), number.(right))
    @test rounded.value == PathFollowing.metric_inner(frozen, left, right)
    @test rounding_bound(rounded) > 0
end

const FAR_TARGET = 1.0f6
"""`BifurcationKitSettings` in `E` that step in the Fisher distance with the log precision term."""
function log_precision_settings(::Type{E})::BifurcationKitSettings{E} where {E <: Real}
    return BifurcationKitSettings{E}(; step_metric = TraceGeometry{E}(; distance = FisherDistance, precision_term = LogPrecisionTerm))
end

const LOG_PRECISION_SETTINGS = log_precision_settings(Float32)

"""
Whether a trace of `gains` to `FAR_TARGET` under `settings` that stopped with `failure` reached precision 50, or
exhausted its precision short of it by rounding (`exhausted_by_rounding`).
"""
function reached_or_rounded(gains::NormalFormGains, failure::TraceFailure, settings::BifurcationKitSettings)::Bool
    return failure.context.precision >= 50 || exhausted_by_rounding(gains, failure, settings, FAR_TARGET)
end

@testset "a trace to a far target opens one first step from the flat start in its own distance" begin
    for payoffs in (far_opening_games(Xoshiro(FAR_OPENING_SEED))..., far_opening_games(fresh_rng())...)
        gains = NormalFormGains(payoffs)
        failure = unwrap_error(
            trace_endpoint(
                LogitGame(gains, FAR_TARGET),
                GameTracing.logit_count(gains),
                LOG_PRECISION_SETTINGS;
                keep_going = StopAtPrecision(zero(Float32)),
            )
        )
        # A first step of length `h` in `log(1 + λ)` reaches at most `expm1(h)`.
        opening = GameTracing.opening_step(LogitGame(gains, FAR_TARGET), LOG_PRECISION_SETTINGS.step_metric, LOG_PRECISION_SETTINGS.max_step)
        @test 0 < failure.context.precision <= 2 * expm1(opening)
    end
end

@testset "a Float32 trace that exhausts its precision does so at rounding noise, which Float64 passes" begin
    payoffs_32 = first(far_opening_games(Xoshiro(ROUNDING_EXHAUSTED_SEED)))
    gains_32 = NormalFormGains(payoffs_32)
    failure = unwrap_error(
        trace_endpoint(
            LogitGame(gains_32, FAR_TARGET),
            GameTracing.logit_count(gains_32),
            LOG_PRECISION_SETTINGS;
            keep_going = StopAtPrecision(50.0f0),
        )
    )
    @test failure.cause == GameTracing.PrecisionExhausted
    @test failure.context.precision < 50
    @test exhausted_by_rounding(gains_32, failure, LOG_PRECISION_SETTINGS, FAR_TARGET)
    # The rounded residual is the engine's Float32 residual, and its bound covers the error measured in BigFloat.
    (; stalled_iterate, stalled_pin, correction) = failure.context
    game_32 = LogitGame(gains_32, FAR_TARGET)
    bounded = rounded_residual(game_32, stalled_iterate, stalled_pin)
    computed = branch_residual(game_32, stalled_iterate, stalled_pin)
    @test [entry.value for entry in bounded] == computed
    game_big = LogitGame(NormalFormGains(map(payoff -> big.(payoff), payoffs_32)), big(FAR_TARGET))
    measured = abs.(big.(computed) .- branch_residual(game_big, big.(stalled_iterate), big(stalled_pin)))
    @test all(measured .<= rounding_bound.(bounded))
    # A correction within the bound is rounding noise. One past it is not, nor is an iterate 64 ulps or a gap off the root.
    explained = rounding_explanation(gains_32, failure, LOG_PRECISION_SETTINGS, FAR_TARGET)
    for (scale, noise) in ((2 // 3, true), (3 // 2, false))
        scaled = TraceFailure(failure.cause, (; failure.context..., correction = Float32(scale * maximum(explained.bound))))
        @test exhausted_by_rounding(gains_32, scaled, LOG_PRECISION_SETTINGS, FAR_TARGET) == noise
    end
    for moved_gap in (nextfloat(first(stalled_iterate), 64), first(stalled_iterate) + 1)
        moved = TraceFailure(failure.cause, (; failure.context..., stalled_iterate = [moved_gap; stalled_iterate[2:end]]))
        @test !exhausted_by_rounding(gains_32, moved, LOG_PRECISION_SETTINGS, FAR_TARGET)
    end
    # An earlier root of the branch solves `F` but lies off the stalled step's arclength constraint.
    earlier = unwrap_error(
        trace_endpoint(
            LogitGame(gains_32, FAR_TARGET), GameTracing.logit_count(gains_32), LOG_PRECISION_SETTINGS; keep_going = StopAtPrecision(5.0f0),
        )
    ).context
    elsewhere = TraceFailure(
        failure.cause,
        (; failure.context..., stalled_iterate = [earlier.gaps; earlier.precision], stalled_pin = earlier.precision, correction = 0.0f0),
    )
    off_step = rounding_explanation(gains_32, elsewhere, LOG_PRECISION_SETTINGS, FAR_TARGET)
    @test all(abs.(off_step.residual[1:(end - 1)]) .<= off_step.residual_bound[1:(end - 1)])
    @test abs(last(off_step.residual)) > last(off_step.residual_bound)
    @test !exhausted_by_rounding(gains_32, elsewhere, LOG_PRECISION_SETTINGS, FAR_TARGET)
    roundoff = eps(Float32) / 2
    @info "rounding explains the stall" correction bound = maximum(explained.bound) ratio = correction / maximum(explained.bound) residual_ratio = maximum(abs.(explained.residual) ./ explained.residual_bound) off_step_ratio = abs(last(off_step.residual)) / last(off_step.residual_bound) noise_roundoffs = round.(Tuple(explained.noise ./ roundoff); sigdigits = 3) measured_roundoffs = round.(Tuple(widen(Float32).(measured ./ roundoff)); sigdigits = 3)
    wide = typeof(float(1))
    payoffs_wide = map(payoff -> wide.(payoff), payoffs_32)
    gains_wide = NormalFormGains(payoffs_wide)
    settings_wide = log_precision_settings(wide)
    reached = unwrap_error(
        trace_endpoint(
            LogitGame(gains_wide, wide(FAR_TARGET)),
            GameTracing.logit_count(gains_wide),
            settings_wide;
            keep_going = StopAtPrecision(wide(50)),
        )
    )
    @test reached.context.precision >= 50
    # At an accepted Float64 root a zero step's record is at noise; moved by a relative √eps it is not.
    root = [reached.context.gaps; reached.context.precision]
    resting = TraceFailure(
        GameTracing.PrecisionExhausted,
        (; reached.context..., stalled_iterate = root, stalled_pin = last(root), stalled_arclength = zero(wide), correction = zero(wide)),
    )
    @test exhausted_by_rounding(gains_wide, resting, settings_wide, FAR_TARGET)
    shifted = TraceFailure(resting.cause, (; resting.context..., stalled_iterate = [first(root) * (1 + sqrt(eps(wide))); root[2:end]]))
    @test !exhausted_by_rounding(gains_wide, shifted, settings_wide, FAR_TARGET)
    # A Float64 record short of 50 meets the far-target check in its own settings. An accepted root is only within the
    # corrector's tolerance, so three Newton steps at its pinned precision polish it to rounding.
    early = unwrap_error(
        trace_endpoint(LogitGame(gains_wide, wide(FAR_TARGET)), GameTracing.logit_count(gains_wide), settings_wide; keep_going = StopAtPrecision(wide(5))),
    ).context
    game_wide = LogitGame(gains_wide, wide(FAR_TARGET))
    residual = GameTracing.GameResidual(game_wide, GameTracing.BranchWorkspace(game_wide, [early.gaps; early.precision]))
    jacobian = GameTracing.GameJacobian(game_wide, GameTracing.BranchWorkspace(game_wide, [early.gaps; early.precision]))
    newton = (point, _) -> point .- jacobian(point, early.precision) \ residual(point, early.precision)
    polished = foldl(newton, 1:3; init = [early.gaps; early.precision])
    stopped_early = TraceFailure(
        GameTracing.PrecisionExhausted,
        (; early..., gaps = polished[1:(end - 1)], stalled_iterate = polished, stalled_pin = early.precision, stalled_arclength = zero(wide), correction = zero(wide)),
    )
    @test early.precision < 50
    @test reached_or_rounded(gains_wide, stopped_early, settings_wide)
end

# Each comparison runs a Gambit trace; the mutation jobs skip it to stay under their time limit.
if get(ENV, "GAMETRACING_MUTATION", "false") != "true"
    @testset "a trace to a far target stays on Gambit's branch ($element)" for element in (Float32, widen(Float32))
        seeded = (
            far_opening_games(Xoshiro(FAR_OPENING_SEED))...,
            last(far_opening_games(Xoshiro(MIXED_STALL_SEED))),
            first(far_opening_games(Xoshiro(ZERO_STEP_SEED))),
            first(far_opening_games(Xoshiro(FAR_FOLD_SEED))),
        )
        settings = log_precision_settings(element)
        for payoffs_32 in (seeded..., far_opening_games(fresh_rng())...)
            payoffs = map(payoff -> element.(payoff), payoffs_32)
            gains = NormalFormGains(payoffs)
            failure = unwrap_error(
                trace_endpoint(
                    LogitGame(gains, element(FAR_TARGET)),
                    GameTracing.logit_count(gains),
                    settings;
                    keep_going = StopAtPrecision(element(50)),
                )
            )
            (; precision, logits) = failure.context
            @test reached_or_rounded(gains, failure, settings)
            if precision < 50
                @info "a far-target trace stopped short" failure.cause failure.context
            end
            oracle = gambit_strategies(payoffs, precision)
            gap = maximum(maximum.(abs, GameTracing.mixed_strategies(gains, logits) .- oracle.strategies))
            @test gap < 1.0f-3
        end
    end
end

@testset "the corrector holds the precision at its pin, whatever the logit correction" begin
    check = GameTracing.CorrectorReach(
        GameTracing.AcceptAll(),
        GameTracing.CorrectorSettings{Float32}(),
        TraceGeometry{Float32}(),
        FAR_TARGET,
        zeros(Float32, 3),
        zeros(Float32, 3),
        zero(Float32),
        nothing,
    )
    # Gaps near 1 at half the far target: logits near 5e5, a pin offset of 4% and 25% of a target.
    point = Float32[1.0f0, -1.0f0, 0.5f0 * FAR_TARGET]
    # A correction that moved no logit settles only at the pin, in Newton's `fx` or PALC's `res_f`.
    @test !GameTracing.settled(check, (; x = point, fx = Float32[0, 0, -0.04f0 * FAR_TARGET]), 0.0f0)
    # The pin row is the last entry of whichever residual the solver reports.
    @test GameTracing.pin_residual((; x = point, fx = Float32[0, 0, -0.04f0 * FAR_TARGET])) == -0.04f0 * FAR_TARGET
    @test GameTracing.pin_residual((; x = point, res_f = Float32[0, 0, 0.25f0 * FAR_TARGET])) == 0.25f0 * FAR_TARGET
    @test GameTracing.settled(check, (; x = point, fx = zeros(Float32, 3)), 0.0f0)
    @test !GameTracing.settled(check, (; x = point, res_f = Float32[0, 0, -0.04f0 * FAR_TARGET]), 0.0f0)
    tolerance = check.settings.logit_tolerance
    @test GameTracing.settled(check, (; x = point, res_f = zeros(Float32, 3)), tolerance)
    @test !GameTracing.settled(check, (; x = point, res_f = zeros(Float32, 3)), nextfloat(tolerance))
end

@testset "a correction within a logit's settle_ulps settles; one past it by the tolerance does not ($element)" for element in (Float32, widen(Float32))
    settings = CorrectorSettings{element}()
    tolerance, ulps = settings.logit_tolerance, settings.settle_ulps
    # A logit whose rounding alone passes the tolerance: 4 ulps of it is 256 tolerances.
    large = 64 * tolerance / eps(element)
    rounded = large + 4 * eps(large)
    @test GameTracing.logit_change(element[large, 1], element[rounded, 1]) > tolerance
    @test last(GameTracing.logit_corrections(element[large, 1], element[rounded, 1], ulps * eps(element), one(element))) <= 0
    @test first(GameTracing.logit_corrections(element[large, 1], element[rounded, 1], ulps * eps(element), one(element))) == rounded - large
    # Twice the logit's allowance leaves an excess of about one allowance, 1024 tolerances.
    beyond = large + 2 * ulps * eps(element) * large
    @test last(GameTracing.logit_corrections(element[large, 1], element[beyond, 1], ulps * eps(element), one(element))) > tolerance
    # A small logit has no allowance to speak of: twice the tolerance stays past it.
    @test last(GameTracing.logit_corrections(element[0, 1], element[2 * tolerance, 1], ulps * eps(element), one(element))) > tolerance
    # The corrector settles the large logit's rounding at its first correction.
    check = GameTracing.CorrectorReach(GameTracing.AcceptAll(), settings, TraceGeometry{element}(), one(element), element[large, 0, 1], element[large, 0, 1], one(element), nothing)
    GameTracing.restate!(check; predictor_move = one(element))
    verdicts = [GameTracing.corrected!(check, (; x = element[first, 0, 1], res_f = zeros(element, 3), step), true) for (step, first) in ((0, large), (1, rounded))]
    @test last(verdicts) isa GameTracing.BifurcationKit.NewtonConverged
    @test check.state[].correction == rounded - large
end

"""A fresh `CorrectorReach` whose current step's predictor moved the logits `predictor_move`."""
function fresh_check(predictor_move::E)::GameTracing.CorrectorReach{GameTracing.AcceptAll, E} where {E <: Real}
    check = GameTracing.CorrectorReach(
        GameTracing.AcceptAll(), CorrectorSettings{E}(), TraceGeometry{E}(), one(E), zeros(E, 3), zeros(E, 3), one(E), nothing,
    )
    GameTracing.restate!(check; predictor_move = predictor_move)
    return check
end

"""
The verdicts of `check` on one PALC solve whose iterates, at precision one and at the pin, have first logit `firsts`;
its step's arclength is a quarter.
"""
function solve_verdicts(check::GameTracing.CorrectorReach, firsts::AbstractVector{E})::Vector where {E <: Real}
    return [
        GameTracing.corrected!(check, (; x = E[first, 0, 1], res_f = zeros(E, 3), step, p = one(E), ds = one(E) / 4), true)
            for (step, first) in zip(0:length(firsts), firsts)
    ]
end

@testset "a correction that fails to contract halves the step unless it outgrows the predictor; two such are noise" begin
    # Corrections of 1e-2 then 1e-2: the second is past 0.6 of the first.
    stalled = Float32[0, 1.0f-2, 2.0f-2, 3.0f-2]
    # A predictor that moved the logits by 1 overshot: halving shrinks it, so the corrector stops and nothing is exhausted.
    poor = fresh_check(1.0f0)
    @test solve_verdicts(poor, stalled)[1:3] == [true, true, false]
    @test !GameTracing.noise_confirmed(poor)
    # It overshoots again at half the step, then the step recovers: the corrections contract and settle.
    GameTracing.restate!(poor; predictor_move = 0.5f0)
    @test solve_verdicts(poor, stalled[1:3]) == [true, true, false]
    settling = solve_verdicts(poor, Float32[0, 1.0f-2, 1.1f-2, 1.101f-2])
    @test settling[1:3] == [true, true, true]
    @test last(settling) isa GameTracing.BifurcationKit.NewtonConverged
    @test poor.state[].settled_here
    @test !GameTracing.noise_confirmed(poor)
    # A predictor that moved less than the stalled correction cannot be halved under it: the second stall is suspected noise.
    noisy = fresh_check(1.0f-3)
    @test solve_verdicts(noisy, stalled) == [true, true, true, false]
    @test !GameTracing.noise_confirmed(noisy)
    # Past a fold, halving shrinks the corrections: a stalled pair under half the suspect's confirms nothing.
    folded = fresh_check(1.0f-3)
    solve_verdicts(folded, stalled)
    GameTracing.restate!(folded; predictor_move = 1.0f-4)
    @test solve_verdicts(folded, Float32[0, 4.0f-3, 8.0f-3, 1.2f-2]) == [true, true, true, false]
    @test !GameTracing.noise_confirmed(folded)
    # A settled solve clears the suspicion.
    solve_verdicts(folded, Float32[0, 1.0f-2, 1.1f-2, 1.101f-2])
    @test isnothing(folded.state[].noise.suspect)
    # At noise the halved step stalls as before: the suspicion is confirmed.
    GameTracing.restate!(noisy; predictor_move = 5.0f-4)
    @test solve_verdicts(noisy, stalled) == [true, true, true, false]
    @test GameTracing.noise_confirmed(noisy)
    @test noisy.state[].noise.suspect.precision == 1
    @test noisy.state[].noise.suspect.correction ≈ 1.0f-2
    # The record holds the iterate whose correction confirmed the stall; a corrector without one has no record.
    @test GameTracing.exhausted_record(noisy) ==
        (; precision = 1.0f0, correction = noisy.state[].correction, stalled_iterate = Float32[3.0f-2, 0, 1], stalled_pin = 1.0f0, stalled_arclength = 0.25f0)
    @test_throws ArgumentError GameTracing.exhausted_record(poor)
    # Every later solve refuses at its start.
    @test solve_verdicts(noisy, Float32[0]) == [false]
end

@testset "a stall that halves with the step is a predictor off the tangent, not noise ($element)" for element in (Float32, widen(Float32))
    stalled = element[0, 1.0f-2, 2.0f-2, 3.0f-2]
    check = fresh_check(element(1.0f-3))
    @test solve_verdicts(check, stalled) == [true, true, true, false]
    @test !GameTracing.noise_confirmed(check)
    # Corrections of 0.5007 the suspect's, as on the 3x3x3 benchmark game of seed 5: a half passed the old 1/2 split.
    GameTracing.restate!(check; predictor_move = element(5.0f-4))
    @test solve_verdicts(check, element(0.5007f0) .* stalled) == [true, true, true, false]
    @test !GameTracing.noise_confirmed(check)
    # Three quarters is noise's side of `noise_ratio`.
    GameTracing.restate!(check; predictor_move = element(2.5f-4))
    @test solve_verdicts(check, element(0.5007f0 * 0.75f0) .* stalled) == [true, true, true, false]
    @test GameTracing.noise_confirmed(check)
end

@testset "a stall far past the noise floor halves the step at once and suspects no noise ($element)" for element in (Float32, widen(Float32))
    # Corrections of 0.45, as a corrector diverging from a long step on an IRVModel stay game: past its predictor's
    # move, but 4,500 tolerances, past `noise_limit` floors.
    diverging = element[0, 0.45f0, 0.9f0]
    check = fresh_check(element(0.19f0))
    @test solve_verdicts(check, diverging) == [true, true, false]
    @test isnothing(check.state[].noise.suspect)
    # The halved step stalls alike and confirms nothing: halving goes on until the corrector converges.
    GameTracing.restate!(check; predictor_move = element(0.095f0))
    @test solve_verdicts(check, diverging) == [true, true, false]
    @test !GameTracing.noise_confirmed(check)
    # Over settled corrections of 1e-3 the floor is 1e-3: the same stall lies within the limit, and two are suspected noise.
    floored = fresh_check(element(0.19f0))
    GameTracing.restate!(floored; noise = GameTracing.NoiseRecord(element(1.0f-3), nothing))
    @test solve_verdicts(floored, element[0, 0.45f0, 0.9f0, 1.35f0]) == [true, true, true, false]
    @test !isnothing(floored.state[].noise.suspect)
end

@testset "a pure equilibrium traced to λ 1e6 reaches it, its logits settled to their rounding ($element)" for element in (Float32, widen(Float32))
    gains = NormalFormGains(map(table -> element.(table), LOGITNASH_GAMES.prisoners_dilemma))
    settings = BifurcationKitSettings{element}(; step_metric = TraceGeometry{element}(; distance = FisherDistance, precision_term = LogPrecisionTerm))
    traced = unwrap(trace_endpoint(LogitGame(gains, element(FAR_TARGET)), GameTracing.logit_count(gains), settings))
    # Defecting gains 1 against a defector, so each logit is λ.
    @test traced.logits ≈ fill(element(FAR_TARGET), 2)
    @test traced.logit_accuracy <= settle_bound(settings.corrector, traced.logits)
    landed = only(unwrap(logit_solve_lambda(gains, element(FAR_TARGET); settings = PathFollowingSettings{element}())))
    @test landed.logits ≈ fill(element(FAR_TARGET), 2)
end

@testset "a gain with noise of size η fails PrecisionExhausted where λ η passes the logit tolerance (salt $salt)" for salt in 1:8
    # The noise starts at λ = 10, far under the predicted λ = tolerance / η = 1e3. Hashed noise from the flat start
    # stalled the opening at λ ≈ 4e-14, whatever η: zero-slope gains whose every evaluation jumps leave no first step.
    pass, target = NoisyFlatPass(1.0f-4, 1.0f-7, 10.0f0, UInt(salt)), 1.0f5
    # The opening `1 / gap_bound` is 1e4 here, so a trace can leap to the target. A corrector may stop there on a lucky
    # draw, and the endpoint's check fails it (issue #134).
    settings = BifurcationKitSettings{Float32}(; LOG_PRECISION_SETTINGS.step_metric, max_steps = 20_000)
    tolerance = settings.corrector.logit_tolerance
    failure = unwrap_error(trace_endpoint(LogitGame(pass, target), 1, settings))
    @test failure.cause == GameTracing.PrecisionExhausted
    (; lambda, correction) = failure.context
    # Corrections are at most 2 λ η, so every one settles below λ = tolerance / 2η.
    @test lambda > tolerance / (2 * pass.noise)
    @test lambda <= target
    # One draw, often under the tolerance (`unresolved_noise`); two roundings of the logit `λ level` add to it.
    @test 0 < correction <= 2 * lambda * pass.noise + 2 * eps(lambda * pass.level)
end

"""A `keep_going` that writes each accepted point of a trace into `precisions` and `logits`, and stops past `ceiling`."""
struct PointLog{E <: Real}
    ceiling::E
    precisions::Vector{E}
    logits::Matrix{E}
end

function (logbook::PointLog)(logits::AbstractVector, precision::Real)::Bool
    point = something(findfirst(isnan, logbook.precisions))
    setindex!(logbook.precisions, precision, point)
    view(logbook.logits, :, point) .= logits
    return precision < logbook.ceiling
end

@testset "the target only caps a log-precision trace: each point to 1e6 up to 100 is the endpoint of a trace to its precision" begin
    gains = NormalFormGains(first(far_opening_games(Xoshiro(FAR_OPENING_SEED))))
    width = GameTracing.logit_count(gains)
    steps = LOG_PRECISION_SETTINGS.max_steps + 1
    logbook = PointLog(100.0f0, fill(Float32(NaN32), steps), zeros(Float32, width, steps))
    trace_endpoint(LogitGame(gains, FAR_TARGET), width, LOG_PRECISION_SETTINGS; keep_going = logbook)
    recorded = 1:count(!isnan, logbook.precisions)
    for (precision, logits) in zip(view(logbook.precisions, recorded), eachcol(view(logbook.logits, :, recorded)))
        near = trace_endpoint(LogitGame(gains, precision), width, LOG_PRECISION_SETTINGS)
        @test !is_error(near)
        if !is_error(near)
            ours = GameTracing.mixed_strategies(gains, logits)
            gap = maximum(maximum.(abs, ours .- GameTracing.mixed_strategies(gains, unwrap(near).logits)))
            # The bound reads both points' own double-precision residuals.
            residual = max(residual_norm(gains, logits .* float(1), precision * float(1)), residual_norm(gains, unwrap(near).logits .* float(1), precision * float(1)))
            @test gap < 2 * probability_bound(gains, logits, precision, residual, 0.5f0)
        end
    end
    @info "accepted points of the trace to 1e6 up to precision 100" points = length(recorded)
end

"""The factored bordered tangent of a restart (`restart_tangent`), up to positive scale: the opening tangent's reference."""
function factored_opening_tangent(game, start, settings)
    width = length(start) - 1
    palc = GameTracing.palc_geometry(settings.step_metric, game.pass!, width, game.target_precision)
    direction = setindex!(zero(start), one(eltype(start)), lastindex(start))
    border = GameTracing.palc_border(palc.dot, eltype(start)(palc.palc_state_weight), direction)
    system = GameTracing.branch_system!(GameTracing.BranchWorkspace(game, start), game, start, last(start))
    return lu([@view(system.jacobian[1:width, :]); transpose(border)]) \ direction
end

@testset "at precision zero the gap Jacobian is the identity and the opening tangent is the factored tangent ($width)" for width in (2, 3, 5)
    rng = Xoshiro(0x0123456789abcdef + width)
    game = LogitGame(LinearGainPass(randn(rng, Float32, width, width), randn(rng, Float32, width)), TARGET_PRECISION)
    settings = BifurcationKitSettings{Float32}()
    start = GameTracing.trace_start(game, zeros(Float32, width), pass_workspace(game.pass!, zeros(Float32, width)))
    system = GameTracing.branch_system!(GameTracing.BranchWorkspace(game, start), game, start, last(start))
    @test system.jacobian[1:width, 1:width] == I(width)
    closed = GameTracing.opening_tangent(game, start, pass_workspace(game.pass!, zeros(Float32, width))).tangent
    factored = factored_opening_tangent(game, start, settings)
    @test closed ./ last(closed) ≈ factored ./ last(factored) atol = 1.0f-5
end

# With noise 0.1 below precision 1e-3 the secant to a solve at a 150th of the first step (the old opening) turned 1.16 rad from the branch, and
# BifurcationKit's turn test refused every step (TraceStuck, no accepted point), as on irv-extremism study draw 116.
@testset "noise in the secant's solve does not stop the trace" begin
    clean = unwrap(trace_endpoint(LogitGame(GAME, TARGET_PRECISION), WIDTH, BifurcationKitSettings{Float32}()))
    noisy = trace_endpoint(
        LogitGame(WindowNoisePass(GAME, 0.1f0, 1.0f-3), TARGET_PRECISION),
        WIDTH,
        BifurcationKitSettings{Float32}(),
    )
    @test !is_error(noisy)
    @test isapprox(unwrap(noisy).logits, clean.logits; atol = 1.0f-3)
end
