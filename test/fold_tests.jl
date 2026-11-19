using BifurcationKit: BifurcationKit
using ErrorTypes: is_error, unwrap
const FOLD_SETTINGS = BifurcationKitSettings{Float32}()

@testset "a trace crosses a fold of the branch" begin
    # At target 2 the game has one equilibrium, past a fold at precision about 1.4: only continuation through the fold reaches it.
    target = 2.0f0
    result = trace_endpoint(LogitGame(FOLD_GAME, target), 1, FOLD_SETTINGS)
    @test !is_error(result)
    @test residual_norm(FOLD_GAME, unwrap(result).logits, target) < 1.0f-3
    @test only(unwrap(result).logits) > 3
    @testset "every tangent and solver choice crosses it" begin
        for settings in (
                BifurcationKitSettings{Float32}(tangent = BifurcationKit.Secant()),
                BifurcationKitSettings{Float32}(
                    bordered_solver = BorderingSolve,
                    orientation_check = false,
                ),
                BifurcationKitSettings{Float32}(step_control = nothing),
                # Past the second fold this arm's Fisher-weighted tangent once turned back (`PlainOrientation`).
                BifurcationKitSettings{Float32}(
                    bordered_solver = BorderingSolve,
                    orientation_check = false,
                    corrector = CorrectorSettings{Float32}(; logit_tolerance = 1.0f-5),
                ),
            )
            traced = trace_endpoint(LogitGame(FOLD_GAME, target), 1, settings)
            @test !is_error(traced)
            @test only(unwrap(traced).logits) ≈ only(unwrap(result).logits) rtol = 1.0f-3
        end
    end
end

@testset "a millionth below the fold, the endpoint passes its pinned check ($element)" for element in (Float32, widen(Float32))
    # The lower branch folds where `z = 1 + exp(z - 3)`, at λ 1.4106649; the pinned Jacobian nearly vanishes below it.
    # Measured: the check's correction is at most 2.1e-5 in Float32 and 6.1e-6 in Float64, from 1e-1 to 1e-7 below.
    target = element(big"1.4106635258075186")
    settings = BifurcationKitSettings{element}()
    result = trace_endpoint(LogitGame(ThresholdPass(element(6), element(3)), target), 1, settings)
    @test !is_error(result)
    @test unwrap(result).logit_accuracy <= settle_bound(settings.corrector, unwrap(result).logits)
end

@testset "below the fold the trace stays on the lower branch" begin
    # Precision 1 has three equilibria; the branch connected to the flat start is the lowest.
    result = trace_endpoint(LogitGame(FOLD_GAME, 1.0f0), 1, FOLD_SETTINGS)
    @test !is_error(result)
    @test residual_norm(FOLD_GAME, unwrap(result).logits, 1.0f0) < 1.0f-3
    @test only(unwrap(result).logits) < 2
end
