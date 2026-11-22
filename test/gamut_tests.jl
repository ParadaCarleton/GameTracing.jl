using ErrorTypes: is_error, unwrap, unwrap_error
using LogitNash: LogitNash

const GAMUT_FIXTURES = joinpath(@__DIR__, "fixtures", "gamut")

"""
Whether the Float32 failure `failure` is a rounding stop: an unresolved step or exhausted precision. These stops vary
with the CPU, so each counts as broken. A double-precision failure, or a Float32 failure of any other cause, fails.
"""
function gamut_rounding_stop(failure::TraceFailure{Float32})::Bool
    stuck = failed_as(failure, TraceCause.TraceStuck) && failure.cause.reason == GameTracing.UnresolvedStep
    return stuck || failed_as(failure, TraceCause.PrecisionExhausted)
end
gamut_rounding_stop(::TraceFailure)::Bool = false

@testset "GAMUT fixture traces" begin
    files = sort(filter(endswith(".nfg"), readdir(GAMUT_FIXTURES)))
    @test length(files) == 66
    @test length(unique(replace(file, r"-\d+\.nfg$" => "") for file in files)) == 22
    known = LogitNash.parse_nfg(read(joinpath(GAMUT_FIXTURES, "RandomGame-11.nfg")))
    @test [Float32(table[2, 1, 1]) for table in known] ≈ [-0.9808699f0, -0.64736623f0, -0.94577558f0]
    for file in files
        payoffs = LogitNash.parse_nfg(read(joinpath(GAMUT_FIXTURES, file)))
        @test length(payoffs) == 3 && all(table -> size(table) == (3, 3, 3), payoffs)
        for element in (Float32, widen(Float32)), engine in (BifurcationKitSettings, PathFollowingSettings)
            @testset "$file $engine $element" begin
                gains = NormalFormGains(map(table -> element.(table), payoffs))
                result = logit_solve(gains; settings = engine{element}())
                if !is_error(result)
                    bound = GameTracing.default_maxregret(element) * GameTracing.payoff_range(gains)
                    @test GameTracing.max_regret(gains, unwrap(result).logits) <= bound
                elseif gamut_rounding_stop(unwrap_error(result))
                    failure = unwrap_error(result)
                    cause = cause_label(failure.cause)
                    @info "GAMUT Float32 rounding stop" file engine cause lambda = failure.stop.precision
                    @test_broken false
                else
                    @test unwrap_error(result) === nothing
                end
            end
        end
    end
end
