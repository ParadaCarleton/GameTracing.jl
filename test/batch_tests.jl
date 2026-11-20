using CommonSolve: init
using ErrorTypes: Result, is_error, unwrap, unwrap_error
using JLArrays: JLArray, JLBackend
using KernelAbstractions: CPU
using PathFollowing: PathFollowing
using Random: AbstractRNG
using StaticArrays: SArray

"""`draws` random payoff tuples of scale 3 in `element` from `rng`, one table of size `counts` per player."""
function drawn_payoffs(rng::AbstractRNG, element::Type{<:Real}, counts::Tuple, draws::Integer)::Vector{<:Tuple}
    return [ntuple(_ -> element.(3 .* randn(rng, counts...)), length(counts)) for _ in 1:draws]
end

"""`logit_solve_lambda_batch` of `games` at `lam` under `settings` on the CPU backend, three work items per group."""
function cpu_batch(games::AbstractVector, lam::Real, settings::PathFollowingSettings)::Vector
    return logit_solve_lambda_batch(games, lam; settings, backend = CPU(), workgroupsize = 3)
end

"""A `ProfileResult`'s logits, or its failure's cause."""
function outcome(result::Result)::Union{AbstractVector, GameTracing.TraceCause}
    return is_error(result) ? unwrap_error(result).cause : unwrap(result).logits
end

"""The largest difference in any probability between two profiles of mixed strategies."""
function largest_strategy_gap(left::Tuple, right::Tuple)::Real
    return maximum(maximum.(abs, left .- right))
end

"""Whether two failure contexts hold one set of keys and equal values, each float within `tolerance` relative."""
function same_context(left::NamedTuple, right::NamedTuple, tolerance::Real)::Bool
    return keys(left) == keys(right) && all(close_or_equal(left[key], right[key], tolerance) for key in keys(left))
end

"""Whether two context values agree: floats within `tolerance` relative, anything else equal."""
function close_or_equal(left::Any, right::Any, ::Real)::Bool
    return isequal(left, right)
end

function close_or_equal(left::T, right::T, tolerance::Real)::Bool where {T <: Union{AbstractFloat, AbstractVector{<:AbstractFloat}}}
    return isapprox(left, right; rtol = tolerance)
end

"""The bytes a warm static `init` and `land!!` of `problem` toward `bound` allocate."""
function landing_bytes(
        problem::PathFollowing.CurveProblem,
        algorithm::PathFollowing.PseudoArclength,
        first_step::Real,
        bound::PathFollowing.Bound,
        max_steps::Integer,
    )::Integer
    PathFollowing.land!!(unwrap(init(problem, algorithm; first_step)), bound, max_steps)
    return @allocated PathFollowing.land!!(unwrap(init(problem, algorithm; first_step)), bound, max_steps)
end

# Two-action games of two and three players, and a game of three strategies each.
const BATCH_SHAPES = ((2, 2), (2, 2, 2), (3, 3))

@testset "logit_solve_lambda_batch lands each game where its sequential static trace lands ($element)" for element in (Float32, widen(Float32))
    rng = fresh_rng()
    lam = element(4)
    settings = batch_settings(element)
    for counts in BATCH_SHAPES
        games = static_game.(drawn_payoffs(rng, element, counts, 12))
        results = cpu_batch(games, lam, settings)
        inputs = GameTracing.batch_inputs(games, lam, settings)
        sequential = [
            PathFollowing.land!!(unwrap(init(problem, inputs.algorithm; first_step)), inputs.bound, settings.max_steps)
                for (problem, first_step) in zip(inputs.problems, inputs.first_steps)
        ]
        @test all(!is_error, results)
        @test isequal(outcome.(results), [collect(GameTracing.logits_at(unwrap(last(landed)))) for landed in sequential])
        @test [unwrap(result).lam for result in results] == fill(lam, length(games))
    end
end

# 2000 draws of each shape: the profiles differ by 8 eps in Float32 and 39 eps in Float64 at most, by rounding alone.
@testset "logit_solve_lambda_batch's profiles agree with logit_solve_lambda's to rounding ($element)" for element in (Float32, widen(Float32))
    rng = fresh_rng()
    lam = element(4)
    for counts in BATCH_SHAPES
        payoffs = drawn_payoffs(rng, element, counts, 12)
        results = cpu_batch(static_game.(payoffs), lam, batch_settings(element))
        @test length(results) == length(payoffs)
        for (payoff, result) in zip(payoffs, results)
            sequential = only(unwrap(logit_solve_lambda(NormalFormGains(payoff), lam; settings = PathFollowingSettings{element}())))
            @test largest_strategy_gap(unwrap(result).profile, sequential.profile) <= 256 * eps(element)
        end
    end
end

@testset "a land_kernel! launch on JLArrays lands as land_batch on Vector storage does ($element)" for element in (Float32, widen(Float32))
    rng = fresh_rng()
    settings = batch_settings(element)
    for counts in BATCH_SHAPES
        inputs = GameTracing.batch_inputs(static_game.(drawn_payoffs(rng, element, counts, 5)), element(4), settings)
        host = PathFollowing.land_batch(inputs.problems, inputs.algorithm, inputs.first_steps, inputs.bound, settings.max_steps; workgroupsize = 2)
        problems = JLArray(inputs.problems)
        device = similar(problems, eltype(host))
        launch = PathFollowing.land_kernel!(JLBackend(), 2)
        launch(device, problems, inputs.algorithm, JLArray(inputs.first_steps), inputs.bound, settings.max_steps; ndrange = length(problems))
        @test isequal(Array(device), host)
        @test all(landing -> landing.cause == PathFollowing.NoFailure, host)
    end
end

@testset "a batch's static system, metric and algorithm are isbits, and a warm static landing allocates nothing ($element)" for element in (Float32, widen(Float32))
    rng = fresh_rng()
    geometries = (GameTracing.default_step_metric(element), TraceGeometry{element}(; distance = SummedDistance))
    for counts in BATCH_SHAPES, step_metric in geometries, distance in (StrategyDistance{element}(), LogitDistance{element}())
        settings = batch_settings(element; step_metric, distance)
        inputs = GameTracing.batch_inputs(static_game.(drawn_payoffs(rng, element, counts, 1)), element(4), settings)
        problem, algorithm = only(inputs.problems), inputs.algorithm
        @test isbits(problem.system!!) && isbits(problem) && isbits(algorithm)
        @test isbits(PathFollowing.frozen_metric(algorithm.metric, problem.start))
        @test landing_bytes(problem, algorithm, only(inputs.first_steps), inputs.bound, settings.max_steps) == 0
    end
end

@testset "logit_solve_lambda_batch fails each game as logit_solve_lambda does ($element)" for element in (Float32, widen(Float32))
    rng = fresh_rng()
    lam = element(4)
    drawn = drawn_payoffs(rng, element, (2, 2), 3)
    poisoned = [(element[NaN32 0; 0 0], element[0 0; 0 0])]
    unsettled = (; newton_iterations = 1, distance = StrategyDistance{element}(; tolerance = eps(element), settle_ulps = 0), max_refusals = 2)
    # As in logit_solve_tests.jl: 2^10 / eps ulps resolve only the move off precision zero.
    unresolved = (; min_step_ulps = round(Int, 2^10 / eps(element)))
    # A normal-form start has `∂F/∂w = I`, so PathFollowing's SingularStart cannot occur.
    for (options, payoffs, cause, reason) in (
            ((; max_steps = 1), drawn, GameTracing.NoConvergence, :max_steps),
            ((;), poisoned, GameTracing.NotFinite, :NotFiniteStart),
            (unsettled, drawn, GameTracing.TraceStuck, :refusals),
            (unresolved, drawn, GameTracing.TraceStuck, :unresolved_step),
        )
        results = cpu_batch(static_game.(payoffs), lam, batch_settings(element; options...))
        @test length(results) == length(payoffs)
        for (payoff, result) in zip(payoffs, results)
            failure = unwrap_error(result)
            expected = unwrap_error(logit_solve_lambda(NormalFormGains(payoff), lam; settings = PathFollowingSettings{element}(; options...)))
            @test (failure.cause, failure.context.reason) == (expected.cause, expected.context.reason) == (cause, reason)
            @test same_context(failure.context, expected.context, 256 * eps(element))
        end
    end
end

@testset "logit_solve_lambda_batch refuses a batch its work items cannot trace (ArgumentError)" begin
    rng = fresh_rng()
    games = static_game.(drawn_payoffs(rng, Float32, (2, 2), 2))
    settings = batch_settings(Float32)
    @test_throws "StaticLU" cpu_batch(games, 1, PathFollowingSettings{Float32}())
    @test_throws "non-negative and finite" cpu_batch(games, -1, settings)
    @test_throws "non-negative and finite" cpu_batch(games, floatmax(widen(Float32)), settings)
    @test_throws "settings trace in" cpu_batch(games, 1, batch_settings(widen(Float32)))
    mixed = Vector{NormalFormGains{2, <:SArray}}([first(games), static_game(only(drawn_payoffs(rng, Float32, (3, 3), 1)))])
    @test_throws "one concrete type" cpu_batch(mixed, 1, settings)
    @test_throws "one strategy count per player" cpu_batch([static_game((zeros(Float32, 2, 3), zeros(Float32, 2, 3)))], 1, settings)
    @test isempty(cpu_batch(empty(games), 1, settings))
end
