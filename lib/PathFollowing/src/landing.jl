"""
    tracer, landed = land!!(tracer, bound, max_steps)

Step `tracer` toward `bound` (`step!!`) until its point lies on it. Return the advanced tracer and a copy of the landed
point, or the `StepFailure` that ends the trace: a failed step's, or `TooManySteps` once the tracer has taken
`max_steps` accepted steps since `init`. A point on the bound lands at once. Use only the returned tracer.
"""
function land!!(
        tracer::T,
        bound::Bound{E},
        max_steps::Integer,
    )::Tuple{T, Result{V, StepFailure{E}}} where {E <: Real, V <: AbstractVector{E}, T <: Tracer{E, V}}
    while !on_bound(bound, tracer.point)
        tracer.steps < max_steps || return tracer, Err(StepFailure(TooManySteps, NoRefusal, tracer.step_length, 0))
        tracer, stepped = step!!(tracer, bound)
        is_error(stepped) && return tracer, Err(unwrap_error(stepped))
    end
    return tracer, Ok(copy(tracer.point))
end

"""
    Landing(point, steps, cause, last_refusal, step_length, refusals)

How a `land_batch` trace ended, as an isbits record that a device array holds: its last `point`, its accepted `steps`,
and its `StepFailure`'s fields. Cause `NoFailure` marks a landing on the bound. `landing_result` gives the `Result`.
"""
struct Landing{V <: StaticVector, E <: Real}
    point::V
    steps::Int
    cause::FailureCause
    last_refusal::Refusal
    step_length::E
    refusals::Int
end

"""`landing`'s landed point, or the `StepFailure` that ended its trace."""
function landing_result(landing::Landing{V, E})::Result{V, StepFailure{E}} where {E <: Real, V <: StaticVector{<:Any, E}}
    landing.cause == NoFailure && return Ok(landing.point)
    return Err(StepFailure(landing.cause, landing.last_refusal, landing.step_length, landing.refusals))
end

"""
    land_batch(problems, algorithm, first_steps, bound, max_steps; workgroupsize)

Trace each of `problems` from its start to `bound`, one KernelAbstractions work item each, on the device that holds
`problems`: `init` with its first step from `first_steps`, then `land!!`. Return a `Landing` for each, in an array
on that device, once the device has finished.

Each start must be static (an `SVector`), with `linear_solver = StaticLU()`, and `algorithm` and the problems isbits.
`workgroupsize` is the work items per group: the CPU backend runs each group on one thread, so use a small group there.
"""
function land_batch(
        problems::AbstractVector{<:CurveProblem{<:Any, V}},
        algorithm::PseudoArclength,
        first_steps::AbstractVector{E},
        bound::Bound{E},
        max_steps::Integer;
        workgroupsize::Integer,
    )::AbstractVector{Landing{V, E}} where {E <: Real, V <: SVector{<:Any, E}}
    count = length(problems)
    length(first_steps) == count || throw(DimensionMismatch(lazy"first_steps holds $(length(first_steps)) steps for $count problems"))
    algorithm.linear_solver isa StaticLU || throw(ArgumentError(lazy"land_batch needs linear_solver = StaticLU(); got $(algorithm.linear_solver)"))
    isbits(algorithm) || throw(ArgumentError(lazy"land_batch needs an isbits algorithm; got a $(typeof(algorithm))"))
    workgroupsize > 0 || throw(ArgumentError(lazy"workgroupsize must be positive; got $workgroupsize"))
    landings = similar(problems, Landing{V, E})
    backend = KernelAbstractions.get_backend(problems)
    land_kernel!(backend, Int(workgroupsize))(landings, problems, algorithm, first_steps, bound, Int(max_steps); ndrange = count)
    KernelAbstractions.synchronize(backend)
    return landings
end

@kernel function land_kernel!(landings, @Const(problems), algorithm, @Const(first_steps), bound, max_steps)
    index = @index(Global, Linear)
    landings[index] = landed(problems[index], algorithm, first_steps[index], bound, max_steps)
end

"""`problem` traced from its start to `bound` (`land!!`), as a `Landing`."""
function landed(
        problem::CurveProblem{<:Any, V},
        algorithm::PseudoArclength,
        first_step::E,
        bound::Bound{E},
        max_steps::Int,
    )::Landing{V, E} where {E <: Real, V <: SVector{<:Any, E}}
    started = CommonSolve.init(problem, algorithm; first_step)
    if is_error(started)
        failure = unwrap_error(started)
        return Landing(problem.start, 0, failure.cause, failure.last_refusal, failure.step_length, failure.refusals)
    end
    tracer, outcome = land!!(unwrap(started), bound, max_steps)
    is_error(outcome) || return Landing(tracer.point, tracer.steps, NoFailure, NoRefusal, tracer.step_length, 0)
    failure = unwrap_error(outcome)
    return Landing(tracer.point, tracer.steps, failure.cause, failure.last_refusal, failure.step_length, failure.refusals)
end
