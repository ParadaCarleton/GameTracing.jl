"""
    tracer, landed = land!!(tracer, bound[, max_steps])

Step `tracer` toward `bound` (`step!!`) until its point lies on it. Return the advanced tracer and a copy of the landed
point, or the `StepFailure` that ends the trace: a failed step's, or `TooManySteps` once the tracer has taken
`max_steps` accepted steps since `init`. `max_steps` defaults to `nothing`, no budget: each step either moves the point
past its rounding or fails. A closed curve that never meets the bound steps around it forever, so pass a budget for
one. A point on the bound lands at once. Use only the returned tracer.
"""
function land!!(
        tracer::T,
        bound::Bound{E},
        max_steps::Union{Nothing, Integer} = nothing,
    )::Tuple{T, Result{V, StepFailure.Type{E}}} where {E <: Real, V <: AbstractVector{E}, T <: Tracer{E, V}}
    while !on_bound(bound, tracer.point)
        limit_reached(max_steps, tracer.steps) && return tracer, Err(StepFailure.TooManySteps{E}())
        tracer, stepped = step!!(tracer, bound)
        is_error(stepped) && return tracer, Err(unwrap_error(stepped))
    end
    return tracer, Ok(copy(tracer.point))
end

"""
    Landing(point, steps, step_length, failure)

How a `land_batch` trace ended, as a record that a device array holds inline: its last `point`, its accepted `steps`,
the step its tracer held (zero after a failure at the start), and `some(StepFailure)`, or `none` for a landing on the
bound. A Moshi sum is no isbits type, but it is stored inline with no pointer (`Base.allocatedinline`), so a device
array holds a `Landing` too. `landing_result` gives the `Result`.
"""
struct Landing{V <: StaticVector, E <: Real}
    point::V
    steps::Int
    step_length::E
    failure::Option{StepFailure.Type{E}}
end

"""`landing`'s landed point, or the `StepFailure` that ended its trace."""
function landing_result(landing::Landing{V, E})::Result{V, StepFailure.Type{E}} where {E <: Real, V <: StaticVector{<:Any, E}}
    # A `none` failure is a landing.
    is_error(landing.failure) && return Ok(landing.point)
    return Err(unwrap(landing.failure))
end

"""
    land_batch(problems, algorithm, first_steps, bound[, max_steps]; workgroupsize)

Trace each of `problems` from its start to `bound`, one KernelAbstractions work item each, on the device that holds
`problems`: `init` with its first step from `first_steps`, then `land!!` with its `max_steps`. Return a `Landing` for each, in an array
on that device, once the device has finished.

Each start must be static (an `SVector`), with `linear_solver = StaticLU()`, and `algorithm` and the problems isbits.
`workgroupsize` is the work items per group: the CPU backend runs each group on one thread, so use a small group there.

An exception in a work item aborts the whole launch, so the host checks the inputs first (`check_batch`). A device
cannot catch an exception, so a callback that throws in a work item still aborts the batch: `system!!`, `converged`,
`distance`, `accept` or a metric. So does a `system!!` that returns other array types at another point. The backend
must define `KernelAbstractions.synchronize`; JLArrays 0.3 does not.
"""
function land_batch(
        problems::AbstractVector{<:CurveProblem{<:Any, V}},
        algorithm::PseudoArclength,
        first_steps::AbstractVector{E},
        bound::Bound{E},
        max_steps::Union{Nothing, Integer} = nothing;
        workgroupsize::Integer,
    )::AbstractVector{Landing{V, E}} where {E <: Real, V <: SVector{<:Any, E}}
    check_batch(problems, algorithm, first_steps, bound, workgroupsize)
    budget = checked_limit(:max_steps, max_steps)
    landings = similar(problems, Landing{V, E})
    isempty(problems) && return landings
    backend = KernelAbstractions.get_backend(problems)
    land_kernel!(backend, Int(workgroupsize))(landings, problems, algorithm, first_steps, bound, budget; ndrange = length(problems))
    KernelAbstractions.synchronize(backend)
    return landings
end

"""
Throw on the host for any input of `land_batch` that would throw in a work item. `init` runs on the first problem,
which checks the settings' types and the metrics' and the solver's sizes for every problem, as all share one type.
"""
function check_batch(
        problems::AbstractVector{<:CurveProblem{<:Any, V}},
        algorithm::PseudoArclength,
        first_steps::AbstractVector{E},
        bound::Bound{E},
        workgroupsize::Integer,
    )::Nothing where {E <: Real, V <: SVector{<:Any, E}}
    count = length(problems)
    length(first_steps) == count || throw(DimensionMismatch(lazy"first_steps holds $(length(first_steps)) steps for $count problems"))
    isbits(algorithm) || throw(ArgumentError(lazy"land_batch needs an isbits algorithm; got a $(typeof(algorithm))"))
    workgroupsize > 0 || throw(ArgumentError(lazy"workgroupsize must be positive; got $workgroupsize"))
    1 <= bound.coordinate <= length(V) || throw(ArgumentError(lazy"the bound's coordinate must lie in 1:$(length(V)); got $(bound.coordinate)"))
    backend = KernelAbstractions.get_backend(problems)
    KernelAbstractions.get_backend(first_steps) == backend ||
        throw(ArgumentError(lazy"first_steps must lie on the problems' device, $backend; got $(KernelAbstractions.get_backend(first_steps))"))
    count == 0 && return nothing
    # One reduction on the device.
    mapreduce(valid_first_step, &, first_steps; init = true) || throw(ArgumentError("every first step must be positive and finite"))
    CommonSolve.init(element(problems, 1), algorithm; first_step = element(first_steps, 1))
    # Checked before the launch, so a backend without it fails before the work.
    applicable(KernelAbstractions.synchronize, backend) || throw(ArgumentError(lazy"$backend defines no KernelAbstractions.synchronize"))
    return nothing
end

# KernelAbstractions 0.9's @kernel drops these source lines, so no run counts them (fixed in 0.10, JuliaGPU/KernelAbstractions.jl#734).
# The land_batch testsets run this kernel; a test fails once the lines survive, to remove these markers.
# COV_EXCL_START
@kernel function land_kernel!(landings, @Const(problems), algorithm, @Const(first_steps), bound, max_steps)
    index = @index(Global, Linear)
    landings[index] = landed(problems[index], algorithm, first_steps[index], bound, max_steps)
end
# COV_EXCL_STOP

"""`problem` traced from its start to `bound` (`land!!`), as a `Landing`."""
function landed(
        problem::CurveProblem{<:Any, V},
        algorithm::PseudoArclength,
        first_step::E,
        bound::Bound{E},
        max_steps::Union{Nothing, Int},
    )::Landing{V, E} where {E <: Real, V <: SVector{<:Any, E}}
    started = CommonSolve.init(problem, algorithm; first_step)
    # A trace that fails at its start holds no step.
    is_error(started) && return Landing(problem.start, 0, zero(E), some(unwrap_error(started)))
    tracer, outcome = land!!(unwrap(started), bound, max_steps)
    is_error(outcome) || return Landing(tracer.point, tracer.steps, tracer.step_length, none(StepFailure.Type{E}))
    return Landing(tracer.point, tracer.steps, tracer.step_length, some(unwrap_error(outcome)))
end
