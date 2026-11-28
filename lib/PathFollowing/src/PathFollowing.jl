"""
    PathFollowing

Trace one solution curve of `H(x) = 0`, `H: ℝⁿ⁺¹ → ℝⁿ`, by pseudo-arclength continuation. Each step predicts along
the curve's tangent, corrects by Newton's method on the hyperplane normal to that tangent, and sizes the next step
from the corrector and the tangent's turn; an orientation check keeps it from running back along the curve. The
caller owns the loop: `init` a `Tracer`, call `tracer, outcome = step!!(tracer)` until its own stopping rule holds. A
`Bound` stops a step on a value of one coordinate, and `land!!` steps until the point lies on it.

A trace runs on the device that holds its start. The tracer's arrays take the start's array type (`similar`), and it
works on them with broadcasts and reductions, so a GPU start keeps every step on that GPU. A static start (`SVector`)
gives static arrays throughout, which a step rebuilds instead of changing (`@into!!`), with no allocation. `land_batch`
traces many static problems at once, one KernelAbstractions work item each. A start in a workgroup's memory
(`GroupVector`) gives arrays that every lane of the group holds, so one trace runs on a whole workgroup. Its tracer
stores `GroupHandle`s, and each step takes the `GroupMemory` beside it: `step!!(memory, tracer)`.
"""
module PathFollowing

using Accessors: @reset
using Base.Checked: checked_mul
using ConstructionBase: getfields
using CommonSolve: CommonSolve
using ErrorTypes: Err, Ok, Option, Result, is_error, none, some, unwrap, unwrap_error
using Moshi.Data: @data
using Moshi.Derive: @derive
using GPUArraysCore: AnyGPUArray, @allowscalar
using KernelAbstractions: KernelAbstractions, @Const, @index, @kernel, @synchronize
using LazyBroadcast: lazy_broadcast
using LinearAlgebra: LinearAlgebra
using LinearSolve: LinearSolve
using SciMLBase: SciMLBase
using StaticArrays: StaticArrays, SArray, SMatrix, StaticArray, StaticVector, SVector, setindex

export CurveProblem, PseudoArclength, Newton, StepControl, DeterminantSign, FixedWeights, StaticLU
export Metric, MetricDistance, HalfStepCheck, correction_bound, frozen_metric, freeze_metric!!, metric_inner, metric_apply!!
export GroupMemory, GroupArray, GroupVector, GroupMatrix, GroupHandle, GroupLU, GroupBarrier, LoneLane, group_array, group_view, reset_memory!
export Tracer, Workspace, AcceptedStep, StepFailure, Refusal, Bound, @into!!, step!!, land!!, land_batch, Landing, landing_result
export NoRefusal, NotConverged, Diverging, NotFiniteResidual, SingularSystem, AcceptRefused, TooFast, OrientationFlip, PastBound, FarFromPredictor, SplitMismatch
# For wrappers that convert their own `reach` and contraction limit before they build a `StepControl` and a `Newton`.
public reach_in, contraction_limit_in
# For a metric whose state lies in group memory: it defines both for that state.
public stored
# For wrappers that share its optional budgets.
public checked_limit, limit_reached

include("device.jl")
include("handle.jl")
include("metric.jl")
include("settings.jl")
include("workspace.jl")
include("group.jl")
include("linear.jl")
include("tracer.jl")
include("landing.jl")

end
