"""
    PathFollowing

Trace one solution curve of `H(x) = 0`, `H: ℝⁿ⁺¹ → ℝⁿ`, by pseudo-arclength continuation. Each step predicts along
the curve's tangent, corrects by Newton's method on the hyperplane normal to that tangent, and sizes the next step
from the corrector and the tangent's turn; an orientation check keeps it from running back along the curve. The
caller owns the loop: `init` a `Tracer`, call `tracer, outcome = step!!(tracer)` until its own stopping rule holds. A
`Bound` stops a step on a value of one coordinate, and `land!!` steps until the point lies on it.

A trace runs on the device that holds its start. The tracer's arrays take the start's array type (`similar`), and it
works on them with broadcasts and reductions, so a GPU start keeps every step on that GPU. A static start (`SVector`)
gives static arrays throughout, which a step rebuilds instead of changing (`Into`), with no allocation. `land_batch`
traces many static problems at once, one KernelAbstractions work item each.
"""
module PathFollowing

using Accessors: @reset
using CommonSolve: CommonSolve
using ErrorTypes: Err, Ok, Option, Result, is_error, none, some, unwrap, unwrap_error
using Moshi.Data: @data
using Moshi.Derive: @derive
using GPUArraysCore: AnyGPUArray, @allowscalar
using KernelAbstractions: KernelAbstractions, @Const, @index, @kernel
using LinearAlgebra: LinearAlgebra
using LinearSolve: LinearSolve
using SciMLBase: SciMLBase
using StaticArrays: StaticArrays, SArray, SMatrix, StaticArray, StaticVector, SVector, setindex

export CurveProblem, PseudoArclength, Newton, LastCorrection, ContractionEstimate, StepControl, DeterminantSign, FixedWeights, StaticLU
export Metric, MetricDistance, HalfStepCheck, correction_bound, frozen_metric, freeze_metric!!, metric_inner, metric_apply!!
export Tracer, Workspace, AcceptedStep, StepFailure, Refusal, Bound, Into, step!!, land!!, land_batch, Landing, landing_result
export NoRefusal, NotConverged, Diverging, NotFiniteResidual, SingularSystem, AcceptRefused, TooFast, OrientationFlip, PastBound, FarFromPredictor, SplitMismatch
# For wrappers that convert their own `reach` before they build a `StepControl`.
public reach_in

include("device.jl")
include("metric.jl")
include("settings.jl")
include("workspace.jl")
include("linear.jl")
include("tracer.jl")
include("landing.jl")

end
