"""
    PathFollowing

Trace one solution curve of `H(x) = 0`, `H: ℝⁿ⁺¹ → ℝⁿ`, by pseudo-arclength continuation. Each step predicts along
the curve's tangent, corrects by Newton's method on the hyperplane normal to that tangent, and sizes the next step
from the corrector and the tangent's turn; an orientation check keeps it from running back along the curve. The
caller owns the loop: `init` a `Tracer`, call `step!` until its own stopping rule holds. `land!` corrects onto the
point near the last step where one coordinate takes a given value.
"""
module PathFollowing

using CommonSolve: CommonSolve
using ErrorTypes: Err, Ok, Option, Result, is_error, none, some, unwrap, unwrap_error
using LinearAlgebra: LinearAlgebra
using LinearSolve: LinearSolve
using SciMLBase: SciMLBase

export CurveProblem, PseudoArclength, Newton, LastCorrection, ContractionEstimate, StepControl, DeterminantSign, FixedWeights
export Metric, MetricDistance, frozen_metric, freeze_metric!, metric_inner, metric_apply!
export Tracer, AcceptedStep, StepFailure, FailureCause, Refusal, land!
export Unresolved, TooManyRefusals, NotFiniteStart, SingularStart
export NoRefusal, NotConverged, Diverging, NotFiniteResidual, SingularSystem, AcceptRefused, TooFast, OrientationFlip, NoCrossing

include("metric.jl")
include("settings.jl")
include("tracer.jl")

end
