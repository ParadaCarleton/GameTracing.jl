"""
    CurveTracing

Trace one solution curve of `H(x) = 0`, `H: ℝⁿ⁺¹ → ℝⁿ`, by pseudo-arclength continuation. Each step predicts along
the curve's tangent, corrects by Newton's method on the hyperplane normal to that tangent, and sizes the next step
from the corrector and the tangent's turn; an orientation check keeps it from running back along the curve. The
caller owns the loop: `init` a `Tracer`, call `step!` until its own
stopping rule holds. `land!` corrects onto the point where one coordinate takes a given value, near the last point.
"""
module CurveTracing

using CommonSolve: CommonSolve
using ErrorTypes: Err, Ok, Result
using LinearAlgebra: LinearAlgebra
using LinearSolve: LinearSolve
using SciMLBase: SciMLBase

export CurveProblem, PseudoArclength, Newton, StepControl, DeterminantSign, FixedWeights, PointWeights, MetricDistance
export Tracer, AcceptedStep, StepFailure, FailureCause, Refusal, land!
export Unresolved, TooManyRefusals, NotFiniteStart, SingularStart
export NoRefusal, NotConverged, Diverging, NotFiniteResidual, SingularSystem, AcceptRefused, TooFast, OrientationFlip

include("metric.jl")
include("settings.jl")
include("tracer.jl")

end
