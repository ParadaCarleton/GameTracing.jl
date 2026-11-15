# PathFollowing.jl

Trace one solution curve of `H(x) = 0`, `H: ℝⁿ⁺¹ → ℝⁿ`, by pseudo-arclength continuation.

Each step predicts along the tangent, corrects by Newton's method on the hyperplane normal to it, and sizes the
next step from the corrector and the tangent's turn (Allgower and Georg, *Numerical Continuation Methods*, §6.1).
The caller owns the loop and its stopping rule. Failures are values (ErrorTypes.jl), and a warm `step!` allocates
nothing.

```julia
using PathFollowing
using CommonSolve: init, step!
using ErrorTypes: is_error, unwrap
using LinearSolve: LUFactorization

# The unit circle: H(x, y) = x² + y² - 1, with its Jacobian, from one call.
function circle!(residual, jacobian, point)
    residual[1] = point[1]^2 + point[2]^2 - 1
    jacobian[1, 1], jacobian[1, 2] = 2point[1], 2point[2]
    return nothing
end

# `scale` turns the last correction into an estimate of the error left in `iterate`.
settled(previous, iterate, residual, scale) = scale * maximum(abs, iterate - previous) <= 1e-10

algorithm = PseudoArclength(;
    corrector = Newton(;
        max_iterations = 8, converged = settled, contraction_limit = 0.5, error_estimate = ContractionEstimate(; safety = 0.1),
    ),
    step_control = StepControl(;
        nominal_contraction = 0.25, nominal_distance = 0.1, nominal_turn = 0.1, growth_limit = 2.0, max_step = 0.5,
    ),
    metric = FixedWeights([1.0, 1.0]),
    orientation = DeterminantSign(; persistence = 4),
    linear_solver = LUFactorization(),
    max_refusals = 60,
)
tracer = unwrap(init(CurveProblem(circle!, [1.0, 0.0], [0.0, 1.0]), algorithm; first_step = 0.1))
while tracer.point[2] >= 0 || tracer.point[1] > 0   # stop past half a turn
    outcome = step!(tracer)
    is_error(outcome) && break
end
```

`land!(tracer, coordinate, value)` corrects onto the curve's point where one coordinate takes `value`, near the last
point. Use it to stop at an exact parameter value that a step passed.

## Settings

- `Newton`: the corrector's iterations, convergence test, contraction limit, error estimate and correction length.
  `ContractionEstimate(; safety)` settles on the error left in the iterate, at most `safety` times the tolerance;
  `LastCorrection()` settles on the last correction, often one iteration later.
- `StepControl`: the nominal contraction, first correction and turn, the growth limit and the largest step.
- `FixedWeights`, `PointWeights`: the step metric and the turn metric. `PointWeights` reweights at each accepted
  point.
- `linear_solver`: any LinearSolve.jl algorithm for the bordered systems.
- `orientation`: `DeterminantSign` refuses a step where `sign det([J; tᵀ])` changes. A long step near a fold can
  land where the oriented tangent runs back along the curve; this check catches it. A persistent change is accepted
  as a bifurcation crossing. `nothing` switches the check off.
- `accept`: a test on each converged point.

## Failures

`init` fails `NotFiniteStart` or `SingularStart`. `step!` fails `Unresolved` when the predictor falls below the
float type's resolution, and `TooManyRefusals` past `max_refusals`. Each failure names the last `Refusal`.
