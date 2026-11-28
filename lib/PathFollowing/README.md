# PathFollowing.jl

Trace one solution curve of `H(x) = 0`, `H: ℝⁿ⁺¹ → ℝⁿ`, by pseudo-arclength continuation.

Each step predicts along the tangent, corrects by Newton's method on the hyperplane normal to it, and sizes the
next step from the corrector and the tangent's turn (Allgower and Georg, *Numerical Continuation Methods*, §6.1).
The caller owns the loop and its stopping rule. Failures are values (ErrorTypes.jl), and a warm `step!!` allocates
nothing.

```julia
using PathFollowing
using CommonSolve: init
using ErrorTypes: is_error, unwrap
using LinearSolve: LUFactorization

# The unit circle: H(x, y) = x² + y² - 1 and its Jacobian, written into the arrays the tracer passes and returned.
function circle!!(residual, jacobian, point)
    residual = Into(residual) .= point[1]^2 + point[2]^2 - 1
    jacobian = Into(jacobian) .= 2 .* transpose(point)
    return residual, jacobian
end

# The last correction estimates the error left in `iterate`.
settled(previous, iterate, residual) = maximum(abs.(iterate .- previous)) <= 1e-10

algorithm = PseudoArclength(;
    corrector = Newton(; max_iterations = 8, converged = settled, contraction_limit = 0.5),
    step_control = StepControl(;
        target_contraction = 0.25, target_correction = 0.1, target_turn = 0.1, max_step_growth = 2.0,
        max_step = 0.5,
    ),
    metric = FixedWeights([1.0, 1.0]),
    orientation_check = DeterminantSign(),
    split_check = HalfStepCheck{Float64}(; trigger = 1, tolerance = 1e-8),
    linear_solver = LUFactorization(),
)
tracer = unwrap(init(CurveProblem(circle!!, [1.0, 0.0], [0.0, 1.0]), algorithm; first_step = 0.1))
while tracer.point[2] >= 0 || tracer.point[1] > 0   # stop past half a turn
    global tracer, outcome = step!!(tracer)
    is_error(outcome) && break
end
```

`step!!` returns the advanced tracer. Use only that tracer: a step changes a mutable tracer's arrays in place.

`step!!(tracer, Bound(coordinate, value))` stops at an exact value of one coordinate. A predictor that reaches the
value is cut there, and Newton's method corrects at that value. A corrected point past it refuses the step as
`PastBound`, and a shorter step tries again. So the trace stops at the value's first crossing that a predictor or a
corrected point shows. A curve that passes the value and comes back within one step goes unseen; keep the turn target
small to keep steps short where the curve bends. `land!!(tracer, bound)` steps until the point lies on the
bound and returns it; `land!!(tracer, bound, max_steps)` also stops after `max_steps` steps.

## Storage and devices

The tracer's arrays take the start's type. A `Vector` start traces on the CPU, and a GPU array start keeps each step
on its GPU. A static start (`SVector`) gives static arrays throughout, with `linear_solver = StaticLU()`. A static
trace allocates nothing, and one GPU thread can run it.

`land_batch(problems, algorithm, first_steps, bound[, max_steps]; workgroupsize)` traces many static problems at once,
one KernelAbstractions work item each, on the device that holds `problems`. It returns a `Landing` per problem;
`landing_result` gives its landed point or its `StepFailure`.

A start in a workgroup's memory (`GroupVector`) runs one trace on a whole KernelAbstractions workgroup, with
`linear_solver = GroupLU()`. Each lane builds a `GroupMemory` over the group's `@localmem` buffers, with its lane
index and `GroupBarrier()`. Every lane then runs the same `init` and `step!!`. Each array write splits its entries
over the lanes between two barriers. Write a group array through `Into`, `fill!` or `copyto!`; it has no `setindex!`,
and an `Into` source must not overlap it. `GroupLU` gives lane `j` column `j` of each elimination. Group memory never
frees an array: call `reset_memory!` before each trace. On the host or the CPU backend, use one lane and `LoneLane()`,
because that backend runs no barrier outside a kernel's own body.
On CUDA, launch the kernel on `CUDABackend(; always_inline = true)`. On an RTX 5050 with GPUCompiler 2.10, inlined
device calls cut the tracer's local memory from about 80 KB to 1 to 4 KB per lane, and a correction ran 29 times faster at
10 coordinates and 15 times faster at 16.

```julia
using StaticArrays: SVector

struct Circle
    radius::Float32
end
(circle::Circle)(residual, jacobian, point) =
    (Into(residual) .= sum(abs2, point) - circle.radius^2, Into(jacobian) .= 2 .* transpose(point))

# Every part of a batch algorithm must be isbits: a callable struct, static weights, StaticLU.
struct Settled
    tolerance::Float32
end
(settled::Settled)(previous, iterate, residual) = maximum(abs.(iterate .- previous)) <= settled.tolerance

static_algorithm = PseudoArclength(;
    corrector = Newton(; max_iterations = 8, converged = Settled(1.0f-5), contraction_limit = 0.5f0),
    step_control = StepControl(;
        target_contraction = 0.25f0, target_correction = 0.1f0, target_turn = 0.1f0, max_step_growth = 2.0f0, max_step = 0.5f0,
    ),
    metric = FixedWeights(SVector(1.0f0, 1.0f0)),
    orientation_check = DeterminantSign(),
    split_check = nothing,
    linear_solver = StaticLU(),
)
radii = range(1.0f0, 3.0f0; length = 1000)
problems = [CurveProblem(Circle(radius), SVector(radius, 0.0f0), SVector(0.0f0, 1.0f0)) for radius in radii]
# `CuArray(problems)` and `CuArray(first_steps)` trace on an NVIDIA GPU.
landings = land_batch(problems, static_algorithm, fill(0.1f0, 1000), Bound(2, 0.5f0); workgroupsize = 16)
points = unwrap.(landing_result.(landings))
```

## Settings

- `Newton`: the corrector's iterations, convergence test, contraction limit and correction length. It settles where
  its convergence test passes the last correction. The contraction limit lies in `(0, 1/2]`: where later corrections
  would contract as fast, the last correction bounds the error left in its iterate.
- `StepControl`: the target contraction, first correction and turn, the largest growth (and the largest after the
  first step) and the largest step. A converged point farther than `reach` predictor lengths from its predictor
  refuses the step (`FarFromPredictor`): the corrector can land on another branch there. A move within the
  corrector's own error estimates, in its `distance`, passes. A target first correction needs a distance that declares
  its largest length (`correction_bound`), and its refusal must lie under that length.
- `metric`, `turn_metric`: the step metric and the turn metric, each a symmetric positive semidefinite operator
  frozen at every accepted point (`frozen_metric`). `FixedWeights` is the constant diagonal one.
- `linear_solver`: any LinearSolve.jl algorithm for the bordered systems of a mutable start, `StaticLU()` for a
  static start of at most 14 coordinates, or `GroupLU()` for a start in a workgroup's memory.
- `orientation_check`: `DeterminantSign` refuses a step where `sign det([J; tᵀ])` changes. A long step near a fold can
  land where the oriented tangent runs back along the curve; this check catches it. At a bifurcation the trace closes
  in on the crossing and fails `Unresolved`. `nothing` switches the check off.
- `split_check`: `HalfStepCheck(; trigger, tolerance)` runs a step with factor at least `trigger` again in two half
  steps. The second ends on the step's own hyperplane and must reach the corrected point within `tolerance`, or the
  step is refused (`SplitMismatch`). A corrector can converge on another branch that passes its predictor, with every
  step measurement in range; the shorter half steps keep the start's branch. `nothing` switches the check off.
- `accept`: a test on each converged point.

## Failures

`init` fails `NotFiniteStart` or `SingularStart`. `step!!` fails `Unresolved` when the predictor falls below the
float type's resolution or a retry rounds back to the refused step. Each refusal shrinks the step, so a step that
never converges ends there. `TooManyRefusals` fails a step past a set `max_refusals`, and `TooManySteps` a `land!!`
past a set step budget; both default to `nothing`, no limit. Each failure names the last `Refusal`.
