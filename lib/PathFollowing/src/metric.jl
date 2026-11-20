"""
    Metric

A step or turn metric: a symmetric positive semidefinite operator `M` on displacements of the curve's points, frozen
at a point. The tracer freezes each metric at its start and again at every accepted point, and the step after a point
uses `M` frozen there. A metric type subtypes `Metric` and defines four functions:

- `frozen_metric(metric, point)`: a fresh state of `metric` in `point`'s element type, frozen at `point`.
- `freeze_metric!!(frozen, point)`: that state frozen again at `point`.
- `metric_inner(frozen, left, right)`: `leftᵀ M right`.
- `metric_apply!!(out, frozen, vector)`: `M vector` in `out`, an array apart from `vector`.

The two `!!` functions follow `Into`: a mutable state or array changes in place and comes back itself, and a static
one comes back new. The tracer uses only the returned value. `FixedWeights` is the constant diagonal metric.
"""
abstract type Metric end

"""`metric` (`Metric`) in `point`'s element type, frozen at `point`."""
function frozen_metric end

"""`frozen`, a metric's state (`frozen_metric`), frozen again at `point`: in place, or a new state."""
function freeze_metric!! end

"""`leftᵀ M right` under the frozen metric `frozen` (`frozen_metric`)."""
function metric_inner end

"""
`M vector` under the frozen metric `frozen` (`frozen_metric`), in `out` (`Into`); `out` is an array apart from `vector`.
"""
function metric_apply!! end

"""
    FixedWeights(weights)

The constant diagonal metric `⟨a, b⟩ = Σᵢ wᵢ aᵢ bᵢ`, one positive weight per coordinate of the curve's points.
"""
struct FixedWeights{V <: AbstractVector{<:Real}} <: Metric
    weights::V
end

"""`metric`'s weights copied into an array like `point`: its element type, on its device, or static for a static point."""
function frozen_metric(metric::FixedWeights, point::AbstractVector{E})::FixedWeights{<:AbstractVector{E}} where {E <: Real}
    matching_weights(metric, point)
    # Through a host `Vector`: a range or a view would copy into a GPU array one element at a time.
    return FixedWeights(copyto!(similar(point), Vector{E}(metric.weights)))
end

function frozen_metric(metric::FixedWeights, point::StaticVector{N, E})::FixedWeights{SVector{N, E}} where {N, E <: Real}
    matching_weights(metric, point)
    return FixedWeights(SVector{N, E}(metric.weights))
end

"""Throw unless `metric` has one weight per coordinate of `point`."""
function matching_weights(metric::FixedWeights, point::AbstractVector)::Nothing
    length(metric.weights) == length(point) ||
        throw(DimensionMismatch("$(length(metric.weights)) weights for points of $(length(point)) coordinates"))
    return nothing
end

function freeze_metric!!(frozen::F, ::AbstractVector)::F where {F <: FixedWeights}
    return frozen
end

function metric_inner(frozen::FixedWeights, left::AbstractVector{E}, right::AbstractVector{E})::E where {E <: Real}
    return sum(elementwise(*, frozen.weights, left, right))
end

function metric_apply!!(out::V, frozen::FixedWeights, vector::AbstractVector{E})::V where {E <: Real, V <: AbstractVector{E}}
    return Into(out) .= frozen.weights .* vector
end

"""`√⟨vector, vector⟩` under the frozen metric `frozen`."""
function metric_norm(frozen::Any, vector::AbstractVector{E})::E where {E <: Real}
    return sqrt(metric_inner(frozen, vector, vector))
end

"""`√⟨to - from, to - from⟩` under the frozen metric `frozen`, the difference written into `scratch` (`Into`)."""
function metric_distance(frozen::Any, scratch::AbstractVector{E}, from::AbstractVector{E}, to::AbstractVector{E})::E where {E <: Real}
    return metric_norm(frozen, Into(scratch) .= to .- from)
end

"""`row ⋅ (to - from)`, without a difference vector."""
function row_offset(row::AbstractVector{E}, from::AbstractVector{E}, to::AbstractVector{E})::E where {E <: Real}
    return sum(elementwise((weight, start, finish) -> weight * (finish - start), row, from, to))
end

"""
The angle in radians between `left` and `right` under the frozen metric `frozen`: `2 atan(‖u - v‖, ‖u + v‖)` for
their unit vectors `u` and `v` (Kahan), accurate at every angle. `scratch` holds `u - v`, then `u + v` (`Into`). An
`acos` of the cosine loses half the digits of a small angle.
"""
function metric_angle(frozen::Any, scratch::AbstractVector{E}, left::AbstractVector{E}, right::AbstractVector{E})::E where {E <: Real}
    left_norm, right_norm = metric_norm(frozen, left), metric_norm(frozen, right)
    difference = metric_norm(frozen, Into(scratch) .= left ./ left_norm .- right ./ right_norm)
    return 2 * atan(difference, metric_norm(frozen, Into(scratch) .= left ./ left_norm .+ right ./ right_norm))
end
