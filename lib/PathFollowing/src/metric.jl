"""
    Metric

A step or turn metric: a symmetric positive semidefinite operator `M` on displacements of the curve's points, frozen
at a point. The tracer freezes each metric at its start and again at every accepted point, and the step after a point
uses `M` frozen there. A metric type subtypes `Metric` and defines four functions:

- `frozen_metric(metric, point)`: a fresh state of `metric` in `point`'s element type, frozen at `point`.
- `freeze_metric!(frozen, point)`: freeze that state again at `point`, and return it.
- `metric_inner(frozen, left, right)`: `leftᵀ M right`.
- `metric_apply!(out, frozen, vector)`: write `M vector` into `out`, an array apart from `vector`, and return it.

`FixedWeights` is the constant diagonal metric.
"""
abstract type Metric end

"""`metric` (`Metric`) in `point`'s element type, frozen at `point`."""
function frozen_metric end

"""`frozen`, a metric's state (`frozen_metric`), frozen again at `point`."""
function freeze_metric! end

"""`leftᵀ M right` under the frozen metric `frozen` (`frozen_metric`)."""
function metric_inner end

"""`out`, filled with `M vector` under the frozen metric `frozen` (`frozen_metric`); `out` is an array apart from `vector`."""
function metric_apply! end

"""
    FixedWeights(weights)

The constant diagonal metric `⟨a, b⟩ = Σᵢ wᵢ aᵢ bᵢ`, one positive weight per coordinate of the curve's points.
"""
struct FixedWeights{V <: AbstractVector{<:Real}} <: Metric
    weights::V
end

function frozen_metric(metric::FixedWeights, point::AbstractVector{E})::FixedWeights{Vector{E}} where {E <: Real}
    if length(metric.weights) != length(point)
        throw(DimensionMismatch("$(length(metric.weights)) weights for points of $(length(point)) coordinates"))
    end
    return FixedWeights(Vector{E}(metric.weights))
end

freeze_metric!(frozen::FixedWeights, ::AbstractVector)::FixedWeights = frozen

function metric_inner(frozen::FixedWeights, left::AbstractVector{E}, right::AbstractVector{E})::E where {E <: Real}
    total = zero(E)
    for index in eachindex(frozen.weights, left, right)
        total += frozen.weights[index] * left[index] * right[index]
    end
    return total
end

function metric_apply!(out::AbstractVector{E}, frozen::FixedWeights, vector::AbstractVector{E})::AbstractVector{E} where {E <: Real}
    out .= frozen.weights .* vector
    return out
end

"""`√⟨vector, vector⟩` under the frozen metric `frozen`."""
function metric_norm(frozen::Any, vector::AbstractVector{E})::E where {E <: Real}
    return sqrt(metric_inner(frozen, vector, vector))
end

"""`√⟨to - from, to - from⟩` under the frozen metric `frozen`, the difference written into `scratch`."""
function metric_distance(frozen::Any, scratch::AbstractVector{E}, from::AbstractVector{E}, to::AbstractVector{E})::E where {E <: Real}
    scratch .= to .- from
    return metric_norm(frozen, scratch)
end

"""`row ⋅ (to - from)`, without a difference vector."""
function row_offset(row::AbstractVector{E}, from::AbstractVector{E}, to::AbstractVector{E})::E where {E <: Real}
    total = zero(E)
    for index in eachindex(row, from, to)
        total += row[index] * (to[index] - from[index])
    end
    return total
end

"""
The angle in radians between `left` and `right` under the frozen metric `frozen`: `2 atan(‖u - v‖, ‖u + v‖)` for
their unit vectors `u` and `v` (Kahan), accurate at every angle. `scratch` holds `u - v`, then `u + v`. An `acos` of
the cosine loses half the digits of a small angle.
"""
function metric_angle(frozen::Any, scratch::AbstractVector{E}, left::AbstractVector{E}, right::AbstractVector{E})::E where {E <: Real}
    left_norm, right_norm = metric_norm(frozen, left), metric_norm(frozen, right)
    scratch .= left ./ left_norm .- right ./ right_norm
    difference = metric_norm(frozen, scratch)
    scratch .= left ./ left_norm .+ right ./ right_norm
    return 2 * atan(difference, metric_norm(frozen, scratch))
end
