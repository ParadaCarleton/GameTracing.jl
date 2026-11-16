"""
    FixedWeights(weights)

The inner product `⟨a, b⟩ = Σᵢ wᵢ aᵢ bᵢ` with constant positive `weights`, one per coordinate of the curve's points.
"""
struct FixedWeights{V <: AbstractVector{<:Real}}
    weights::V
end

"""
    PointWeights(weights!)

The inner product `⟨a, b⟩ = Σᵢ wᵢ aᵢ bᵢ` with weights `weights!(weights, point)` writes at each accepted point. The
weights stay frozen for the step after that point.
"""
struct PointWeights{F}
    weights!::F
end

"""Write the weights of `metric` at `point` into `weights`."""
function reweight!(weights::AbstractVector{E}, metric::FixedWeights, ::AbstractVector{E})::AbstractVector{E} where {E <: Real}
    return copyto!(weights, metric.weights)
end

function reweight!(weights::AbstractVector{E}, metric::PointWeights, point::AbstractVector{E})::AbstractVector{E} where {E <: Real}
    metric.weights!(weights, point)
    return weights
end

"""`Σᵢ wᵢ aᵢ bᵢ` over `weights`, `left` and `right`."""
function weighted_dot(weights::AbstractVector{E}, left::AbstractVector{E}, right::AbstractVector{E})::E where {E <: Real}
    total = zero(E)
    for index in eachindex(weights, left, right)
        total += weights[index] * left[index] * right[index]
    end
    return total
end

"""`√⟨vector, vector⟩` over `weights`."""
function weighted_norm(weights::AbstractVector{E}, vector::AbstractVector{E})::E where {E <: Real}
    return sqrt(weighted_dot(weights, vector, vector))
end

"""`√⟨to - from, to - from⟩` over `weights`, without a difference vector."""
function weighted_distance(weights::AbstractVector{E}, from::AbstractVector{E}, to::AbstractVector{E})::E where {E <: Real}
    total = zero(E)
    for index in eachindex(weights, from, to)
        total += weights[index] * (to[index] - from[index])^2
    end
    return sqrt(total)
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
The angle in radians between `left` and `right` under `weights`: `2 atan(‖u - v‖, ‖u + v‖)` for their unit vectors
`u` and `v` (Kahan), accurate at every angle. An `acos` of the cosine loses half the digits of a small angle.
"""
function weighted_angle(weights::AbstractVector{E}, left::AbstractVector{E}, right::AbstractVector{E})::E where {E <: Real}
    left_norm, right_norm = weighted_norm(weights, left), weighted_norm(weights, right)
    difference, total = zero(E), zero(E)
    for index in eachindex(weights, left, right)
        unit_left, unit_right = left[index] / left_norm, right[index] / right_norm
        difference += weights[index] * (unit_left - unit_right)^2
        total += weights[index] * (unit_left + unit_right)^2
    end
    return 2 * atan(sqrt(difference), sqrt(total))
end
