"""
A tracer's work arrays. `residual` and `jacobian` hold `H` and `∂H/∂x` at the last point the system saw, `row` is the
border row of the next bordered system, and the `split_` arrays hold a corrected point, its tangent and its `previous`
through the split check. `scratch`, `bordered` and `rhs` hold values only within one call. A trace in group memory
stores `GroupHandle`s here (`stored`).
"""
struct Workspace{V <: StoredVector, R <: StoredVector, J <: StoredMatrix, B <: StoredMatrix}
    scratch::V
    residual::R
    jacobian::J
    bordered::B
    row::V
    rhs::V
    predicted::V
    iterate::V
    previous::V
    candidate::V
    split_point::V
    split_tangent::V
    split_previous::V
end

"""`work` with `operation` applied to each of its arrays, as a stored array or its view (`stored_view`)."""
function map_arrays(operation::Any, work::Workspace)::Workspace
    return Workspace(map(operation, Tuple(getfields(work)))...)
end

"""A zero vector like `start` in `E`: `similar` to it, so on its device, or static for a static start."""
function zero_point(start::AbstractVector, ::Type{E})::AbstractVector{E} where {E <: Real}
    return fill!(similar(start, E), zero(E))
end

function zero_point(::StaticVector{N}, ::Type{E})::SVector{N, E} where {N, E <: Real}
    return zero(SVector{N, E})
end

"""Zeroed work arrays for points like `start` in `E`: `similar` to it, so on its device, or static for a static start."""
function workspace(start::AbstractVector, ::Type{E})::Workspace where {E <: Real}
    size = length(start)
    point() = zero_point(start, E)
    return Workspace(
        point(), fill!(similar(start, E, size - 1), zero(E)), fill!(similar(start, E, size - 1, size), zero(E)),
        fill!(similar(start, E, size, size), zero(E)), point(), point(), point(), point(), point(), point(), point(), point(),
        point(),
    )
end

function workspace(start::StaticVector{N}, ::Type{E})::Workspace where {N, E <: Real}
    point = zero_point(start, E)
    return Workspace(
        point, zero(SVector{N - 1, E}), zero(SMatrix{N - 1, N, E}), zero(SMatrix{N, N, E}), point, point, point, point,
        point, point, point, point, point,
    )
end
