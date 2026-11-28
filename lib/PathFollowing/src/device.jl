"""`array[index]`. A GPU array reads the one element with `@allowscalar`, a transfer to the host."""
function element(array::AbstractArray{E}, index::Integer)::E where {E}
    return array[index]
end

function element(array::AnyGPUArray{E}, index::Integer)::E where {E}
    return @allowscalar array[index]
end

"""
`array` with `entry` at `index`: written in place, with `@allowscalar` on a GPU array, or a new static array. Use the
returned array.
"""
function set_element!!(array::A, index::Integer, entry::Number)::A where {A <: AbstractArray}
    array[index] = entry
    return array
end

function set_element!!(array::A, index::Integer, entry::Number)::A where {A <: AnyGPUArray}
    @allowscalar array[index] = entry
    return array
end

function set_element!!(array::A, index::Integer, entry::Number)::A where {A <: SArray}
    return setindex(array, entry, index)
end

"""`vector` set to the unit vector along `index`: one broadcast on any device, or a new static vector."""
function unit!!(vector::V, index::Integer)::V where {E <: Real, V <: AbstractVector{E}}
    return into!!(vector, lazy_broadcast.(E.(eachindex(vector) .== index)))
end

# A broadcast over a static vector's indices builds a heap `SizedVector`.
function unit!!(vector::V, index::Integer)::V where {E <: Real, V <: SVector{<:Any, E}}
    return setindex(zero(vector), one(E), index)
end

"""
`operation` broadcast over `arrays` without materializing. `mapreduce` over it runs as one GPU reduction, and on the
CPU it allocates nothing.
"""
function elementwise(operation::Any, arrays::AbstractArray...)::Broadcast.Broadcasted
    return Broadcast.instantiate(Broadcast.broadcasted(operation, arrays...))
end

"""
The sum in `E` of `operation` over `arrays`' entries (`elementwise`): Neumaier's compensated fold in the arrays' promoted
element type, which keeps small terms that larger ones cancel around. Base's `sum` of 16 or more entries calls its
recursive `mapreduce_impl`, so a kernel's stack size would have no static bound. GPU arrays sum in one GPU reduction.
"""
function elementwise_sum(::Type{E}, operation::Any, arrays::AbstractArray...)::E where {E <: Real}
    S = promote_type(E, map(eltype, arrays)...)
    total, compensation = foldl(compensated_add, elementwise(operation, arrays...); init = (zero(S), zero(S)))
    return E(total + compensation)
end

function elementwise_sum(::Type{E}, operation::Any, arrays::AnyGPUArray...)::E where {E <: Real}
    return E(mapreduce(identity, +, elementwise(operation, arrays...); init = zero(promote_type(E, map(eltype, arrays)...))))
end

"""
Neumaier's step: the running `total` with `entry` added, and `compensation` with the rounding error that adds. A
non-finite total takes no error term, so an overflow stays infinite and a NaN stays NaN.
"""
function compensated_add((total, compensation)::Tuple{S, S}, entry::Real)::Tuple{S, S} where {S <: Real}
    added = total + entry
    error = abs(total) >= abs(entry) ? (total - added) + entry : (entry - added) + total
    return added, compensation + ifelse(isfinite(added), error, zero(error))
end

"""
    into!!(destination, lazy_broadcast.(expression))
    into!!(destination, source)

Write the broadcast `expression`, or an array or number `source`, into `destination` and return the result. A mutable
array (`Vector`, `Matrix`, a GPU array) changes in place and comes back itself, so the step allocates nothing. A static
array (`SVector`, `SMatrix`) cannot change, so a new one of its type comes back, as a GPU kernel needs. Use the returned
array:

```julia
using LazyBroadcast: lazy_broadcast
jacobian = into!!(jacobian, lazy_broadcast.(2 .* transpose(point)))
residual = into!!(residual, sum(abs2, point) - radius^2)
```

`lazy_broadcast` (LazyBroadcast.jl) hands `into!!` the fused broadcast unbuilt; without it, a dotted `source` allocates a
temporary. `into!!` is a call because a `.=` expression's value differs across Julia versions.
"""
function into!!(destination::A, broadcast::Broadcast.Broadcasted)::A where {A <: AbstractArray}
    return Broadcast.materialize!(destination, broadcast)
end

function into!!(destination::A, broadcast::Broadcast.Broadcasted)::A where {A <: SArray}
    # As `materialize!`: the broadcast takes the destination's axes, so a scalar or a singleton fills every entry.
    # StaticArrays' broadcast takes its result type from a static argument, which a scalar source lacks.
    shaped = Broadcast.instantiate(Broadcast.Broadcasted(broadcast.style, broadcast.f, broadcast.args, axes(destination)))
    indices = CartesianIndices(destination)
    return A(ntuple(at -> @inbounds(shaped[indices[at]]), Val(length(A))))
end

function into!!(destination::A, source::Union{AbstractArray, Number})::A where {A <: AbstractArray}
    return into!!(destination, Broadcast.Broadcasted(identity, (source,)))
end
