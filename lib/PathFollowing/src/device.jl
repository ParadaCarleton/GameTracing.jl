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
    return Into(vector) .= E.(eachindex(vector) .== index)
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
    Into(destination) .= expression

Write the broadcast `expression` into `destination` and return the result. A mutable array (`Vector`, `Matrix`, a GPU
array) changes in place and comes back itself, so the step allocates nothing. A static array (`SVector`, `SMatrix`)
cannot change, so a new one of its type comes back, as a GPU kernel needs. Use the returned array:

```julia
residual = Into(residual) .= sum(abs2, point) - radius^2
```
"""
struct Into{A <: AbstractArray}
    destination::A
end

function Broadcast.materialize!(into::Into{A}, broadcast::Broadcast.Broadcasted)::A where {A <: AbstractArray}
    return into!!(into.destination, broadcast)
end

"""`broadcast` written into `destination` in place, or into a new array of `destination`'s static type."""
function into!!(destination::A, broadcast::Broadcast.Broadcasted)::A where {A <: AbstractArray}
    return Broadcast.materialize!(destination, broadcast)
end

function into!!(::A, broadcast::Broadcast.Broadcasted)::A where {A <: SArray}
    return A(Broadcast.materialize(broadcast))
end
