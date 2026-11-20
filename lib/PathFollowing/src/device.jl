"""`array[index]`. A GPU array reads the one element with `@allowscalar`, a transfer to the host."""
function element(array::AbstractArray{E}, index::Integer)::E where {E}
    return array[index]
end

function element(array::AnyGPUArray{E}, index::Integer)::E where {E}
    return @allowscalar array[index]
end

"""`array` with `entry` at `index`. A GPU array writes the one element with `@allowscalar`."""
function set_element!(array::A, index::Integer, entry::Number)::A where {A <: AbstractArray}
    array[index] = entry
    return array
end

function set_element!(array::A, index::Integer, entry::Number)::A where {A <: AnyGPUArray}
    @allowscalar array[index] = entry
    return array
end

"""`vector` set to the unit vector along `index`, by one broadcast on any device."""
function unit!(vector::V, index::Integer)::V where {E <: Real, V <: AbstractVector{E}}
    vector .= E.(eachindex(vector) .== index)
    return vector
end

"""
`operation` broadcast over `arrays` without materializing. `mapreduce` over it runs as one GPU reduction, and on the
CPU it allocates nothing.
"""
function elementwise(operation::Any, arrays::AbstractArray...)::Broadcast.Broadcasted
    return Broadcast.instantiate(Broadcast.broadcasted(operation, arrays...))
end
