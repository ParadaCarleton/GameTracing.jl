using StaticArrays: SArray, setindex

"""
`array` with `entry` at `indices`: written in place into a mutable array, or a new static array (StaticArrays'
`setindex`). Use the returned array. GameTracing's static path never writes one element of a GPU array; PathFollowing's
`set_element!!` does.
"""
@inline function set_entry!!(array::A, entry::Number, indices::Integer...)::A where {A <: AbstractArray}
    array[indices...] = entry
    return array
end

@inline function set_entry!!(array::A, entry::Number, indices::Integer...)::A where {A <: SArray}
    return setindex(array, entry, indices...)
end

"""`array` set to zeros: in place for a mutable array, or a new static array. Use the returned array."""
@inline function zero!!(array::A)::A where {A <: AbstractArray}
    return fill!(array, zero(eltype(array)))
end

@inline function zero!!(array::A)::A where {A <: SArray}
    return zero(array)
end

"""`collection` with `item` at `index`: written into a vector, or a new tuple. Use the returned collection."""
@inline function with_item!!(collection::C, index::Integer, item::Any)::C where {C <: AbstractVector}
    collection[index] = item
    return collection
end

@inline function with_item!!(collection::C, index::Integer, item::Any)::C where {C <: Tuple}
    return setindex(collection, item, index)
end
