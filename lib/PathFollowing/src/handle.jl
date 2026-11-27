"""
    GroupHandle{E, N}

Where an array of `E` lies in a workgroup's memory (`GroupMemory`): its entries at `offset + 1` onward, column major,
of size `dims`. A trace in group memory stores its arrays as handles, so a copy of its state copies a few integers;
each use views a handle through the memory that the trace's caller passes beside it (`stored_view`).
"""
struct GroupHandle{E <: Real, N}
    offset::Int32
    dims::NTuple{N, Int32}
end

"""A trace's stored vector of `E`: an `AbstractVector`, or a `GroupHandle` of one dimension."""
const StoredVector{E} = Union{AbstractVector{E}, GroupHandle{E, 1}}

"""A trace's stored matrix of `E`: an `AbstractMatrix`, or a `GroupHandle` of two dimensions."""
const StoredMatrix{E} = Union{AbstractMatrix{E}, GroupHandle{E, 2}}

"""
    stored(view)

What a trace stores for `view`: a `GroupArray`'s `GroupHandle`, a value that holds group arrays with each one stored,
and any other value itself. `stored_view` takes it back.
"""
function stored(view::S)::S where {S}
    return view
end
