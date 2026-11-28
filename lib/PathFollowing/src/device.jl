"""
    @into!! destination .= expression

Write the broadcast `expression` into `destination` and return the result. A mutable array (`Vector`, `Matrix`, a GPU
array) changes in place and comes back itself, so the step allocates nothing. A static array (`SVector`, `SMatrix`)
cannot change, so a new one of its type comes back, as a GPU kernel needs. Use the returned array:

```julia
jacobian = @into!! jacobian .= 2 .* transpose(point)
residual = @into!! residual .= sum(abs2, point) - radius^2
```

The macro hands `into!!` the fused broadcast unbuilt (`lazy_broadcast`, LazyBroadcast.jl), so `expression` builds no
temporary. An array or a number on the right is copied or fills every entry, as Base's `.=` does. Only the dotted
calls fuse: an undotted `a + b` of two arrays builds its array first, as in Base.
"""
macro into!!(assignment)
    Meta.isexpr(assignment, :.=, 2) || throw(ArgumentError("@into!! takes `destination .= expression`, not `$assignment`"))
    destination, expression = assignment.args
    # A typed literal `E[1, 2]` parses as indexing `x[i]` does; `typed_literal` tells them apart by the type of `E`.
    target = Meta.isexpr(destination, :ref) ? :($typed_literal($(map(esc, destination.args)...))) : esc(destination)
    # `identity.` makes a bare array or number a broadcast, so `into!!` takes no eagerly built source.
    source = dotted_call(expression) ? esc(expression) : :($identity.($(esc(expression))))
    return :($into!!($target, $lazy_broadcast.($source)))
end

"""The typed literal `type[entries...]`, an `@into!!` destination that reads like indexing."""
typed_literal(type::Type, entries::Any...)::AbstractArray = type[entries...]

"""Throws: `@into!! x[i] .= ...` would fill a copy of `x[i]`, as `x[i]` builds one. A view writes through to `x`."""
function typed_literal(indexed::Any, ::Any...)::Union{}
    throw(ArgumentError(lazy"@into!! writes a whole array, not an indexed $(typeof(indexed)); write `@into!! (@view x[i]) .= ...`"))
end

"""
Whether `expression` is a dotted call (`f.(x)`, `a .+ b`), which `lazy_broadcast` turns into a broadcast as it stands.
A misread either way fails safe: a missed one takes an `identity.` layer, and a false one throws in `into!!`.
"""
function dotted_call(expression::Any)::Bool
    Meta.isexpr(expression, :.) && return Meta.isexpr(expression.args[2], :tuple)
    Meta.isexpr(expression, :call) && expression.args[1] isa Symbol || return false
    # Only operators start with a dot, and `..` is the one undotted operator among them.
    return startswith(String(expression.args[1]), '.') && expression.args[1] != :..
end

"""
    into!!(destination, broadcast)

The unbuilt `broadcast` written into `destination`, as `@into!!` documents. It takes no array or number: an eager broadcast
`a .* b` arrives as a temporary array, so the call refuses it. Write `@into!! destination .= expression`.
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
    return @into!! vector .= E.(eachindex(vector) .== index)
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
