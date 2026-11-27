"""
The barrier between one lane's writes to a workgroup's memory and the other lanes' reads of them: `GroupBarrier()`
for a workgroup of several lanes, `LoneLane()` for a group of one.
"""
abstract type LaneBarrier end

"""
    GroupBarrier()

KernelAbstractions' `@synchronize`: every lane of the workgroup waits there. Only a GPU backend runs it outside a
kernel's own body, so use `LoneLane()` on the CPU backend.
"""
struct GroupBarrier <: LaneBarrier end

"""
    LoneLane()

No barrier: a group of one lane, on the host or in a CPU backend kernel of one-lane workgroups.
"""
struct LoneLane <: LaneBarrier end

"""Wait at `barrier` until every lane of the group arrives."""
synchronize_lanes(::GroupBarrier)::Nothing = (@synchronize(); nothing)
synchronize_lanes(::LoneLane)::Nothing = nothing

"""
    GroupMemory(floats, cursor, lane, lanes, barrier)

One workgroup's memory for a trace that each of its `lanes` runs in step: a flat buffer for the trace's floats, a
`cursor` of one `Int` (the count of floats taken), this lane's index in `1:lanes`, and the `barrier` between lanes
(`LaneBarrier`). In a kernel the buffers are `@localmem` arrays; on the host they are arrays, with one lane and
`LoneLane()`. Build one per launch and pass it beside the trace (`step!!(memory, tracer)`): the trace stores only
`GroupHandle`s, which `group_view(memory, tracer)` reads through the memory. `reset_memory!` empties it, and `group_array` takes
arrays from it.
"""
struct GroupMemory{E <: Real, F <: AbstractVector{E}, C <: AbstractVector{Int}, B <: LaneBarrier}
    floats::F
    cursor::C
    lane::Int32
    lanes::Int32
    barrier::B
    # Constant messages: a GPU cannot build an interpolated one.
    function GroupMemory(floats::F, cursor::C, lane::Integer, lanes::Integer, barrier::B)::GroupMemory{E, F, C, B} where {E <: Real, F <: AbstractVector{E}, C <: AbstractVector{Int}, B <: LaneBarrier}
        1 <= lane <= lanes || throw(ArgumentError("a group memory's lane lies in 1:lanes"))
        barrier isa LoneLane && lanes != 1 && throw(ArgumentError("LoneLane() runs a group of one lane"))
        length(cursor) == 1 || throw(ArgumentError("a group memory's cursor has one entry"))
        # By identity: buffers of zeros compare equal.
        storage(floats) === storage(cursor) && throw(ArgumentError("a group memory's buffer and cursor have separate storage"))
        return new{E, F, C, B}(floats, cursor, Int32(lane), Int32(lanes), barrier)
    end
end

"""What a trace's steps take beside it: its `GroupMemory`, or `nothing` for a start on the host or in a device array."""
const TraceMemory = Union{Nothing, GroupMemory}

"""
    GroupArray

A view of a `GroupHandle` in a workgroup's memory (`stored_view`): an array that each lane holds, its entries at the
handle's `offset + 1` onward, column major. Build it at each use, as it holds the whole memory, and store its handle
(`stored`). Every lane reads every entry. A write through `Into`, `set_element!!`, `fill!` or `copyto!` splits the
entries over the lanes between two barriers, so every lane sees the whole array after it. It has no `setindex!`: a
write from every lane at once races. An `Into` write throws when an argument shares the array's storage. A function
that captures the array and reads its other entries races; Base's broadcast checks no captures either.
`similar` takes a new array from the same memory; the memory is full when its buffer is.
"""
struct GroupArray{E <: Real, N, G <: GroupMemory{E}} <: AbstractArray{E, N}
    memory::G
    handle::GroupHandle{E, N}
end

"""A `GroupArray` of one dimension: a trace's point, tangent or residual in a workgroup's memory."""
const GroupVector{E, G} = GroupArray{E, 1, G}

"""A `GroupArray` of two dimensions: a trace's Jacobian or bordered system in a workgroup's memory."""
const GroupMatrix{E, G} = GroupArray{E, 2, G}

# `typeof` form until a fatou release fixes jolars/fatou#156 (call-arity reads this as the only `size` method).
function (::typeof(Base.size))(array::GroupArray)::Tuple
    return map(Int, array.handle.dims)
end

function Base.IndexStyle(::Type{A})::IndexLinear where {A <: GroupArray}
    return IndexLinear()
end

Base.@propagate_inbounds function Base.getindex(array::GroupArray{E}, index::Int)::E where {E <: Real}
    @boundscheck checkbounds(array, index)
    return array.memory.floats[array.handle.offset + index]
end

"""Write `entry` at linear `index` of `array` from this lane alone; callers split the entries over lanes between barriers."""
Base.@propagate_inbounds function store!(array::GroupArray, entry::Real, index::Int)::Nothing
    @boundscheck checkbounds(array, index)
    array.memory.floats[array.handle.offset + index] = entry
    return nothing
end

"""
The value that a trace stores as `stored`, for use: a `GroupHandle` viewed in `memory` (`GroupMemory`) as a
`GroupArray`, a value that holds handles with each one viewed, and any other value itself. `memory` is `nothing` for
a trace on the host or a device array. `stored` takes the view back. It cannot tell which memory a handle came from,
so outside the trace's own functions read through `group_view(memory, tracer)`, which checks.
"""
function stored_view(::TraceMemory, stored::S)::S where {S}
    return stored
end

function stored_view(::Nothing, ::GroupHandle)::Union{}
    throw(ArgumentError("a trace in group memory takes its GroupMemory beside it"))
end

function stored_view(::GroupMemory, ::GroupHandle)::Union{}
    throw(ArgumentError("a group handle holds its memory's float type"))
end

function stored_view(memory::M, handle::GroupHandle{E, N})::GroupArray{E, N, M} where {E <: Real, N, M <: GroupMemory{E}}
    return GroupArray{E, N, M}(memory, handle)
end

function stored_view(memory::GroupMemory, work::Workspace)::Workspace
    return map_arrays(array -> stored_view(memory, array), work)
end

function stored_view(memory::GroupMemory, metric::FixedWeights)::FixedWeights
    return FixedWeights(stored_view(memory, metric.weights))
end

function stored_view(memory::GroupMemory, problem::CurveProblem)::CurveProblem
    return CurveProblem(problem.system!!, stored_view(memory, problem.start), stored_view(memory, problem.direction))
end

function stored(array::GroupArray{E, N})::GroupHandle{E, N} where {E <: Real, N}
    return array.handle
end

stored(work::Workspace)::Workspace = map_arrays(stored, work)
stored(metric::FixedWeights)::FixedWeights = FixedWeights(stored(metric.weights))
stored(problem::CurveProblem)::CurveProblem = CurveProblem(problem.system!!, stored(problem.start), stored(problem.direction))

"""
The tag of the memory that a trace steps in: its buffer's and its cursor's addresses combined, or zero for `nothing`.
A trace keeps its memory's tag, so a step in another memory throws, as in one that shares the buffer under another
cursor. A buffer or cursor that has no `pointer` throws.
"""
memory_tag(::Nothing)::UInt = UInt(0)
memory_tag(memory::GroupMemory)::UInt = address(pointer(memory.floats)) ⊻ bitrotate(address(pointer(memory.cursor)), 32)

"""The address of a host or device pointer, as a number to compare."""
address(start::Ptr)::UInt = UInt(start)
address(start::Core.LLVMPtr)::UInt = reinterpret(UInt, start)

"""The type that `stored_view` gives for a stored `V` in a memory of type `M` (`nothing` for none)."""
viewed_type(::Type{<:TraceMemory}, ::Type{V}) where {V} = V
viewed_type(::Type{M}, ::Type{GroupHandle{E, N}}) where {E <: Real, N, M <: GroupMemory{E}} = GroupArray{E, N, M}

"""The memory that holds `start`'s entries: a `GroupArray`'s, or `nothing` for any other array."""
group_memory(start::GroupArray)::GroupMemory = start.memory
group_memory(::AbstractArray)::Nothing = nothing

"""Empty `memory`: lane 1 rewinds its cursor, then every lane waits. Call it before each trace's first `group_array`."""
function reset_memory!(memory::M)::M where {M <: GroupMemory}
    synchronize_lanes(memory.barrier)
    memory.lane == 1 && (memory.cursor[1] = 0)
    synchronize_lanes(memory.barrier)
    return memory
end

"""
    group_array(memory, E, dims)

A new `GroupArray` of `E` (`memory`'s float type) and size `dims` from `memory`'s free entries; every lane takes the
same one. Throws for another type, a negative dimension, and when `memory`'s buffer cannot hold it.
"""
function group_array(memory::M, ::Type{E}, dims::Dims{N})::GroupArray{E, N, M} where {E <: Real, M <: GroupMemory{E}, N}
    offset = memory.cursor[1]
    # Constant messages: a GPU cannot build an interpolated one.
    all(>=(0), dims) || throw(ArgumentError("a group array's dimensions are not negative"))
    count = foldl(checked_mul, dims; init = 1)
    count <= length(memory.floats) - offset || throw(ArgumentError("the group memory is full"))
    # Every lane reads the cursor before lane 1 moves it, and sees it moved after.
    synchronize_lanes(memory.barrier)
    memory.lane == 1 && (memory.cursor[1] = offset + count)
    synchronize_lanes(memory.barrier)
    return GroupArray{E, N, M}(memory, GroupHandle{E, N}(Int32(offset), map(Int32, dims)))
end

function group_array(::GroupMemory, ::Type{<:Real}, ::Dims)::Union{}
    throw(ArgumentError("a group memory holds arrays of its own float type"))
end

# `typeof` form until a fatou release fixes jolars/fatou#156 (call-arity reads this as the only `similar` method).
function (::typeof(Base.similar))(array::GroupArray, ::Type{E}, dims::Dims{N})::GroupArray{E, N} where {E <: Real, N}
    return group_array(array.memory, E, dims)
end

"""
Write `entry(index)` into each entry of `destination`, split over the lanes, between two barriers: the first waits
for every lane's reads of the old entries, the second for every lane's writes. `entry` takes a `CartesianIndex`.
"""
function lane_write!(entry::F, destination::A)::A where {F, A <: GroupArray}
    (; memory) = destination
    synchronize_lanes(memory.barrier)
    indices = CartesianIndices(destination)
    for index in memory.lane:memory.lanes:length(destination)
        @inbounds store!(destination, entry(indices[index]), index)
    end
    synchronize_lanes(memory.barrier)
    return destination
end

"""Whether `source`, a broadcast argument, reads entries of `destination` other than through `destination` itself."""
reads_destination(::GroupArray, ::Any)::Bool = false
reads_destination(destination::GroupArray, source::Broadcast.Broadcasted)::Bool = reads_any(destination, source.args...)

"""Whether any of `sources` reads entries of `destination` (`reads_destination`). Unrolled: `any` over a nested
broadcast's arguments does not infer, and a GPU cannot run the dynamic call."""
reads_any(::GroupArray)::Bool = false
function reads_any(destination::GroupArray, source::Any, sources::Any...)::Bool
    return reads_destination(destination, source) || reads_any(destination, sources...)
end
reads_destination(destination::GroupArray, source::AbstractArray)::Bool = source !== destination && shares_entries(destination, source)

"""The array at the end of `array`'s chain of `parent`s: the storage that holds its entries."""
storage(array::AbstractArray)::AbstractArray = parent(array) === array ? array : storage(parent(array))

"""Whether `source`, or an array it wraps (`parent`), may hold entries of `destination`."""
function shares_entries(destination::GroupArray, source::GroupArray)::Bool
    destination_buffer, source_buffer = destination.memory.floats, source.memory.floats
    # Two memories' buffers in one storage may lie anywhere in it.
    destination_buffer === source_buffer || return storage(destination_buffer) === storage(source_buffer)
    return source.handle.offset < destination.handle.offset + length(destination) &&
        destination.handle.offset < source.handle.offset + length(source)
end

function shares_entries(destination::GroupArray, source::AbstractArray)::Bool
    parent(source) === source && return source === storage(destination.memory.floats)
    return shares_entries(destination, parent(source))
end

function into!!(destination::A, broadcast::Broadcast.Broadcasted)::A where {A <: GroupArray}
    # A lane can overwrite an entry before another lane reads it, so a source must not share the destination's entries.
    reads_destination(destination, broadcast) && throw(ArgumentError("a broadcast into a GroupArray reads an array that overlaps it"))
    # As `Base.Broadcast.materialize!`: the broadcast takes the destination's axes, which it must fit.
    shaped = Broadcast.instantiate(Broadcast.Broadcasted(broadcast.style, broadcast.f, broadcast.args, axes(destination)))
    return lane_write!(index -> @inbounds(shaped[index]), destination)
end

function Base.fill!(array::A, entry::Real)::A where {A <: GroupArray}
    return lane_write!(_ -> entry, array)
end

"""As Base's `copyto!`: `source`'s entries into `destination`'s first `length(source)`, in linear order."""
function Base.copyto!(destination::A, source::AbstractArray)::A where {A <: GroupArray}
    count = length(source)
    count == 0 || checkbounds(destination, count)
    reads_destination(destination, source) && throw(ArgumentError("a broadcast into a GroupArray reads an array that overlaps it"))
    linear, first_entry = LinearIndices(destination), firstindex(source)
    # Each lane reads only the entries it writes, so keeping the tail is safe.
    return lane_write!(destination) do index
        at = linear[index]
        return at <= count ? source[first_entry + at - 1] : destination[at]
    end
end

function set_element!!(array::A, index::Integer, entry::Number)::A where {A <: GroupArray}
    (; memory) = array
    synchronize_lanes(memory.barrier)
    # The lane that `lane_write!` gives the entry.
    mod1(index, memory.lanes) == memory.lane && store!(array, entry, Int(index))
    synchronize_lanes(memory.barrier)
    return array
end

function border!!(bordered::X, jacobian::GroupMatrix, row::GroupVector)::X where {X <: GroupMatrix}
    width = size(jacobian, 1)
    return lane_write!(index -> index[1] <= width ? jacobian[index] : row[index[2]], bordered)
end

function negated_stack!!(rhs::V, residual::GroupVector{E}, offset::E)::V where {E <: Real, V <: GroupVector{E}}
    width = length(residual)
    return lane_write!(index -> index[1] <= width ? -residual[index] : -offset, rhs)
end

# Concrete types in the signature: a GPU cannot build a `UnionAll` such as `FixedWeights{<:GroupVector{E}}`.
function frozen_metric(metric::FixedWeights, point::V)::FixedWeights{V} where {E <: Real, V <: GroupVector{E}}
    matching_weights(metric, point)
    return FixedWeights(Into(similar(point)) .= metric.weights)
end

"""
    GroupLU()

`PseudoArclength`'s `linear_solver` for a start in a workgroup's memory (`GroupVector`): Gaussian elimination with
partial pivoting of `[A b]`, as LAPACK's `getrf` picks pivots. Lane `j` eliminates column `j`, one barrier per column;
then the lanes split the back substitution, one barrier per column. A zero pivot fails the solve. The orientation
check reads its sign from the pivots. It keeps no factors, so each solve eliminates anew, and it takes any size the
memory holds.
"""
struct GroupLU end

"""`GroupLU`'s state: the orientation sign of its last solve and the handle of its eliminations' pivots."""
struct GroupSolve{E <: Real}
    sign::Int8
    pivots::GroupHandle{E, 1}
end

"""`GroupLU`'s state for the bordered systems of `work`, a start's work arrays viewed in a workgroup's memory."""
function linear_state(::GroupLU, work::Workspace)::GroupSolve
    (; rhs) = work
    rhs isa GroupVector || throw(ArgumentError("GroupLU needs a start in a workgroup's memory (a GroupVector)"))
    return GroupSolve(Int8(0), stored(similar(rhs)))
end

"""
Solve `bordered \\ rhs` into `destination` by `GroupLU`, eliminating in `bordered` and `rhs`: the new state with the
system's orientation sign, the solution, and whether the solve succeeded with a finite solution.
"""
function linear_solve!!(
        linear::L,
        bordered::GroupMatrix{E},
        rhs::GroupVector{E},
        destination::V,
    )::Tuple{L, V, Bool} where {E <: Real, L <: GroupSolve{E}, V <: GroupVector{E}}
    (; memory) = bordered
    (; lane, lanes, barrier) = memory
    pivots, entries = stored_view(memory, linear.pivots), memory.floats
    order = length(rhs)
    sign = Int8(1)
    for column in 1:order
        # Every lane picks the same pivot: the first largest magnitude, as LAPACK's `idamax`. No lane writes `column`.
        pivot_row, largest = column, abs(bordered[column, column])
        for row in (column + 1):order
            magnitude = abs(bordered[row, column])
            if magnitude > largest
                pivot_row, largest = row, magnitude
            end
        end
        pivot = bordered[pivot_row, column]
        sign *= pivot_sign(pivot) * ifelse(pivot_row == column, Int8(1), Int8(-1))
        # Zero for every lane alike, so every lane returns here.
        iszero(sign) && return (GroupSolve(sign, linear.pivots), destination, false)
        # Lane `j` swaps the pivot row into place and eliminates below it in column `column + j`; `order + 1` is `rhs`.
        for later in (column + lane):lanes:(order + 1)
            base = later <= order ? bordered.handle.offset + (later - 1) * order : Int(rhs.handle.offset)
            top = entries[base + pivot_row]
            pivot_row == column || (entries[base + pivot_row] = entries[base + column])
            entries[base + column] = top
            for row in (column + 1):order
                # Column `column` keeps its rows unswapped: the pivot row's old entry sits at `pivot_row` there.
                below = bordered[ifelse(row == pivot_row, column, row), column]
                entries[base + row] -= below / pivot * top
            end
        end
        mod1(column, lanes) == lane && store!(pivots, pivot, column)
        synchronize_lanes(barrier)
    end
    # Back substitution by columns: each lane updates its rows of `rhs` above the solved entry.
    for column in order:-1:1
        solved = rhs[column] / pivots[column]
        for row in lane:lanes:(column - 1)
            store!(rhs, rhs[row] - bordered[row, column] * solved, row)
        end
        mod1(column, lanes) == lane && store!(destination, solved, column)
        synchronize_lanes(barrier)
    end
    return (GroupSolve(sign, linear.pivots), destination, all(isfinite, destination))
end

orientation_sign!(linear::GroupSolve, ::DeterminantSign, ::Workspace)::Int8 = linear.sign
