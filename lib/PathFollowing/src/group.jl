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
    GroupMemory(floats, integers, cursor, lane, lanes, barrier)

One workgroup's memory for a trace that each of its `lanes` runs in step: flat buffers for the trace's floats and
`Int`s, a `cursor` of two entries (the next free float and the next free integer), this lane's index in `1:lanes`,
and the `barrier` between lanes (`LaneBarrier`).
In a kernel the buffers are `@localmem` arrays; on the host they are `Vector`s, with one lane and `LoneLane()`.
`reset_memory!` empties it, and `group_array` takes arrays from it.
"""
struct GroupMemory{E <: Real, F <: AbstractVector{E}, I <: AbstractVector{Int}, C <: AbstractVector{Int}, B <: LaneBarrier}
    floats::F
    integers::I
    cursor::C
    lane::Int
    lanes::Int
    barrier::B
    # Constant messages: a GPU cannot build an interpolated one.
    function GroupMemory(floats::F, integers::I, cursor::C, lane::Integer, lanes::Integer, barrier::B)::GroupMemory{E, F, I, C, B} where {E <: Real, F <: AbstractVector{E}, I <: AbstractVector{Int}, C <: AbstractVector{Int}, B <: LaneBarrier}
        1 <= lane <= lanes || throw(ArgumentError("a group memory's lane lies in 1:lanes"))
        barrier isa LoneLane && lanes != 1 && throw(ArgumentError("LoneLane() runs a group of one lane"))
        length(cursor) == 2 || throw(ArgumentError("a group memory's cursor has two entries"))
        # By identity: `allunique` compares entries, and buffers of zeros compare equal.
        floats_root, integers_root, cursor_root = storage(floats), storage(integers), storage(cursor)
        shared = floats_root === integers_root || integers_root === cursor_root || floats_root === cursor_root
        shared && throw(ArgumentError("a group memory's buffers and cursor have separate storage"))
        return new{E, F, I, C, B}(floats, integers, cursor, Int(lane), Int(lanes), barrier)
    end
end

"""
    GroupArray

An array in a workgroup's memory (`GroupMemory`) that each lane holds: its entries at `offset + 1` onward, column
major. Every lane reads every entry. A write through `into!!`, `set_element!!`, `fill!` or `copyto!` splits the entries
over the lanes between two barriers, so every lane sees the whole array after it. It has no `setindex!`: a write
from every lane at once races. An `into!!` write throws when an argument shares the array's storage. A function that
captures the array and reads its other entries races; Base's broadcast checks no captures either.
`similar` takes a new array from the same memory; the memory is full when its buffers are.
"""
struct GroupArray{E <: Real, N, G <: GroupMemory} <: AbstractArray{E, N}
    memory::G
    offset::Int
    dims::NTuple{N, Int}
end

"""A `GroupArray` of one dimension: a trace's point, tangent or residual in a workgroup's memory."""
const GroupVector{E, G} = GroupArray{E, 1, G}

"""A `GroupArray` of two dimensions: a trace's Jacobian or bordered system in a workgroup's memory."""
const GroupMatrix{E, G} = GroupArray{E, 2, G}

# `typeof` form until a fatou release fixes jolars/fatou#156 (call-arity reads this as the only `size` method).
function (::typeof(Base.size))(array::GroupArray)::Tuple
    return array.dims
end

function Base.IndexStyle(::Type{A})::IndexLinear where {A <: GroupArray}
    return IndexLinear()
end

"""The buffer of `memory` that holds `GroupArray`s of `E`: its integers for `Int`, else its floats."""
function buffer(memory::GroupMemory, ::Type{E})::AbstractVector where {E <: Real}
    return E === Int ? memory.integers : memory.floats
end

"""The cursor entry that counts a memory's taken entries of `E`: 2 for `Int`, else 1."""
(cursor_slot(::Type{E})::Int) where {E <: Real} = E === Int ? 2 : 1

Base.@propagate_inbounds function Base.getindex(array::GroupArray{E}, index::Int)::E where {E <: Real}
    @boundscheck checkbounds(array, index)
    return buffer(array.memory, E)[array.offset + index]
end

"""Write `entry` at linear `index` of `array` from this lane alone; callers split the entries over lanes between barriers."""
Base.@propagate_inbounds function store!(array::GroupArray{E}, entry::Real, index::Int)::Nothing where {E <: Real}
    @boundscheck checkbounds(array, index)
    buffer(array.memory, E)[array.offset + index] = entry
    return nothing
end

"""Empty `memory`: lane 1 rewinds its cursor, then every lane waits. Call it before each trace's first `group_array`."""
function reset_memory!(memory::GroupMemory)::GroupMemory
    synchronize_lanes(memory.barrier)
    if memory.lane == 1
        memory.cursor[1] = 0
        memory.cursor[2] = 0
    end
    synchronize_lanes(memory.barrier)
    return memory
end

"""
    group_array(memory, E, dims)

A new `GroupArray` of `E` (`memory`'s float type or `Int`) and size `dims` from `memory`'s free entries; every lane
takes the same one. Throws for a negative dimension, and when `memory`'s buffer of `E` cannot hold it.
"""
function group_array(memory::M, ::Type{E}, dims::Dims{N})::GroupArray{E, N, M} where {F <: Real, M <: GroupMemory{F}, E <: Real, N}
    E === F || E === Int || throw(ArgumentError("a group memory holds its own float type and Int"))
    slot = cursor_slot(E)
    offset = memory.cursor[slot]
    # Constant messages: a GPU cannot build an interpolated one.
    all(>=(0), dims) || throw(ArgumentError("a group array's dimensions are not negative"))
    count = foldl(checked_mul, dims; init = 1)
    count <= length(buffer(memory, E)) - offset || throw(ArgumentError("the group memory is full"))
    # Every lane reads the cursor before lane 1 moves it, and sees it moved after.
    synchronize_lanes(memory.barrier)
    memory.lane == 1 && (memory.cursor[slot] = offset + count)
    synchronize_lanes(memory.barrier)
    return GroupArray{E, N, M}(memory, offset, dims)
end

# `typeof` form until a fatou release fixes jolars/fatou#156 (call-arity reads this as the only `similar` method).
function (::typeof(Base.similar))(array::GroupArray{<:Real, <:Any, M}, ::Type{E}, dims::Dims{N})::GroupArray{E, N, M} where {M <: GroupMemory, E <: Real, N}
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
function shares_entries(destination::GroupArray{E}, source::GroupArray{S})::Bool where {E <: Real, S <: Real}
    destination_buffer, source_buffer = buffer(destination.memory, E), buffer(source.memory, S)
    # Two memories' buffers in one storage may lie anywhere in it.
    destination_buffer === source_buffer || return storage(destination_buffer) === storage(source_buffer)
    return source.offset < destination.offset + length(destination) && destination.offset < source.offset + length(source)
end
function shares_entries(destination::GroupArray{E}, source::AbstractArray)::Bool where {E <: Real}
    parent(source) === source && return source === storage(buffer(destination.memory, E))
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
    return FixedWeights(into!!(similar(point), metric.weights))
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

"""`GroupLU`'s state: the orientation sign of its last solve and the pivots of its eliminations, in group memory."""
struct GroupSolve{V <: GroupVector}
    sign::Int8
    pivots::V
end

"""`GroupLU`'s state for the bordered systems of `work`, a start in a workgroup's memory."""
function linear_state(::GroupLU, work::Workspace)::GroupSolve
    (; rhs) = work
    rhs isa GroupVector || throw(ArgumentError("GroupLU needs a start in a workgroup's memory (a GroupVector)"))
    return GroupSolve(Int8(0), similar(rhs))
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
    )::Tuple{L, V, Bool} where {E <: Real, L <: GroupSolve, V <: GroupVector{E}}
    (; memory) = bordered
    (; lane, lanes, barrier) = memory
    (; pivots) = linear
    entries = buffer(memory, E)
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
        iszero(sign) && return (GroupSolve(sign, pivots), destination, false)
        # Lane `j` swaps the pivot row into place and eliminates below it in column `column + j`; `order + 1` is `rhs`.
        for later in (column + lane):lanes:(order + 1)
            base = later <= order ? bordered.offset + (later - 1) * order : rhs.offset
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
    return (GroupSolve(sign, pivots), destination, all(isfinite, destination))
end

orientation_sign!(linear::GroupSolve, ::DeterminantSign, ::Workspace)::Int8 = linear.sign
