using BifurcationKit: BifurcationKit
using LinearAlgebra: mul!
using PathFollowing: @into!!
using StaticArrays: SMatrix, SVector, StaticMatrix, StaticVector, pop, push
"""
    LogitGame(pass!!, target_precision)

A logit game whose equilibria solve `z = λ G(z, λ)` at precision `λ = 1 / temperature`, `G` the payoff gains the
logits scale. `pass!!(gain, gain_slope, gain_precision_slope, z, λ)` writes `G(z, λ)`, its Jacobian `∂G/∂z` and its
slope `∂G/∂λ` into the three arrays, from one evaluation, and returns them (`@into!!`): mutable arrays changed in place,
static arrays new. The mutable arrays are views into the trace's own residual and Jacobian (`branch_rows!!`), so a
pass allocates nothing of its own. A pass that needs work arrays of its own gets them from `pass_workspace`. `G` must
be finite at `λ = 0`, where the flat logits are the only equilibrium. `target_precision` is the precision a trace ends
at, non-negative and finite (`ArgumentError` otherwise). A traced pass also defines `logit_ranges` and `gap_bound`.
"""
struct LogitGame{P, T <: Real}
    pass!!::P
    target_precision::T
    function LogitGame{P, T}(pass!!::P, target_precision::Real)::LogitGame{P, T} where {P, T <: Real}
        stored = convert(T, target_precision)
        # Both: a conversion can round a negative target to -0 or a huge one to Inf.
        if !(reachable(target_precision) && reachable(stored))
            throw(ArgumentError("target_precision must be non-negative and finite in $T; got $target_precision"))
        end
        return new{P, T}(pass!!, stored)
    end
end

function LogitGame(pass!!::P, target_precision::T)::LogitGame{P, T} where {P, T <: Real}
    return LogitGame{P, T}(pass!!, target_precision)
end

"""Whether a trace can reach `precision`: it rises from its start at `λ = 0`, so a non-negative, finite one."""
reachable(precision::Real)::Bool = precision >= 0 && isfinite(precision)

"""
    pass_workspace(pass!!, logits)

The work arrays `pass!!` needs beyond its outputs, in the element type, length and storage of `logits`, or `nothing`
(the default) when it needs none. A pass with work arrays is called as
`pass!!(work, gain, gain_slope, gain_precision_slope, z, λ)` and returns the work first, then its outputs (`@into!!`).
A trace builds them once per game and shares them across every evaluation of that game. So a pass may read what
`pass_workspace` wrote, such as the game's constants, and writes every other work array before it reads it. The work
belongs to its game: no other game's pass gets it. Each trace builds its own, so a game shared by two traces stays
safe. A redrawn game's work must have the type of the trace's (`swapped`). Static work arrays come back new from each
pass, and a static trace drops them: the next pass starts again from `pass_workspace`'s.
"""
pass_workspace(::Any, ::AbstractVector)::Nothing = nothing

"""
    logit_ranges(pass!!)

Each player's range in the stacked logits of `pass!!`'s game: the logits of its strategies against its reference
strategy, one `UnitRange{Int}` per player in a tuple or vector, tiling the logits in order (`checked_ranges`). A player
with two strategies has one logit; a player with one strategy has an empty range. The Fisher metric (`FisherDistance`) and the corrector's KL limit (`max_kl`) read
it. It has no default: a pass traced under either must define it.
"""
function logit_ranges end

"""
    gap_bound(pass!!)

A bound on every payoff gap `|w|` of `pass!!`'s game over the strategies the trace meets: the inverse of each trace's
first step (`opening_step`). A normal-form game's is its payoff range (`payoff_range`). Every traced pass
defines it; a pass without its own method throws an `ArgumentError` that names the method to define.
"""
function gap_bound(pass!!::Any)::Union{}
    throw(ArgumentError("$(typeof(pass!!)) has no gap bound: define GameTracing.gap_bound(::$(typeof(pass!!))), a bound on its payoff gaps"))
end

"""
The first arclength step of a trace of `game` in `geometry`, at most `max_step`: `root / gap_bound(game.pass!!)`.
A step `h` from `λ = 0` moves `λ` by at most `h / root` (`opening_root`), so no logit `z = λ w` moves by more than one,
whatever the target. Step control grows or shrinks it from there. The quotient overflows only for a tiny bound; a step
that is not positive and finite (a zero, tiny or non-finite bound) is one, and a non-finite start fails (`NotFinite`).
"""
function opening_step(game::LogitGame, geometry::TraceGeometry{E}, max_step::Real)::E where {E <: Real}
    step = opening_root(geometry, game.target_precision) / E(gap_bound(game.pass!!))
    return min(step > 0 && isfinite(step) ? step : one(E), E(max_step))
end

"""
`ranges` (`logit_ranges`) when they tile the `width` logits in order, one range per player, else an `ArgumentError`. An
empty range, a player of one strategy, starts where the next range starts.
"""
function checked_ranges(ranges::R, width::Integer)::R where {R}
    next, tiled = 1, true
    for range in ranges
        tiled &= first(range) == next
        next = last(range) + 1
    end
    if !(tiled && next == width + 1)
        throw(ArgumentError("the logit ranges $(collect(ranges)) do not tile the logits 1:$width in order"))
    end
    return ranges
end

"""
Call `pass!!` on its outputs, with its work arrays `work` unless it has none (`nothing`). Return the work and the
three outputs, each changed in place or new (`@into!!`). A pass that returns anything else, such as `nothing`, throws an
`ArgumentError`. The check folds away for a pass whose return type infers.
"""
function run_pass!!(
        pass!!::Any,
        work::W,
        gain::G,
        gain_slope::S,
        gain_precision_slope::P,
        logits::AbstractVector,
        precision::Real,
    )::Tuple{W, G, S, P} where {W, G <: AbstractVector, S <: AbstractMatrix, P <: AbstractVector}
    if isnothing(work)
        outputs = pass!!(gain, gain_slope, gain_precision_slope, logits, precision)
        outputs isa Tuple{G, S, P} || throw(ArgumentError(lazy"pass!! must return (gain, gain_slope, gain_precision_slope) as a $(Tuple{G, S, P}); got a $(typeof(outputs))"))
        return (work, outputs...)
    end
    returned = pass!!(work, gain, gain_slope, gain_precision_slope, logits, precision)
    returned isa Tuple{W, G, S, P} || throw(ArgumentError(lazy"pass!! must return (work, gain, gain_slope, gain_precision_slope) as a $(Tuple{W, G, S, P}); got a $(typeof(returned))"))
    return returned
end

"""
    BranchWorkspace(point, pass_work)
    BranchWorkspace(game, point)

A residual, Jacobian, logits and a game's pass work arrays `pass_work` (`pass_workspace`) for branch points like
`point = [w; λ]`, refilled by every call of the one callable that owns it. BifurcationKit consumes each result
before calling that callable again, so the residual, fused system and Jacobian each own one workspace per segment; a
shared one would let `J` overwrite a residual `FJ` returned. They share the pass work, which no pass reads before it
writes (`pass_workspace`). The form with `game` builds fresh pass work, for a single evaluation outside a trace.
"""
struct BranchWorkspace{E <: Real, W}
    residual::Vector{E}
    jacobian::Matrix{E}
    logits::Vector{E}
    pass_work::W
end

function BranchWorkspace(point::AbstractVector{E}, pass_work::Any)::BranchWorkspace{E} where {E <: Real}
    return BranchWorkspace(
        Vector{E}(undef, length(point)),
        Matrix{E}(undef, length(point), length(point)),
        Vector{E}(undef, length(point) - 1),
        pass_work,
    )
end

function BranchWorkspace(game::LogitGame, point::AbstractVector{E})::BranchWorkspace{E} where {E <: Real}
    # A `Vector`, as `trace_endpoint` passes: a redraw's swap needs pass work of the trace's type.
    return BranchWorkspace(point, pass_workspace(game.pass!!, zeros(E, length(point) - 1)))
end

"""
    GameResidual(game, workspace)

BifurcationKit's `F(x, pin)` of `game`'s branch in payoff gaps, with the precision carried as the state's
last coordinate: `x = [w; λ]`, `F = [w - G(λ w, λ); λ - pin]`. The logits are `z = λ w`,
so `F = 0` is `z = λ G(z, λ)`, and `F` stays regular at `λ = 0`.
`pin` is BifurcationKit's continuation parameter, which only pins `λ`. Each call refills `workspace` and returns a
copy of its residual, which BifurcationKit may keep past its next call of `F`.
"""
struct GameResidual{G <: LogitGame, W <: BranchWorkspace}
    game::G
    workspace::W
end

function (residual::GameResidual)(point::AbstractVector{E}, pin::Real)::AbstractVector{E} where {E <: Real}
    return copy(branch_system!(residual.workspace, residual.game, point, pin).residual)
end

"""
    GameSystem(game, workspace)

BifurcationKit's fused `FJ(x, pin)` of `game`'s branch: the residual and Jacobian from one `pass!!`, written into
`workspace` (`branch_system!`). PALC's Newton method calls it in place of the residual and `J`.
"""
struct GameSystem{G <: LogitGame, W <: BranchWorkspace}
    game::G
    workspace::W
end

function (system::GameSystem)(point::AbstractVector{E}, pin::Real)::Tuple{
        Vector{E},
        Matrix{E},
    } where {E <: Real}
    filled = branch_system!(system.workspace, system.game, point, pin)
    return filled.residual, filled.jacobian
end

"""BifurcationKit's `J(x, pin)` of `game`'s branch, written into `workspace`."""
struct GameJacobian{G <: LogitGame, W <: BranchWorkspace}
    game::G
    workspace::W
end

function (jacobian::GameJacobian)(point::AbstractVector{E}, pin::Real)::AbstractMatrix{E} where {E <: Real}
    return branch_system!(jacobian.workspace, jacobian.game, point, pin).jacobian
end

"""The payoff-gap entries `w` of a vector over `[w; λ]`: a branch point's gaps, or a residual's or weight's gap part."""
payoff_gaps(point::AbstractVector)::AbstractVector = view(
    point,
    firstindex(point):(lastindex(point) - 1),
)

payoff_gaps(point::StaticVector)::StaticVector = pop(point)

"""The logits `z = λ w` of a branch point `x = [w; λ]`."""
function logits_at(point::AbstractVector{E})::AbstractVector{E} where {E <: Real}
    return last(point) .* payoff_gaps(point)
end

"""
    branch_rows!!(residual, jacobian, logits, pass_work, game, x)

`game`'s branch residual `F = w - G(λ w, λ)` at `x = [w; λ]` in `residual`, and its Jacobian `[∂F/∂w  ∂F/∂λ]` in
`jacobian`: `∂F/∂w = I - λ ∂G/∂z` and `∂F/∂λ = -(∂G/∂z w + ∂G/∂λ)`. The logits `z = λ w` go into `logits`, and the
pass runs on its work arrays `pass_work` (`pass_workspace`). Return the four arrays (`@into!!`). Mutable arrays hold
`G`, `∂G/∂z` and `∂G/∂λ` in the blocks of `F` and its Jacobian (`pass_blocks`), which then form in place. Static
arrays form new ones.
"""
function branch_rows!!(
        residual::R,
        jacobian::J,
        logits::L,
        pass_work::W,
        game::LogitGame,
        point::AbstractVector{E},
    )::Tuple{R, J, L, W} where {E <: Real, R <: AbstractVector{E}, J <: AbstractMatrix{E}, L <: AbstractVector{E}, W}
    gaps = payoff_gaps(point)
    precision = point[end]
    logits = @into!! logits .= precision .* gaps
    gain, gain_slope, precision_column = pass_blocks(residual, jacobian)
    pass_work, gain, gain_slope, precision_column = run_pass!!(game.pass!!, pass_work, gain, gain_slope, precision_column, logits, precision)
    # `∂F/∂λ` first: it reads `∂G/∂z`, which the next line overwrites.
    precision_column = precision_slope!!(precision_column, gain_slope, gaps)
    gain_slope = @into!! gain_slope .= .-precision .* gain_slope
    for diagonal in axes(gain_slope, 1)
        gain_slope = set_entry!!(gain_slope, gain_slope[diagonal, diagonal] + one(E), diagonal, diagonal)
    end
    gain = @into!! gain .= gaps .- gain
    return (joined_rows!!(residual, jacobian, gain, gain_slope, precision_column)..., logits, pass_work)
end

"""
The blocks a pass writes `G`, `∂G/∂z` and `∂G/∂λ` into, for a residual and a Jacobian `[∂F/∂w  ∂F/∂λ]`: views of
mutable arrays, or new static arrays of their sizes.
"""
function pass_blocks(residual::AbstractVector, jacobian::AbstractMatrix)::Tuple{AbstractVector, AbstractMatrix, AbstractVector}
    width = length(residual)
    return residual, @view(jacobian[:, 1:width]), @view(jacobian[:, width + 1])
end

function pass_blocks(residual::StaticVector{W, E}, ::StaticMatrix)::Tuple{StaticVector, StaticMatrix, StaticVector} where {W, E <: Real}
    return residual, zeros(SMatrix{W, W, E}), zeros(SVector{W, E})
end

"""`∂F/∂λ = -(∂G/∂z w + ∂G/∂λ)` from `∂G/∂λ` in `column`, `∂G/∂z` in `slope` and the gaps `w` (`@into!!`)."""
function precision_slope!!(column::C, slope::AbstractMatrix{E}, gaps::AbstractVector{E})::C where {E <: Real, C <: AbstractVector{E}}
    return mul!(column, slope, gaps, -one(E), -one(E))
end

function precision_slope!!(
        column::C,
        slope::StaticMatrix{<:Any, <:Any, E},
        gaps::StaticVector{<:Any, E},
    )::C where {E <: Real, C <: StaticVector{<:Any, E}}
    return C(.-(slope * gaps) .- column)
end

"""
The residual and Jacobian `[∂F/∂w  ∂F/∂λ]` from their blocks (`pass_blocks`): the mutable arrays, which the blocks
view, or new static ones.
"""
function joined_rows!!(
        residual::R,
        jacobian::J,
        ::AbstractVector,
        ::AbstractMatrix,
        ::AbstractVector,
    )::Tuple{R, J} where {R <: AbstractVector, J <: AbstractMatrix}
    return residual, jacobian
end

function joined_rows!!(
        ::R,
        ::J,
        gain::StaticVector,
        gain_slope::StaticMatrix,
        precision_column::StaticVector,
    )::Tuple{R, J} where {R <: StaticVector, J <: StaticMatrix}
    return R(gain), J(hcat(gain_slope, precision_column))
end

"""
    branch_system!(workspace, game, x, pin)

`workspace` with `game`'s branch residual and Jacobian at `x = [w; λ]`, BifurcationKit's square system: the rows of
`branch_rows!!` over the precision's own row, `λ - pin` and `[0 … 0 1]`.
"""
function branch_system!(
        workspace::BranchWorkspace{E},
        game::LogitGame,
        point::AbstractVector{E},
        pin::Real,
    )::typeof(workspace) where {E <: Real}
    (; residual, jacobian, logits, pass_work) = workspace
    width = length(point) - 1
    branch_rows!!(view(residual, 1:width), @view(jacobian[1:width, :]), logits, pass_work, game, point)
    fill!(@view(jacobian[width + 1, 1:width]), zero(E))
    jacobian[end, end] = one(E)
    residual[end] = point[end] - pin
    return workspace
end

"""
The residual and Jacobian `[∂F/∂w  ∂F/∂λ]` of `branch_rows!!` for `width` gaps, uninitialized: mutable arrays like
`flat`, or static ones for a static `flat`.
"""
function branch_arrays(flat::AbstractVector{E})::Tuple{AbstractVector{E}, AbstractMatrix{E}} where {E <: Real}
    width = length(flat)
    return similar(flat), similar(flat, width, width + 1)
end

function branch_arrays(::StaticVector{W, E})::Tuple{StaticVector{W, E}, StaticMatrix{W}} where {W, E <: Real}
    return zeros(SVector{W, E}), zeros(SMatrix{W, W + 1, E})
end

"""`vector` with `entry` appended: a new mutable array, or a new static array."""
function appended(vector::AbstractVector{E}, entry::E)::AbstractVector{E} where {E <: Real}
    return [vector; entry]
end

function appended(vector::StaticVector{<:Any, E}, entry::E)::StaticVector{<:Any, E} where {E <: Real}
    return push(vector, entry)
end

"""
    trace_start(game, flat, pass_work)

The branch's start `[w; 0]` at zero precision: the gaps `G(0, 0)` that the `flat` logits (all zero) face, from one
pass on `game`'s work arrays `pass_work` (`pass_workspace`), in `flat`'s storage.
"""
function trace_start(game::LogitGame, flat::AbstractVector{E}, pass_work::Any)::AbstractVector{E} where {E <: Real}
    residual, jacobian = branch_arrays(flat)
    point = appended(flat, zero(E))
    residual, _, _, _ = branch_rows!!(residual, jacobian, zero(flat), pass_work, game, point)
    # The residual at the flat start is `0 - G`.
    return appended(.-residual, zero(E))
end

"""
    PinSlope(start)

`∂F/∂pin` of the branch residual `[F(w, λ); λ - pin]`, BifurcationKit's jet `R01`: `[0 … 0 -1]`, exactly, in
`start`'s element type and length. BifurcationKit copies the vector it gets, so one serves every call.
The constructor is inner because the field's default one would take `start` itself as the slope.
"""
struct PinSlope{E <: Real}
    slope::Vector{E}

    function PinSlope(start::AbstractVector{E})::PinSlope{E} where {E <: Real}
        return new{E}(setindex!(zeros(E, length(start)), -one(E), length(start)))
    end
end

function (pin_slope::PinSlope{E})(::AbstractVector{E}, ::Any)::AbstractVector{E} where {
        E <:
        Real,
    }
    return pin_slope.slope
end

"""
    branch_problem(game, start, pin, pass_work; record_from_solution)

BifurcationKit's problem for `game`'s branch from `start = [w; λ]` at `pin`, with the fused system and
exact `∂F/∂pin` supplied. The residual, system and Jacobian each own a `BranchWorkspace` on the pass work `pass_work`.
"""
function branch_problem(
        game::LogitGame,
        start::AbstractVector{E},
        pin::Real,
        pass_work::Any;
        record_from_solution::Any,
    )::BifurcationKit.AbstractBifurcationProblem where {E <: Real}
    return BifurcationKit.BifurcationProblem(
        GameResidual(game, BranchWorkspace(start, pass_work)),
        start,
        pin;
        J = GameJacobian(game, BranchWorkspace(start, pass_work)),
        FJ = GameSystem(game, BranchWorkspace(start, pass_work)),
        delta = one(E),
        inplace = false,
        R01 = PinSlope(start),
        record_from_solution,
    )
end
