using BifurcationKit: BifurcationKit
using LinearAlgebra: mul!
"""
    LogitGame(pass!, target_precision)

A logit game whose equilibria solve `z = λ G(z, λ)` at precision `λ = 1 / temperature`, `G` the payoff gains the
logits scale. `pass!(gain, gain_slope, gain_precision_slope, z, λ)` writes `G(z, λ)`, its Jacobian `∂G/∂z` and its
slope `∂G/∂λ` into the three arrays, from one evaluation, and returns `nothing`. The arrays are views into the
trace's own residual and Jacobian (`BranchWorkspace`), so a pass allocates nothing of its own. A pass that needs work
arrays of its own gets them from `pass_workspace`. `G` must be finite at `λ = 0`, where the flat logits are the only
equilibrium. `target_precision` is the precision a trace ends at, non-negative and finite (`ArgumentError` otherwise).
"""
struct LogitGame{P, T <: Real}
    pass!::P
    target_precision::T
    function LogitGame{P, T}(pass!::P, target_precision::Real)::LogitGame{P, T} where {P, T <: Real}
        stored = convert(T, target_precision)
        # Both: a conversion can round a negative target to -0 or a huge one to Inf.
        if !(reachable(target_precision) && reachable(stored))
            throw(ArgumentError("target_precision must be non-negative and finite in $T; got $target_precision"))
        end
        return new{P, T}(pass!, stored)
    end
end

function LogitGame(pass!::P, target_precision::T)::LogitGame{P, T} where {P, T <: Real}
    return LogitGame{P, T}(pass!, target_precision)
end

"""Whether a trace can reach `precision`: it rises from its start at `λ = 0`, so a non-negative, finite one."""
reachable(precision::Real)::Bool = precision >= 0 && isfinite(precision)

"""
    pass_workspace(pass!, logits)

The work arrays `pass!` needs beyond its outputs, in the element type and length of `logits`, or `nothing` (the
default) when it needs none. Each `BranchWorkspace` builds its own, so a game shared by two traces stays safe. A pass
with work arrays is called as `pass!(work, gain, gain_slope, gain_precision_slope, z, λ)`.
"""
pass_workspace(::Any, ::AbstractVector)::Nothing = nothing

"""Call `pass!` on its outputs, with its work arrays `work` unless it has none."""
function run_pass!(
        pass!::Any,
        work::Any,
        gain::AbstractVector,
        gain_slope::AbstractMatrix,
        gain_precision_slope::AbstractVector,
        logits::AbstractVector,
        precision::Real,
    )::Nothing
    if isnothing(work)
        pass!(gain, gain_slope, gain_precision_slope, logits, precision)
    else
        pass!(work, gain, gain_slope, gain_precision_slope, logits, precision)
    end
    return nothing
end

"""
    BranchWorkspace(game, point)

A residual, Jacobian, logits and `game`'s pass work arrays (`pass_workspace`) for branch points like
`point = [w; λ]`, refilled by every call of the one callable that owns it. BifurcationKit consumes each result
before calling that callable again, so the fused system and the Jacobian each own one workspace per trace; a shared
one would let `J` overwrite a residual `FJ` returned.
"""
struct BranchWorkspace{E <: Real, W}
    residual::Vector{E}
    jacobian::Matrix{E}
    logits::Vector{E}
    pass_work::W
end

function BranchWorkspace(game::LogitGame, point::AbstractVector{E})::BranchWorkspace{E} where {E <: Real}
    return BranchWorkspace(
        Vector{E}(undef, length(point)),
        Matrix{E}(undef, length(point), length(point)),
        Vector{E}(undef, length(point) - 1),
        pass_workspace(game.pass!, payoff_gaps(point)),
    )
end

"""
    GameResidual(game)

BifurcationKit's `F(x, pin)` of `game`'s branch in payoff gaps, with the precision carried as the state's
last coordinate: `x = [w; λ]`, `F = [w - G(λ w, λ); λ - pin]`. The logits are `z = λ w`,
so `F = 0` is `z = λ G(z, λ)`, and `F` stays regular at `λ = 0`.
`pin` is BifurcationKit's continuation parameter, which only pins `λ`. Each call fills a fresh `BranchWorkspace`:
BifurcationKit keeps its problem's `F` apart from the fused system.
"""
struct GameResidual{G <: LogitGame}
    game::G
end

function (residual::GameResidual)(point::AbstractVector{E}, pin::Real)::AbstractVector{E} where {E <: Real}
    return branch_system!(BranchWorkspace(residual.game, point), residual.game, point, pin).residual
end

"""
    GameSystem(game, workspace)

BifurcationKit's fused `FJ(x, pin)` of `game`'s branch: the residual and Jacobian from one `pass!`, written into
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

"""The logits `z = λ w` of a branch point `x = [w; λ]`."""
function logits_at(point::AbstractVector{E})::AbstractVector{E} where {E <: Real}
    return last(point) .* payoff_gaps(point)
end

"""
    branch_system!(workspace, game, x, pin)

`workspace` with `game`'s branch residual and Jacobian at `x = [w; λ]`: `∂F/∂w = I - λ ∂G/∂z` and
`∂F/∂λ = -(∂G/∂z w + ∂G/∂λ)`, over the precision's own row `[0 … 0 1]`. The pass writes `G`, `∂G/∂z`
and `∂G/∂λ` into the blocks that hold `F` and its Jacobian, which are then formed in place.
"""
function branch_system!(
        workspace::BranchWorkspace{E},
        game::LogitGame,
        point::AbstractVector{E},
        pin::Real,
    )::typeof(workspace) where {E <: Real}
    (; residual, jacobian, logits, pass_work) = workspace
    gaps = payoff_gaps(point)
    width = length(gaps)
    precision = point[end]
    logits .= precision .* gaps
    gain = view(residual, 1:width)
    gain_slope = @view jacobian[1:width, 1:width]
    precision_column = @view jacobian[1:width, width + 1]
    run_pass!(game.pass!, pass_work, gain, gain_slope, precision_column, logits, precision)
    # `∂F/∂λ` first: it reads `∂G/∂z`, which the next line overwrites.
    mul!(precision_column, gain_slope, gaps, -one(E), -one(E))
    gain_slope .= .-precision .* gain_slope
    for diagonal in axes(gain_slope, 1)
        gain_slope[diagonal, diagonal] += one(E)
    end
    gain .= gaps .- gain
    fill!(@view(jacobian[width + 1, 1:width]), zero(E))
    jacobian[end, end] = one(E)
    residual[end] = precision - pin
    return workspace
end

"""
    trace_start(game, flat)

The branch's start `[w; 0]` at zero precision: the gaps `G(0, 0)` that the `flat` logits (all zero) face.
"""
function trace_start(game::LogitGame, flat::AbstractVector{E})::AbstractVector{E} where {E <: Real}
    start = [flat; zero(E)]
    payoff_gaps(start) .= .-payoff_gaps(branch_system!(BranchWorkspace(game, start), game, start, zero(E)).residual)
    return start
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
    branch_problem(game, start, pin; record_from_solution)

BifurcationKit's problem for `game`'s branch from `start = [w; λ]` at `pin`, with the fused system and
exact `∂F/∂pin` supplied. The system and Jacobian each own a `BranchWorkspace`.
"""
function branch_problem(
        game::LogitGame,
        start::AbstractVector{E},
        pin::Real;
        record_from_solution::Any,
    )::BifurcationKit.AbstractBifurcationProblem where {E <: Real}
    return BifurcationKit.BifurcationProblem(
        GameResidual(game),
        start,
        pin;
        J = GameJacobian(game, BranchWorkspace(game, start)),
        FJ = GameSystem(game, BranchWorkspace(game, start)),
        delta = one(E),
        inplace = false,
        R01 = PinSlope(start),
        record_from_solution,
    )
end
