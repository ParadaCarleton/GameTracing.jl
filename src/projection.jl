using LinearAlgebra: Diagonal

"""
    projection_step(game, logits, precision, geometry)

One minimum-norm Gauss–Newton step from `x = [logits / λ; λ]` onto `game`'s branch `F(x) = 0` in payoff gaps, in
`geometry`'s plain norm `W` at the game's target precision (`plain_weights`): `dx = -W⁻¹ Jᵀ (J W⁻¹ Jᵀ)⁻¹ F`.
It is the minimum-norm solution of `(J W^-½) y = -F` by pivoted QR, scaled back by `W^-½`; the normal matrix is never
formed. The step moves `λ` as well, so it stays defined at a fold, where a correction at fixed `λ` is singular.
Payoff gaps `w = z / λ` exist at every finite `λ ≠ 0`, so `λ` must be one.
"""
function projection_step(
        game::LogitGame,
        logits::AbstractVector{E},
        precision::Real,
        geometry::TraceGeometry,
    )::AbstractVector{E} where {E <: Real}
    if !(0 < geometry.state_weight < 1)
        throw(ArgumentError("state_weight must lie in (0, 1); got $(geometry.state_weight)"))
    end
    if iszero(precision) || !isfinite(precision)
        throw(ArgumentError("payoff gaps need a finite nonzero precision; got $precision"))
    end
    width = length(logits)
    point = [logits ./ E(precision); E(precision)]
    system = branch_system!(BranchWorkspace(game, point), game, point, precision)
    jacobian = system.jacobian[1:width, :]
    gap, precision_weight = plain_weights(geometry, width, game.target_precision)
    scale = Diagonal(inv.(sqrt.([fill(gap, width); precision_weight])))
    return -(scale * ((jacobian * scale) \ system.residual[1:width]))
end

"""
    branch_projection(game, logits, precision, settings)

`(; logits, precision)`: the point one `projection_step` from `(logits, λ)` toward `game`'s branch, in the
trace's norm (`settings.step_metric`). The projection is first order: its residual is quadratic in the start's.
A step to a precision that is not positive and finite, as one from far off the branch can take, throws a `DomainError`.
"""
function branch_projection(
        game::LogitGame,
        logits::AbstractVector{E},
        precision::Real,
        settings::BifurcationKitSettings,
    )::NamedTuple{(:logits, :precision), Tuple{Vector{E}, E}} where {E <: Real}
    step = projection_step(game, logits, precision, settings.step_metric)
    moved = [logits ./ E(precision); E(precision)] .+ step
    if !(last(moved) > 0 && isfinite(last(moved)))
        throw(DomainError(last(moved), "the projection left the positive precisions"))
    end
    return (; logits = logits_at(moved), precision = last(moved))
end
