# First-order rounding bounds: a float computation run on `Rounded` numbers carries each result's slope to every
# rounding behind it, so its rounding error is bounded where it ran.

using LinearAlgebra: I

"""
One rounding `fl(a ∘ b) = (a ∘ b)(1 + δ)` of a float operation, with `|δ| ≤ bound` (Higham, *Accuracy and Stability of
Numerical Algorithms*, 2nd ed., (2.4)). Each rounding is its own object, so a `Dict` tells two apart by identity.
"""
mutable struct Rounding{W <: AbstractFloat}
    const bound::W
end

"""
    Rounded{E, W}(value, slopes)

A float `value` of type `E` and its `slopes`, in the wider type `W`, to each `Rounding` behind it: `value` exceeds the
exact result by `Σ slope δ`, up to terms in `δ²`. A generic float computation on `Rounded` inputs does the same float
operations on `value`, so `rounding_bound` bounds that computation's own rounding error. This is FPTaylor's first-order
error bound (Solovyev et al., ACM TOPLAS 41(1), 2018) by operator overloading.
"""
struct Rounded{E <: AbstractFloat, W <: AbstractFloat} <: Real
    value::E
    slopes::Dict{Rounding{W}, W}
end

# Julia 1.13's Float32 `exp` errs by at most 0.887 ulp over every Float32 in [-104, 0], the softmax's arguments.
const EXP_ULPS = 1

"""
An exact input `value` as a `Rounded`, with no rounding behind it; a `value` that `E` cannot hold throws. A `Rounded`
of another type has no method: its slopes are to roundings of another format.
"""
function Rounded{E, W}(value::Union{Integer, AbstractFloat})::Rounded{E, W} where {E <: AbstractFloat, W <: AbstractFloat}
    if E(value) != value
        throw(InexactError(:Rounded, E, value))
    end
    return Rounded{E, W}(E(value), Dict{Rounding{W}, W}())
end

# Exact constants promote; two `Rounded` types have no common type, so their arithmetic throws.
Base.promote_rule(::Type{Rounded{E, W}}, ::Type{<:Union{Integer, AbstractFloat}}) where {E, W} = Rounded{E, W}

"""
The `Rounded` result `value` of one float operation that rounds by at most `bound` relative to its result and, under
`floatmin(E)`, by at most `underflow` absolutely (Higham, 2nd ed., §2.1, gradual underflow). Each operand's slopes
times its partial derivative, given as `partial => operand`, join the slope `value` to the new rounding. An overflowed
or NaN `value` has no rounding bound and throws.
"""
function rounded(value::E, bound::W, underflow::W, parts::Pair{W, Rounded{E, W}}...)::Rounded{E, W} where {E <: AbstractFloat, W <: AbstractFloat}
    if !isfinite(value)
        throw(DomainError(value, "a rounded operation left the finite floats, where no rounding bound holds"))
    end
    slopes = Dict{Rounding{W}, W}(Rounding(bound) => W(value))
    if abs(value) < floatmin(E) && underflow > 0
        slopes[Rounding(underflow)] = one(W)
    end
    for (partial, operand) in parts, (rounding, slope) in operand.slopes
        slopes[rounding] = get(slopes, rounding, zero(W)) + partial * slope
    end
    return Rounded{E, W}(value, slopes)
end

"""The unit roundoff `u = eps(E) / 2` of round to nearest, in `W`."""
function unit_roundoff(::Type{Rounded{E, W}})::W where {E <: AbstractFloat, W <: AbstractFloat}
    return W(eps(E)) / 2
end

"""Half the subnormal spacing of `E`, `u floatmin(E)`, in `W`: the absolute rounding of a product or quotient under `floatmin`."""
function subnormal_roundoff(::Type{Rounded{E, W}})::W where {E <: AbstractFloat, W <: AbstractFloat}
    return unit_roundoff(Rounded{E, W}) * W(floatmin(E))
end

# A sum or difference under `floatmin` is exact, so it has no underflow rounding.
function Base.:+(left::R, right::R)::R where {R <: Rounded}
    return rounded(left.value + right.value, unit_roundoff(R), zero(unit_roundoff(R)), one(unit_roundoff(R)) => left, one(unit_roundoff(R)) => right)
end

function Base.:-(left::R, right::R)::R where {R <: Rounded}
    return rounded(left.value - right.value, unit_roundoff(R), zero(unit_roundoff(R)), one(unit_roundoff(R)) => left, -one(unit_roundoff(R)) => right)
end

function Base.:*(left::Rounded{E, W}, right::Rounded{E, W})::Rounded{E, W} where {E, W}
    return rounded(
        left.value * right.value, unit_roundoff(Rounded{E, W}), subnormal_roundoff(Rounded{E, W}), W(right.value) => left, W(left.value) => right,
    )
end

function Base.:/(left::Rounded{E, W}, right::Rounded{E, W})::Rounded{E, W} where {E, W}
    quotient = left.value / right.value
    return rounded(
        quotient, unit_roundoff(Rounded{E, W}), subnormal_roundoff(Rounded{E, W}), inv(W(right.value)) => left, -W(quotient) / W(right.value) => right,
    )
end

"""`-operand`, exact."""
function Base.:-(operand::Rounded{E, W})::Rounded{E, W} where {E, W}
    return Rounded{E, W}(-operand.value, Dict(rounding => -slope for (rounding, slope) in operand.slopes))
end

"""`exp(operand)`, within `EXP_ULPS` ulps; in the subnormal range an ulp is the subnormal spacing `eps(E) floatmin(E)`."""
function Base.exp(operand::Rounded{E, W})::Rounded{E, W} where {E, W}
    power = exp(operand.value)
    return rounded(power, EXP_ULPS * W(eps(E)), EXP_ULPS * W(eps(E)) * W(floatmin(E)), W(power) => operand)
end

"""
Whether `left` and `right` compare in exact arithmetic as their values do: both are exact, or their values differ by
more than their rounding bounds.
"""
function separated(left::Rounded{E, W}, right::Rounded{E, W})::Bool where {E, W}
    margin = rounding_bound(left) + rounding_bound(right)
    return iszero(margin) || abs(W(left.value) - W(right.value)) > margin
end

"""`comparison` of the values of `left` and `right`; operands not `separated` may compare otherwise exactly, and throw."""
function separated_comparison(comparison::Function, left::R, right::R)::Bool where {R <: Rounded}
    if !separated(left, right)
        throw(ArgumentError("$(left.value) and $(right.value) lie within their rounding bounds; exact operands may compare otherwise"))
    end
    return comparison(left.value, right.value)
end

Base.:(==)(left::R, right::R) where {R <: Rounded} = separated_comparison(==, left, right)
Base.:<(left::R, right::R) where {R <: Rounded} = separated_comparison(<, left, right)
Base.:<=(left::R, right::R) where {R <: Rounded} = separated_comparison(<=, left, right)
Base.isless(left::R, right::R) where {R <: Rounded} = separated_comparison(isless, left, right)

"""
The larger of `left` and `right`, with its slopes, when they are `separated`. Closer operands may sit in either order
in exact arithmetic: the exact maximum then errs by at most the larger bound, carried as one fresh rounding.
"""
function Base.max(left::Rounded{E, W}, right::Rounded{E, W})::Rounded{E, W} where {E, W}
    larger = ifelse(isless(right.value, left.value), left, right)
    if separated(left, right)
        return larger
    end
    return Rounded{E, W}(larger.value, Dict(Rounding(max(rounding_bound(left), rounding_bound(right))) => one(W)))
end

"""The first-order bound `Σ |slope| bound` on the rounding error of `number`'s value, over the roundings behind it."""
function rounding_bound(number::Rounded{E, W})::W where {E, W}
    return sum(abs(slope) * rounding.bound for (rounding, slope) in number.slopes; init = zero(W))
end

"""
`F(x, pin)`, the branch residual `[w - G(λ w); λ - pin]` of `game` at `x = [w; λ]`, evaluated by the engine's own
`GameResidual` on `Rounded` numbers: each entry carries its first-order rounding (`rounding_bound`).
"""
function rounded_residual(game::LogitGame, point::AbstractVector{E}, pin::E)::Vector where {E <: AbstractFloat}
    number = Rounded{E, widen(E)}
    return branch_residual(game, number.(point), number(pin))
end

"""`frozen` with each value it holds as an exact `number` (`Rounded`), so its products bound their own rounding."""
function rounded_frozen(frozen::GameTracing.FrozenFisher{<:Any, E}, number::Type{Rounded{E, W}})::GameTracing.FrozenFisher where {E, W}
    point = number.(frozen.point)
    return GameTracing.FrozenFisher(
        frozen.metric, point, number.(frozen.probabilities), number.(frozen.references), frozen.pivots,
        number(frozen.state_weight), number(frozen.palc_state_weight), number.(frozen.root),
    )
end

"""
    rounding_explanation(pass, failure, settings, target_precision)

`(; bound, residual, residual_bound, noise, amplification)` at the stalled corrector of `failure`, a
`PrecisionExhausted` trace of the game `pass` to `target_precision` under `settings`. PALC's Newton step at the stalled iterate
`x = [w; λ]`, its pin `p` and arclength `ds` solves the bordered system `B = [J ∂F/∂pin; θ A τu' (1 - θ) τp]`
(`bordered_matrix!`): `J` the Jacobian of the residual `F` (`rounded_residual`), `θ` PALC's state weight, and `A` the
dot weights and `τ` the tangent at the last accepted point `z0`. Its right side holds `F(x, p)` and the arclength row
`N = θ ⟨x - z0, τ⟩_A + (1 - θ) (p - p0) τp - ds` of BifurcationKit's `arc_length_eq`. The trace records `z0` and `τ`
by their gaps and pin parts; the pin row keeps their precision parts equal to these at accepted points. `noise` bounds
each row's rounding as the engine and `arc_length_eq` evaluate it. `amplification`, `[λ I  w] B⁻¹` in the wider type,
maps the rows to the logit change `Δz = λ Δw + w Δλ`. At noise the step's start is one noisy step from the root, so each
logit's correction is within `bound`: twice its amplified noise, plus `2u |z|` from the corrector's own products
`z = λ w`. That step leaves each row of `[F; N]` at `(x, p)` within its noise plus the update's own rounding
`|B| u |[x; p]|`; the `residual` computed there adds the noise again, within `residual_bound`. The solve's rounding of
a step this small is second order.
"""
function rounding_explanation(pass::Union{NormalFormGains, NormalFormPass}, failure::TraceFailure, settings::BifurcationKitSettings, target_precision::Real)::NamedTuple
    stalled_iterate, stalled_pin, stalled_arclength = failure.cause.stall.iterate, failure.cause.stall.pin, failure.cause.stall.arclength
    (; gaps, precision) = failure.stop
    (; gap_tangent, tangent_precision) = failure.engine
    element, width = eltype(stalled_iterate), length(gaps)
    # Only PALC's corrector runs strictly inside (0, target); a plain Newton solve at a bound has no border row.
    if !(0 < last(stalled_iterate) < target_precision)
        throw(ArgumentError("a stall at precision $(last(stalled_iterate)) is no PALC corrector inside (0, $target_precision)"))
    end
    game = LogitGame(pass, element(target_precision))
    jacobian = GameTracing.GameJacobian(game, GameTracing.BranchWorkspace(game, stalled_iterate))(stalled_iterate, stalled_pin)
    palc = GameTracing.palc_geometry(settings.step_metric, game.pass!!, width, target_precision)
    state_weight, start, tangent = element(palc.palc_state_weight), [gaps; precision], [gap_tangent; tangent_precision]
    # The step's weights are those of the point it starts from, the last accepted one.
    GameTracing.BifurcationKit.reweight!(palc.dot, GameTracing.BifurcationKit.BorderedArray(start, precision))
    bordered = GameTracing.bordered_matrix!(
        similar(jacobian, width + 2, width + 2), jacobian, GameTracing.PinSlope(stalled_iterate).slope, tangent,
        tangent_precision, state_weight, one(element) - state_weight, palc.dot.apply!,
    )
    logit_map = hcat(widen(last(stalled_iterate)) * I(width), widen.(stalled_iterate[1:width]))
    amplification = logit_map * inv(widen.(bordered))[1:(width + 1), :]
    arclength_row = GameTracing.BifurcationKit.arc_length_eq(
        palc.dot, stalled_iterate, start, stalled_pin - precision, tangent, tangent_precision, state_weight, stalled_arclength,
    )
    number = Rounded{element, widen(element)}
    rounded_dot = GameTracing.BifurcationKit.DotTheta(GameTracing.MetricDot(rounded_frozen(palc.dot.dot.frozen, number)))
    rounded_arclength_row = GameTracing.BifurcationKit.arc_length_eq(
        rounded_dot, number.(stalled_iterate), number.(start), number(stalled_pin) - number(precision),
        number.(tangent), number(tangent_precision), number(state_weight), number(stalled_arclength),
    )
    noise = [rounding_bound.(rounded_residual(game, stalled_iterate, stalled_pin)); rounding_bound(rounded_arclength_row)]
    bound = 2 .* (abs.(amplification) * noise) .+ eps(element) .* abs.(GameTracing.logits_at(stalled_iterate))
    residual = [branch_residual(game, stalled_iterate, stalled_pin); arclength_row]
    roundoff = eps(element) / 2
    residual_bound = 2 .* noise .+ abs.(widen.(bordered)) * (roundoff .* abs.(widen.([stalled_iterate; stalled_pin])))
    return (; bound, residual, residual_bound, noise, amplification)
end

"""
Whether `failure` is a `PrecisionExhausted` stall at rounding noise, on a trace of the game `pass` to `target_precision` under
`settings`: the residual of `F` and of its step's arclength row at the stalled iterate is within `residual_bound` of
`rounding_explanation`, so the iterate is at the step's root up to noise, and its last correction in logits is within
`bound`.
"""
function exhausted_by_rounding(pass::Union{NormalFormGains, NormalFormPass}, failure::TraceFailure, settings::BifurcationKitSettings, target_precision::Real)::Bool
    if !failed_as(failure, TraceCause.PrecisionExhausted)
        return false
    end
    (; bound, residual, residual_bound) = rounding_explanation(pass, failure, settings, target_precision)
    return all(abs.(residual) .<= residual_bound) && failure.cause.stall.correction <= maximum(bound)
end

"""`failure`'s stop and engine record, as a stall mid-trace at `stall`."""
function stalled_at(failure::TraceFailure{E}, stall::StalledCorrection{E})::TraceFailure where {E <: Real}
    return TraceFailure(TraceCause.PrecisionExhausted(GameTracing.MidTrace, stall), failure.stop, failure.engine, failure.context)
end

"""`failure`, a `PrecisionExhausted` stall, with the fields of its stall that `changes` names replaced."""
function restalled(failure::TraceFailure, changes::NamedTuple)::TraceFailure
    (; precision, correction, iterate, pin, arclength) = failure.cause.stall
    # `merge` keeps the field order, so an unknown name adds a sixth argument and the constructor throws.
    return stalled_at(failure, StalledCorrection(merge((; precision, correction, iterate, pin, arclength), changes)...))
end
