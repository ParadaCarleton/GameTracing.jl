# First-order rounding bounds: a float computation run on `Rounded` numbers carries each result's slope to every
# rounding behind it, so its rounding error is bounded where it ran.

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
