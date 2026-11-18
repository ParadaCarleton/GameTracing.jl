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

"""An exact input `value` as a `Rounded`, with no rounding behind it; a `value` that `E` cannot hold throws."""
function Rounded{E, W}(value::Real)::Rounded{E, W} where {E <: AbstractFloat, W <: AbstractFloat}
    if E(value) != value
        throw(InexactError(:Rounded, E, value))
    end
    return Rounded{E, W}(E(value), Dict{Rounding{W}, W}())
end

Base.promote_rule(::Type{Rounded{E, W}}, ::Type{<:Real}) where {E, W} = Rounded{E, W}

"""
The `Rounded` result `value` of one float operation that rounds by at most `bound`: each operand's slopes times its
partial derivative, given as `partial => operand`, and the slope `value` to the new rounding.
"""
function rounded(value::E, bound::W, parts::Pair{W, Rounded{E, W}}...)::Rounded{E, W} where {E <: AbstractFloat, W <: AbstractFloat}
    slopes = Dict{Rounding{W}, W}(Rounding(bound) => W(value))
    for (partial, operand) in parts, (rounding, slope) in operand.slopes
        slopes[rounding] = get(slopes, rounding, zero(W)) + partial * slope
    end
    return Rounded{E, W}(value, slopes)
end

"""The unit roundoff `u = eps(E) / 2` of round to nearest, in `W`."""
function unit_roundoff(::Type{Rounded{E, W}})::W where {E <: AbstractFloat, W <: AbstractFloat}
    return W(eps(E)) / 2
end

function Base.:+(left::R, right::R)::R where {R <: Rounded}
    return rounded(left.value + right.value, unit_roundoff(R), one(unit_roundoff(R)) => left, one(unit_roundoff(R)) => right)
end

function Base.:-(left::R, right::R)::R where {R <: Rounded}
    return rounded(left.value - right.value, unit_roundoff(R), one(unit_roundoff(R)) => left, -one(unit_roundoff(R)) => right)
end

function Base.:*(left::Rounded{E, W}, right::Rounded{E, W})::Rounded{E, W} where {E, W}
    return rounded(left.value * right.value, unit_roundoff(Rounded{E, W}), W(right.value) => left, W(left.value) => right)
end

function Base.:/(left::Rounded{E, W}, right::Rounded{E, W})::Rounded{E, W} where {E, W}
    quotient = left.value / right.value
    return rounded(quotient, unit_roundoff(Rounded{E, W}), inv(W(right.value)) => left, -W(quotient) / W(right.value) => right)
end

"""`-operand`, exact."""
function Base.:-(operand::Rounded{E, W})::Rounded{E, W} where {E, W}
    return Rounded{E, W}(-operand.value, Dict(rounding => -slope for (rounding, slope) in operand.slopes))
end

function Base.exp(operand::Rounded{E, W})::Rounded{E, W} where {E, W}
    power = exp(operand.value)
    return rounded(power, EXP_ULPS * W(eps(E)), W(power) => operand)
end

# Comparisons read the values, so a computation branches as it does on plain floats; `max` then picks an operand.
Base.isless(left::R, right::R) where {R <: Rounded} = isless(left.value, right.value)
Base.:<(left::R, right::R) where {R <: Rounded} = left.value < right.value
Base.:<=(left::R, right::R) where {R <: Rounded} = left.value <= right.value
Base.:(==)(left::R, right::R) where {R <: Rounded} = left.value == right.value

"""The first-order bound `Σ |slope| bound` on the rounding error of `number`'s value, over the roundings behind it."""
function rounding_bound(number::Rounded{E, W})::W where {E, W}
    return sum(abs(slope) * rounding.bound for (rounding, slope) in number.slopes; init = zero(W))
end
