@testset "a Rounded bound covers underflow and near ties, and throws where no bound holds ($element)" for element in (Float32, widen(Float32))
    number = Rounded{element, widen(element)}
    # Half the smallest subnormal rounds to zero; the bound covers the half it lost.
    halved = number(nextfloat(zero(element))) / number(2)
    @test iszero(halved.value)
    @test rounding_bound(halved) >= big(nextfloat(zero(element))) / 2
    # `exp` deep in the subnormal range errs by up to the subnormal spacing.
    deep = element(round(log(floatmin(element))) - 16)
    power = exp(number(deep))
    @test issubnormal(power.value)
    @test abs(big(power.value) - exp(big(deep))) <= rounding_bound(power)
    @test_throws DomainError number(floatmax(element)) * number(2)
    @test_throws DomainError number(0) / number(0)
    # `1 + u` rounds to 1: tied with an exact 1, the maximum keeps the lost `u` in either order.
    nudged = number(1) + number(eps(element) / 2)
    @test nudged.value == 1
    @test rounding_bound(max(nudged, number(1))) >= eps(element) / 2
    @test rounding_bound(max(number(1), nudged)) >= eps(element) / 2
    # Separated operands keep the larger one's slopes; close ones refuse to compare.
    @test max(nudged, number(2)).slopes == number(2).slopes
    @test nudged < number(2)
    @test number(1) == number(1)
    @test_throws ArgumentError nudged == number(1)
    @test_throws ArgumentError nudged <= number(1)
    # Slopes to roundings of another format have no common type with these.
    other = Rounded{widen(element), widen(widen(element))}
    @test_throws ErrorException number(1) + other(1)
    @test_throws MethodError number(other(1))
end
