"""
The state of a LinearSolve.jl solver for a mutable start: its `cache`, and the matrix `oriented` and row swaps `pivots`
of the orientation check's second factorization, for a solver that keeps no LU. Only LAPACK's factorization of a
`Matrix` writes `pivots`, so it stays on the host.
"""
struct CachedSolve{C, X <: AbstractMatrix}
    cache::C
    oriented::X
    pivots::Vector{LinearAlgebra.BlasInt}
end

"""`StaticLU`'s state: the LU of the last bordered system it solved."""
struct StaticSolve{F <: StaticArrays.LU}
    factorization::F
end

"""The state of `solver` for the bordered systems of `work`: a `CachedSolve` for a mutable start."""
function linear_state(solver::Any, work::Workspace)::CachedSolve
    work.bordered isa StaticArray && throw(ArgumentError("a static start needs linear_solver = StaticLU(); got $solver"))
    # A failed solve is a refusal (`SingularSystem`), so LinearSolve's own warning would only repeat it.
    cache = LinearSolve.init(LinearSolve.LinearProblem(work.bordered, work.rhs), solver; verbose = false)
    oriented = fill!(similar(work.bordered), zero(eltype(work.bordered)))
    return CachedSolve(cache, oriented, zeros(LinearAlgebra.BlasInt, size(work.bordered, 1)))
end

"""`StaticLU`'s state for the bordered systems of `work`, a static start's of at most 14 coordinates."""
function linear_state(::StaticLU, work::Workspace)::StaticSolve
    (; bordered) = work
    bordered isa SMatrix || throw(ArgumentError("StaticLU needs a static start (an SVector); got bordered systems of type $(typeof(bordered))"))
    # Past 14 × 14, StaticArrays factors through a heap `Matrix`.
    size(bordered, 1) <= 14 || throw(ArgumentError("StaticLU factors at most 14 × 14 systems without allocation; got $(size(bordered, 1))"))
    return StaticSolve(LinearAlgebra.lu(bordered, LinearAlgebra.RowMaximum(); check = false))
end

"""
Solve `bordered \\ rhs` into `destination` under `linear`, the solver's state: the new state, the solution, and whether
the solve succeeded with a finite solution.
"""
function linear_solve!!(
        linear::L,
        bordered::AbstractMatrix{E},
        rhs::AbstractVector{E},
        destination::V,
    )::Tuple{L, V, Bool} where {E <: Real, L <: CachedSolve, V <: AbstractVector{E}}
    linear.cache.A = bordered
    linear.cache.b = rhs
    solution = LinearSolve.solve!(linear.cache)
    SciMLBase.successful_retcode(solution) || return (linear, destination, false)
    copyto!(destination, solution.u)
    return (linear, destination, all(isfinite, destination))
end

function linear_solve!!(
        ::L,
        bordered::SMatrix{N, N, E},
        rhs::SVector{N, E},
        destination::V,
    )::Tuple{L, V, Bool} where {N, E <: Real, L <: StaticSolve, V <: SVector{N, E}}
    factorization = LinearAlgebra.lu(bordered, LinearAlgebra.RowMaximum(); check = false)
    # A zero pivot makes StaticArrays' triangular solve throw; a NaN pivot gives a NaN solution.
    LinearAlgebra.issuccess(factorization) || return (StaticSolve(factorization), destination, false)
    solution = factorization \ rhs
    return (StaticSolve(factorization), solution, all(isfinite, solution))
end

"""`[jacobian; rowᵀ]` in `bordered`: in place, or a new static matrix."""
function border!!(bordered::X, jacobian::AbstractMatrix, row::AbstractVector)::X where {X <: AbstractMatrix}
    width = size(jacobian, 1)
    @views bordered[1:width, :] .= jacobian
    @views bordered[end, :] .= row
    return bordered
end

function border!!(::X, jacobian::SMatrix, row::SVector)::X where {X <: SMatrix}
    return vcat(jacobian, transpose(row))
end

"""`[-residual; -offset]` in `rhs`: in place, or a new static vector."""
function negated_stack!!(rhs::V, residual::AbstractVector{E}, offset::E)::V where {E <: Real, V <: AbstractVector{E}}
    @views rhs[1:length(residual)] .= .-residual
    return set_element!!(rhs, lastindex(rhs), -offset)
end

function negated_stack!!(::V, residual::SVector{M, E}, offset::E)::V where {M, E <: Real, V <: SVector{<:Any, E}}
    return vcat(-residual, SVector(-offset))
end

"""
The sign of `det([J; tᵀ])` under `check`, for `J = work.jacobian` and the tangent `t` that the bordered solve just put
in `work.candidate`; zero without a check. Call it only from `tangent_solve!!`, which holds that solve.
"""
orientation_sign!(::Any, ::Nothing, ::Workspace)::Int8 = Int8(0)
orientation_sign!(linear::CachedSolve, ::DeterminantSign, work::Workspace)::Int8 = solved_sign!(linear, linear.cache.cacheval, work)

orientation_sign!(linear::StaticSolve, ::DeterminantSign, ::Workspace)::Int8 = factored_sign(linear.factorization)

"""
The sign from the bordered solve's own LU of `[J; rᵀ]`, as `StaticLU`'s. With `rᵀt = 1`, the matrix determinant lemma
gives `det([J; tᵀ]) = det([J; rᵀ]) · tᵀt`, so both signs agree.
"""
solved_sign!(::CachedSolve, solved::LinearAlgebra.LU, ::Workspace)::Int8 = factored_sign(solved.factors, solved.ipiv)

"""The sign from a second factorization, of `[J; tᵀ]` itself, for a linear solver that keeps no LU."""
function solved_sign!(linear::CachedSolve, ::Any, work::Workspace)::Int8
    oriented = border!!(linear.oriented, work.jacobian, work.candidate)
    (factors, pivots) = factor!(oriented, linear.pivots)
    return factored_sign(factors, pivots)
end

"""`matrix`'s pivoted LU factors in place and its row swaps: LAPACK's `getrf!` for BLAS floats, else `lu!`."""
function factor!(
        matrix::Matrix{E},
        pivots::Vector{LinearAlgebra.BlasInt},
    )::Tuple{Matrix{E}, Vector{LinearAlgebra.BlasInt}} where {E <: LinearAlgebra.BlasReal}
    LinearAlgebra.LAPACK.getrf!(matrix, pivots; check = false)
    return (matrix, pivots)
end

function factor!(matrix::AbstractMatrix{E}, ::Vector{LinearAlgebra.BlasInt})::Tuple{AbstractMatrix{E}, AbstractVector{<:Integer}} where {E <: Real}
    factored = LinearAlgebra.lu!(matrix; check = false)
    return (factored.factors, factored.ipiv)
end

"""
The sign of a determinant from its pivoted LU `factors` and row swaps `pivots`; zero for a zero or NaN pivot. Two
reductions, so the factors stay on their device.
"""
function factored_sign(factors::AbstractMatrix{E}, pivots::AbstractVector{<:Integer})::Int8 where {E <: Real}
    diagonal = view(factors, LinearAlgebra.diagind(factors, IndexCartesian()))
    odd_swaps = mapreduce(identity, xor, elementwise(!=, pivots, eachindex(pivots)); init = false)
    return odd_swaps ? -pivot_signs(diagonal) : pivot_signs(diagonal)
end

"""The sign of the determinant that `factorization`, StaticArrays' LU, factors; zero for a zero or NaN pivot."""
function factored_sign(factorization::StaticArrays.LU)::Int8
    signs = pivot_signs(LinearAlgebra.diag(parent(factorization.U)))
    return odd_permutation(factorization.p) ? -signs : signs
end

"""The product of the signs of `pivots`, a factorization's diagonal; zero for a zero or NaN pivot."""
function pivot_signs(pivots::AbstractVector{<:Real})::Int8
    return mapreduce(pivot_sign, *, pivots; init = Int8(1))
end

"""The sign of `pivot`: zero for zero or NaN."""
pivot_sign(pivot::Real)::Int8 = Int8(pivot > 0) - Int8(pivot < 0)

"""Whether `permutation` is odd: it has an odd count of inversions, pairs `i < j` with `permutation[i] > permutation[j]`."""
function odd_permutation(permutation::StaticVector{N, <:Integer})::Bool where {N}
    inversions = 0
    for later in 2:N, earlier in 1:(later - 1)
        inversions += permutation[earlier] > permutation[later]
    end
    return isodd(inversions)
end
