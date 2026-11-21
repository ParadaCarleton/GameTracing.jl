using ErrorTypes: unwrap
using LinearAlgebra: Diagonal, I, det, dot, norm, opnorm, pinv
using PathFollowing: PathFollowing
using Random: AbstractRNG, RandomDevice, Xoshiro
using StaticArrays: SArray

"""A generator on a fresh seed, logged so a failing draw can be replayed with `Xoshiro(seed)`."""
function fresh_rng()::Xoshiro
    seed = rand(RandomDevice(), UInt)
    @info "fresh draw" seed
    return Xoshiro(seed)
end

"""
The dense operator `state_weight JᵀGJ + precision_weight e eᵀ` of the Fisher distance on displacements `[Δw; Δλ]` at
`point = [w; λ]`. `G` is the Fisher information of the logits `z = λ w`, one block `diag(q) - q qᵀ` per player of
`ranges`; `J = [λI w]` is the logits' Jacobian; `e` is the precision's unit vector.
"""
function fisher_operator(ranges::Any, point::AbstractVector{E}, state_weight::E, precision_weight::E)::Matrix{E} where {E <: Real}
    width = length(point) - 1
    gaps = point[1:width]
    information = zeros(E, width, width)
    for range in ranges
        logits = last(point) .* gaps[range]
        shift = max(zero(E), maximum(logits))
        weights = exp.(logits .- shift)
        probabilities = weights ./ (exp(-shift) + sum(weights))
        information[range, range] .= Diagonal(probabilities) .- probabilities .* probabilities'
    end
    jacobian = [last(point) .* Matrix{E}(I, width, width) gaps]
    operator = state_weight .* (jacobian' * information * jacobian)
    operator[end, end] += precision_weight
    return operator
end

"""Every `TraceGeometry` at state weight `state_weight`: each plain distance, and the Fisher distance with each precision term."""
function every_geometry(state_weight::E)::Vector{TraceGeometry{E}} where {E <: Real}
    plain = [TraceGeometry{E}(; distance, state_weight) for distance in (AveragedDistance, SummedDistance)]
    return [plain; [TraceGeometry{E}(; distance = FisherDistance, state_weight, precision_term) for precision_term in instances(PrecisionTerm)]]
end

"""`frozen`, a frozen PathFollowing metric over points of `width` coordinates, as a dense matrix of `metric_inner`."""
function inner_matrix(frozen::Any, ::Type{E}, width::Integer)::Matrix{E} where {E <: Real}
    units = [setindex!(zeros(E, width), one(E), index) for index in 1:width]
    return [PathFollowing.metric_inner(frozen, left, right) for left in units, right in units]
end

"""`frozen` as a dense matrix of `metric_apply!!`, one column per unit vector."""
function applied_matrix(frozen::Any, ::Type{E}, width::Integer)::Matrix{E} where {E <: Real}
    applied = zeros(E, width, width)
    for index in 1:width
        PathFollowing.metric_apply!!(@view(applied[:, index]), frozen, setindex!(zeros(E, width), one(E), index))
    end
    return applied
end

"""Payoff gains `G = A p + b` with `p = logistic(logits)`, independent of the precision: the logit map `λ (A p + b)`."""
struct LinearGainPass{E <: Real}
    interaction::Matrix{E}
    stand_alone::Vector{E}
end

# Each logit is one player's choice between two strategies.
GameTracing.logit_ranges(pass::LinearGainPass)::Vector{UnitRange{Int}} = UnitRange.(eachindex(pass.stand_alone), eachindex(pass.stand_alone))
# Each gain is its stand-alone term plus interaction weights on probabilities in (0, 1).
GameTracing.gap_bound(pass::LinearGainPass)::Real = maximum(abs.(pass.stand_alone) .+ vec(sum(abs, pass.interaction; dims = 2)))

function (pass::LinearGainPass)(
        gain::G,
        gain_slope::S,
        gain_precision_slope::P,
        logits::AbstractVector{E},
        ::Real,
    )::Tuple{G, S, P} where {E <: Real, G <: AbstractVector, S <: AbstractMatrix, P <: AbstractVector}
    stays = inv.(one(E) .+ exp.(-logits))
    gain .= pass.interaction * stays .+ pass.stand_alone
    gain_slope .= pass.interaction .* transpose(stays .* (one(E) .- stays))
    fill!(gain_precision_slope, zero(E))
    return gain, gain_slope, gain_precision_slope
end

"""One logit that gains `amplitude * logistic(logits - threshold)`: a coordination game whose branch folds back when `threshold` is large."""
struct ThresholdPass{E <: Real}
    amplitude::E
    threshold::E
end

GameTracing.logit_ranges(::ThresholdPass)::Tuple{UnitRange{Int}} = (1:1,)
GameTracing.gap_bound(pass::ThresholdPass)::Real = abs(pass.amplitude)

function (pass::ThresholdPass)(
        gain::G,
        gain_slope::S,
        gain_precision_slope::P,
        logits::AbstractVector{E},
        ::Real,
    )::Tuple{G, S, P} where {E <: Real, G <: AbstractVector, S <: AbstractMatrix, P <: AbstractVector}
    stays = inv.(one(E) .+ exp.(pass.threshold .- logits))
    gain .= pass.amplitude .* stays
    gain_slope .= reshape(pass.amplitude .* stays .* (one(E) .- stays), 1, 1)
    fill!(gain_precision_slope, zero(E))
    return gain, gain_slope, gain_precision_slope
end

"""A `LinearGainPass` whose gains shrink with the precision: `G = (A p + b) / sqrt(1 + (λ s)²)`, so `∂G/∂λ ≠ 0`."""
struct DampedGainPass{E <: Real}
    pass::LinearGainPass{E}
    spread::E
end

GameTracing.logit_ranges(pass::DampedGainPass)::Vector{UnitRange{Int}} = GameTracing.logit_ranges(pass.pass)
# The damping is at most one.
GameTracing.gap_bound(pass::DampedGainPass)::Real = GameTracing.gap_bound(pass.pass)

function (pass::DampedGainPass)(
        gain::G,
        gain_slope::S,
        gain_precision_slope::P,
        logits::AbstractVector{E},
        precision::Real,
    )::Tuple{G, S, P} where {E <: Real, G <: AbstractVector, S <: AbstractMatrix, P <: AbstractVector}
    gain, gain_slope, gain_precision_slope = pass.pass(gain, gain_slope, gain_precision_slope, logits, precision)
    damping = inv(sqrt(1 + (precision * pass.spread)^2))
    gain_precision_slope .= -precision * pass.spread^2 * damping^2 .* damping .* gain
    gain .*= damping
    gain_slope .*= damping
    return gain, gain_slope, gain_precision_slope
end

"""One logit whose gain is the precision itself, `G = λ`: the branch `z = λ²`."""
struct PrecisionGainPass end

GameTracing.logit_ranges(::PrecisionGainPass)::Tuple{UnitRange{Int}} = (1:1,)
# The gain `λ` reaches one at the opening step this bound sets.
GameTracing.gap_bound(::PrecisionGainPass)::Real = 1

function (::PrecisionGainPass)(
        gain::G,
        gain_slope::S,
        gain_precision_slope::P,
        ::AbstractVector{E},
        precision::Real,
    )::Tuple{G, S, P} where {E <: Real, G <: AbstractVector, S <: AbstractMatrix, P <: AbstractVector}
    fill!(gain, E(precision))
    fill!(gain_slope, zero(E))
    fill!(gain_precision_slope, one(E))
    return gain, gain_slope, gain_precision_slope
end

"""
One logit with a flat gain `level`, plus deterministic noise of size `noise` past the precision `onset`: a uniform draw
in `[-1, 1)` hashed from the logit's bits and `salt`, which the slope does not see. Newton's corrections there are
`λ noise` times a difference of two draws. One rounding change redraws the whole stream, so test several salts.
"""
struct NoisyFlatPass{E <: Real}
    level::E
    noise::E
    onset::E
    salt::UInt
end

GameTracing.logit_ranges(::NoisyFlatPass)::Tuple{UnitRange{Int}} = (1:1,)
GameTracing.gap_bound(pass::NoisyFlatPass)::Real = abs(pass.level) + abs(pass.noise)

function (pass::NoisyFlatPass)(
        gain::G,
        gain_slope::S,
        gain_precision_slope::P,
        logits::AbstractVector{E},
        precision::Real,
    )::Tuple{G, S, P} where {E <: Real, G <: AbstractVector, S <: AbstractMatrix, P <: AbstractVector}
    draw = (E(hash(only(logits), pass.salt) % 65536) / 32768 - one(E)) * (precision > pass.onset)
    fill!(gain, pass.level + pass.noise * draw)
    fill!(gain_slope, zero(E))
    fill!(gain_precision_slope, zero(E))
    return gain, gain_slope, gain_precision_slope
end

"""A pass whose gain and slopes are NaN once the precision passes `poison`, as a game that overflows there is."""
struct PoisonedPass{P, E <: Real}
    pass::P
    poison::E
end

GameTracing.logit_ranges(pass::PoisonedPass)::Union{Tuple, AbstractVector} = GameTracing.logit_ranges(pass.pass)
GameTracing.gap_bound(pass::PoisonedPass)::Real = GameTracing.gap_bound(pass.pass)

function (pass::PoisonedPass)(
        gain::G,
        gain_slope::S,
        gain_precision_slope::P,
        logits::AbstractVector,
        precision::Real,
    )::Tuple{G, S, P} where {G <: AbstractVector, S <: AbstractMatrix, P <: AbstractVector}
    gain, gain_slope, gain_precision_slope = pass.pass(gain, gain_slope, gain_precision_slope, logits, precision)
    if precision > pass.poison
        fill!(gain, NaN32)
        fill!(gain_slope, NaN32)
        fill!(gain_precision_slope, NaN32)
    end
    return gain, gain_slope, gain_precision_slope
end

"""A pass that gives a NaN slope at precision `from` and above, so the Jacobian is non-finite where the residual is not."""
struct NaNSlopePass{P, F <: Real}
    pass::P
    from::F
end

GameTracing.logit_ranges(pass::NaNSlopePass)::Union{Tuple, AbstractVector} = GameTracing.logit_ranges(pass.pass)
GameTracing.gap_bound(pass::NaNSlopePass)::Real = GameTracing.gap_bound(pass.pass)

function (pass::NaNSlopePass)(
        gain::G,
        gain_slope::S,
        gain_precision_slope::P,
        logits::AbstractVector,
        precision::Real,
    )::Tuple{G, S, P} where {G <: AbstractVector, S <: AbstractMatrix, P <: AbstractVector}
    gain, gain_slope, gain_precision_slope = pass.pass(gain, gain_slope, gain_precision_slope, logits, precision)
    precision >= pass.from && fill!(gain_slope, NaN32)
    return gain, gain_slope, gain_precision_slope
end

"""A pass that throws a caller's `ArgumentError` once the precision is positive."""
struct ThrowingPass{P}
    pass::P
end

GameTracing.logit_ranges(pass::ThrowingPass)::Union{Tuple, AbstractVector} = GameTracing.logit_ranges(pass.pass)
GameTracing.gap_bound(pass::ThrowingPass)::Real = GameTracing.gap_bound(pass.pass)

function (pass::ThrowingPass)(
        gain::G,
        gain_slope::S,
        gain_precision_slope::P,
        logits::AbstractVector,
        precision::Real,
    )::Tuple{G, S, P} where {G <: AbstractVector, S <: AbstractMatrix, P <: AbstractVector}
    if precision > 0
        throw(ArgumentError("bad game"))
    end
    return pass.pass(gain, gain_slope, gain_precision_slope, logits, precision)
end

"""
The logit map `T = λ G` of `pass` at `logits` and precision `λ`, in fresh arrays: `(; response, logit_slope,
precision_slope)`, which are `T`, `∂T/∂z = λ ∂G/∂z` and `∂T/∂λ = G + λ ∂G/∂λ`.
"""
function evaluated(pass::Any, logits::AbstractVector{E}, precision::Real)::NamedTuple where {E <: Real}
    width = length(logits)
    blank = (Vector{E}(undef, width), Matrix{E}(undef, width, width), Vector{E}(undef, width))
    gain, gain_slope, gain_precision_slope = pass(blank..., logits, precision)
    return (;
        response = precision .* gain,
        logit_slope = precision .* gain_slope,
        precision_slope = gain .+ precision .* gain_precision_slope,
    )
end

"""The largest `|logits - T(logits, precision)|` entry: zero at an equilibrium."""
residual_norm(pass::Any, logits::AbstractVector, precision::Real) = maximum(
    abs,
    logits .- evaluated(pass, logits, precision).response,
)

"""Picard iterates of `logits -> T(logits, precision)` from the flat start: the equilibrium when `T` contracts."""
function picard_equilibrium(
        pass::Any,
        width::Integer,
        precision::E,
        rounds::Integer,
    )::Vector{E} where {E <: Real}
    logits = zeros(E, width)
    for _ in 1:rounds
        logits = evaluated(pass, logits, precision).response
    end
    return logits
end

"""`pass`'s game under a type that defines `logit_ranges` but no `gap_bound`, as a pass written before it existed."""
struct UnboundedPass{P}
    pass::P
end

(wrapper::UnboundedPass)(arguments::Any...)::Tuple = wrapper.pass(arguments...)
GameTracing.logit_ranges(wrapper::UnboundedPass)::Any = GameTracing.logit_ranges(wrapper.pass)

"""
Payoff gains of a normal-form game in which every player has two actions: `G_p = U_p(1, π) - U_p(2, π)`,
each rival `q` playing its first action with probability `logistic(z_q)`. `payoffs[p][profile]` is player `p`'s payoff.
Payoffs with another action count throw an `ArgumentError`.
"""
struct NormalFormPass{N, A <: AbstractArray{<:Real, N}}
    payoffs::NTuple{N, A}

    function NormalFormPass(payoffs::NTuple{N, A})::NormalFormPass{N, A} where {N, A <: AbstractArray{<:Real, N}}
        if any(table -> any(!=(2), size(table)), payoffs)
            throw(ArgumentError("NormalFormPass takes two actions per player; got sizes $(size.(payoffs))"))
        end
        return new{N, A}(payoffs)
    end
end

# Each player has two actions, so one logit.
GameTracing.logit_ranges(pass::NormalFormPass)::Vector{UnitRange{Int}} = UnitRange.(eachindex(pass.payoffs), eachindex(pass.payoffs))
# A gap is one payoff less another.
GameTracing.gap_bound(pass::NormalFormPass)::Real = maximum(maximum, pass.payoffs) - minimum(minimum, pass.payoffs)

"""The probability that a player whose first action has probability `first_choice` plays `action`."""
function action_probability(first_choice::E, action::Integer)::E where {E <: Real}
    return ifelse(action == 1, first_choice, 1 - first_choice)
end

"""`+1` for a player's first action, `-1` for its second: the sign it carries in a payoff gain."""
action_sign(action::Integer)::Int = 3 - 2 * action

function (pass::NormalFormPass{N})(
        gain::G,
        gain_slope::S,
        gain_precision_slope::P,
        logits::AbstractVector{E},
        ::Real,
    )::Tuple{G, S, P} where {N, E <: Real, G <: AbstractVector, S <: AbstractMatrix, P <: AbstractVector}
    first_choices = inv.(one(E) .+ exp.(-logits))
    fill!(gain, zero(E))
    fill!(gain_slope, zero(E))
    fill!(gain_precision_slope, zero(E))
    for profile in CartesianIndices(first(pass.payoffs)), (player, payoff) in pairs(pass.payoffs)
        signed = action_sign(profile[player]) * payoff[profile]
        chances = [action_probability(first_choices[rival], profile[rival]) for rival in 1:N]
        gain[player] += signed * prod((chances[rival] for rival in 1:N if rival != player); init = one(E))
        for rival in 1:N
            if rival != player
                others = prod((chances[other] for other in 1:N if other != player && other != rival); init = one(E))
                gain_slope[player, rival] += signed * action_sign(profile[rival]) * others
            end
        end
    end
    gain_slope .*= transpose(first_choices .* (one(E) .- first_choices))
    return gain, gain_slope, gain_precision_slope
end

"""Whether `logits` solves `pass` at `precision` to `Float64` rounding at `scale`, times the logits' size."""
function settled_root(pass::Any, logits::AbstractVector{E}, precision::E, scale::E)::Bool where {E <: Real}
    return residual_norm(pass, logits, precision) <= 10^4 * eps(scale) * (1 + maximum(abs, logits))
end

"""
    level_root(pass, logits, precision)

The root of `pass` at fixed `precision` that `Float64` Newton reaches from `logits`; an `ErrorException` when 50 steps do not settle.
"""
function level_root(pass::Any, logits::AbstractVector, precision::Real)::AbstractVector
    root, level = float.(collect(logits)) .* float(1), precision * float(1)
    for _ in 1:50
        if settled_root(pass, root, level, max(level, one(level)))
            return root
        end
        gains = evaluated(pass, root, level)
        root = root .- (one(gains.logit_slope) .- gains.logit_slope) \ (root .- gains.response)
    end
    error("level_root: Newton did not settle at precision $level")
end

"""
One double-precision Newton correction of `pass`'s map `z = T(z, λ)` from `logits` at `precision`: to first order, the
move from `logits` to the root.
"""
function newton_step(pass::Any, logits::AbstractVector, precision::Real)::Vector
    root, level = float.(collect(logits)) .* float(1), precision * float(1)
    gains = evaluated(pass, root, level)
    return (one(gains.logit_slope) .- gains.logit_slope) \ (root .- gains.response)
end

"""The largest logit move of `newton_step`: to first order, how far `logits` lie from the root."""
function newton_correction(pass::Any, logits::AbstractVector, precision::P)::typeof(float(one(P)) * float(1)) where {P <: Real}
    return maximum(abs, newton_step(pass, logits, precision))
end

"""
The logit part of one double-precision Newton correction of the branch equations of `pass` at `[logits; precision]`,
with the precision free: the least-norm step (`pinv`) that a corrector bordered by the branch's tangent takes. At a fold
`I - ∂T/∂z` is singular and `newton_step` is unbounded; this step is not.
"""
function branch_step(pass::Any, logits::AbstractVector, precision::Real)::Vector
    (; residual, jacobian) = reference_system(pass, [float.(collect(logits)) .* float(1); precision * float(1)])
    return (pinv(jacobian) * residual)[1:(end - 1)]
end

"""
Whether branch point `logits` solve `gains` at `precision` as PathFollowing's `StrategyDistance` corrector settles them:
`branch_step` moves the profile (`StrategyChange`) no more than twice the settle tests' bound. The Fisher–Rao test
allows the tolerance plus `settle_ulps` ulps of the logits' spread (`FisherRaoSettled`). Gambit's norm, over √2,
allows its tolerance plus each strategy's allowance (`gambit_allowance`). Large Float32 logits round past the tolerance alone.
"""
function settled_strategies(gains::NormalFormGains, logits::AbstractVector, precision::Real, distance::StrategyDistance{E})::Bool where {E <: Real}
    root, level = float.(collect(logits)) .* float(1), precision * float(1)
    corrected = root .- branch_step(gains, logits, precision)
    measure = GameTracing.corrector_measure(distance, gains, logit_count(gains))
    # The measure reads only the logits `z = λ w`, so `[z; 1]` serves at every precision, zero included.
    point = [root; one(level)]
    ranges = GameTracing.logit_ranges(gains)
    relative = distance.settle_ulps * eps(E)
    spread = sqrt(sum(range -> last(GameTracing.fisher_rao_spread(point, point, range))^2, ranges))
    fisher_rao = distance.tolerance + relative * spread
    gambit = (distance.log_probability_tolerance + gambit_allowance(point, ranges, relative)) / sqrt(2)
    return measure(point, [corrected; one(level)]) <= 2 * max(fisher_rao, gambit)
end

"""
The root sum of squares of every strategy's rounding allowance in Gambit's norm at `point = [z; 1]`, `relative` times
its rounding scale (`log_probability_corrections`), over the players of `ranges`.
"""
function gambit_allowance(point::AbstractVector, ranges::Any, relative::Real)::Real
    squared = zero(float(relative))
    for range in ranges
        normalizer = GameTracing.log_normalizer(point, range)
        shared = GameTracing.normalizer_rounding(point, range, normalizer)
        squared += (relative * (GameTracing.centered_rounding(point, 0, first(normalizer)) + shared))^2
        for index in range
            squared += (relative * (GameTracing.centered_rounding(point, index, first(normalizer)) + shared))^2
        end
    end
    return sqrt(squared)
end

"""
The largest last correction, in Gambit's norm, a settled corrector leaves at `logits`: `logit_tolerance` plus twice the
root sum of squares of its allowances, as a step may land on logits larger than the ones it left. Each strategy's
allowance is at most `settle_ulps` ulps of three times the largest logit, and the `n` logits have at most `2n` strategies.
"""
function settle_bound(corrector::CorrectorSettings{E}, logits::AbstractVector)::E where {E <: Real}
    largest = isempty(logits) ? zero(E) : E(maximum(abs, logits))
    return corrector.logit_tolerance + 2 * corrector.settle_ulps * eps(E) * 3 * largest * sqrt(E(2 * length(logits)))
end

"""Gambit's norm of the move `step` from `logits` (`LogProbabilityChange`), over `pass`'s players, in double precision."""
function gambit_move(pass::Any, logits::AbstractVector, step::AbstractVector)::Real
    point = [float.(collect(logits)) .* float(1); float(1)]
    return GameTracing.log_probability_change(point, point .- [step; zero(float(1))], GameTracing.logit_ranges(pass))
end

"""
The largest Newton correction, in logits, that PathFollowing's corrector leaves settled under `distance` at the logits
`logits` of `pass`'s game, to first order. The profile settles once it moves at most `tolerance` plus `settle_ulps` ulps
of the root sum of squares of its logit spreads (`FisherRaoSettled`), doubled as in `settled_strategies`. Each player's
move `d` is at most that. With the first logit fixed it moves logit `j` at most `d √(1/πⱼ + 1/π₀)`.
"""
function settle_bound(distance::StrategyDistance{E}, pass::Any, logits::AbstractVector)::E where {E <: Real}
    # The probabilities read only the logits `z = λ w`, so `[z; 1]` serves at every precision.
    point = [float.(collect(logits)) .* float(1); float(1)]
    largest = zero(last(point))
    spread = sqrt(sum(range -> last(GameTracing.fisher_rao_spread(point, point, range))^2, GameTracing.logit_ranges(pass)))
    move = 2 * (distance.tolerance + distance.settle_ulps * eps(E) * spread)
    for range in GameTracing.logit_ranges(pass)
        normalizer = GameTracing.log_normalizer(point, range)
        first_share = exp(-GameTracing.log_probability(point, 0, normalizer))
        for index in range
            largest = max(largest, move * sqrt(exp(-GameTracing.log_probability(point, index, normalizer)) + first_share))
        end
    end
    return E(largest)
end

"""Whether `logits` solve `pass` at `precision` as the trace's corrector settles them: `newton_step` within `settle_bound` (`gambit_move`)."""
function settled_logits(pass::Any, logits::AbstractVector, precision::Real, settings::BifurcationKitSettings)::Bool
    return gambit_move(pass, logits, newton_step(pass, logits, precision)) <= settle_bound(settings.corrector, logits)
end

"""Whether branch point `logits` solve `pass` at `precision` as the trace's corrector settles them: `branch_step` within `settle_bound` (`gambit_move`)."""
function settled_branch_logits(pass::Any, logits::AbstractVector, precision::Real, settings::BifurcationKitSettings)::Bool
    return gambit_move(pass, logits, branch_step(pass, logits, precision)) <= settle_bound(settings.corrector, logits)
end

"""`(; residual, jacobian)` of the branch equations `z - T(z, λ) = 0` of `pass` at `point = [z; λ]`."""
function reference_system(pass::Any, point::AbstractVector{E})::NamedTuple where {E <: Real}
    logits, level = point[1:(end - 1)], last(point)
    gains = evaluated(pass, logits, level)
    return (; residual = logits .- gains.response, jacobian = [one(gains.logit_slope) .- gains.logit_slope -gains.precision_slope])
end

"""The unit tangent of the branch of `pass` at `point`, oriented along `previous`."""
function reference_tangent(pass::Any, point::AbstractVector{E}, previous::AbstractVector{E})::Vector{E} where {E <: Real}
    jacobian = reference_system(pass, point).jacobian
    tangent = [jacobian; transpose(previous)] \ [zeros(E, size(jacobian, 1)); one(E)]
    return tangent ./ norm(tangent)
end

"""The sign of `det(∂F/∂z)` times that of `dλ/ds`: constant along a smooth branch, through folds included."""
function reference_orientation(pass::Any, point::AbstractVector{E}, tangent::AbstractVector{E})::E where {E <: Real}
    jacobian = reference_system(pass, point).jacobian
    return sign(det(jacobian[:, 1:(end - 1)])) * sign(last(tangent))
end

"""
`(; settled, point)`: Newton on the branch equations and the arclength condition `τ ⋅ (y - point) = step`, from the
predictor `point + step τ`; `settled` when it solves the map to `settled_root` at `scale` within `step / 10` of the predictor.
"""
function corrected_point(
        pass::Any,
        point::AbstractVector{E},
        tangent::AbstractVector{E},
        step::E,
        scale::E,
    )::NamedTuple where {E <: Real}
    predicted = point .+ step .* tangent
    candidate = copy(predicted)
    for _ in 1:6
        system = reference_system(pass, candidate)
        candidate = candidate .- [system.jacobian; transpose(tangent)] \ [system.residual; dot(tangent, candidate .- point) - step]
        if settled_root(pass, candidate[1:(end - 1)], last(candidate), scale)
            return (; settled = norm(candidate .- predicted) <= step / 10, point = candidate)
        end
    end
    return (; settled = false, point = candidate)
end

"""
    branch_reference(pass, width, precision; start, start_precision)

`(; reached, logits, first_choices, crossings)`: an independent reference for the branch of `pass` at `precision`, by pseudo-arclength
continuation in `Float64` over `[z; λ]` from the root `level_root` reaches from `start` at `start_precision` (default:
zero logits at zero precision, the principal branch). It ends where λ first reaches `precision`, through folds included.

A step of arclength at most `1/100` halves until its corrector settles near the predictor, the tangent turns under
`acos(0.99)`, and `reference_orientation` keeps its sign. A sign change marks a step onto another sheet: on
`UNFOLDED_PITCHFORK_GAME` the principal branch turns steeply past the middle arm, which a step in λ alone lands on.
A sign change that persists to a step of `1e-6` is taken as a crossing of a singular point, as at an exact pitchfork;
`crossings` lists the precision of the last point before each, and a generic game's branch has none. `reached = false`
when the step underflows or `10^6` steps do not reach `precision`.
"""
function branch_reference(
        pass::Any,
        width::Integer,
        precision::Real;
        start::AbstractVector = zeros(width),
        start_precision::Real = 0,
    )::NamedTuple
    # `float(1)` widens a 32-bit precision, which no Float64 literal names.
    target = precision * float(1)
    point = [level_root(pass, start, start_precision); start_precision * float(1)]
    tangent = reference_tangent(pass, point, [zeros(width); one(target)])
    side = reference_orientation(pass, point, tangent)
    step, largest, singular_step = inv(float(1000)), inv(float(100)), inv(float(10^6))
    crossings = typeof(target)[]
    for _ in 1:(10^6)
        if !(last(point) < target) || !(step > eps(target))
            break
        end
        corrected = corrected_point(pass, point, tangent, step, target)
        if corrected.settled
            next_tangent = reference_tangent(pass, corrected.point, tangent)
            next_side = reference_orientation(pass, corrected.point, next_tangent)
            if dot(next_tangent, tangent) >= 99 // 100 && (next_side == side || step <= singular_step)
                if next_side != side
                    push!(crossings, last(point))
                end
                if !(last(corrected.point) < target)
                    # The step brackets the target: Newton at the target from the bracket's interpolation.
                    weight = (target - last(point)) / (last(corrected.point) - last(point))
                    logits = level_root(pass, point[1:width] .+ weight .* (corrected.point[1:width] .- point[1:width]), target)
                    return (; reached = true, logits, first_choices = inv.(1 .+ exp.(-logits)), crossings)
                end
                point, tangent, side, step = corrected.point, next_tangent, next_side, min(2 * step, largest)
                continue
            end
        end
        step /= 2
    end
    logits = point[1:width]
    return (; reached = last(point) == target, logits, first_choices = inv.(1 .+ exp.(-logits)), crossings)
end

"""
    MapSwap(before, after, precision, swap_precision, width)

A trace over `width` logits to `precision` whose game changes from `before` to `after` once the trace reaches
precision `swap_precision`: a redraw of the map at an accepted point.
"""
struct MapSwap{B, A, E <: Real}
    before::B
    after::A
    precision::E
    swap_precision::E
    width::Int
end

# The swap moves a two-player branch by under a tenth: a redraw a fixed-precision correction absorbs.
const NEAR_SWAP = MapSwap(
    LinearGainPass([0 0.8f0; 0.6f0 0], [0.3f0, -0.2f0]),
    LinearGainPass([0 0.8f0; 0.6f0 0], [0.34f0, -0.23f0]),
    4.0f0, 2.0f0, 2,
)

# The swap removes the low branch at precision 1/2: the only root there is the high one, about 3.5 logits away,
# and the homotopy between the maps reaches it with `s` rising throughout (`rising_homotopy`).
const FAR_SWAP = MapSwap(ThresholdPass(8.0f0, 4.0f0), ThresholdPass(8.0f0, 1.5f0), 1.0f0, 0.5f0, 1)

# The new map's root at precision 1/2 lies past a fold of the homotopy: the blended map loses the low root near s = 0.44.
const FOLD_SWAP = MapSwap(ThresholdPass(8.0f0, 4.0f0), ThresholdPass(16.0f0, 2.0f0), 1.0f0, 0.5f0, 1)

# Two symmetric anti-coordination maps at precision 2: the blend's symmetric root keeps its symmetric mode positive
# while its antisymmetric mode changes sign, a pitchfork with no fold (`symmetric_modes`).
const PITCHFORK_SWAP = MapSwap(
    LinearGainPass([0 -2.0f0; -2.0f0 0], [0.0f0, 0.0f0]),
    LinearGainPass([0 -8.0f0; -8.0f0 0], [0.0f0, 0.0f0]),
    4.0f0, 2.0f0, 2,
)

"""
    symmetric_modes(interaction, precision)

`(; symmetric, antisymmetric)`: the two eigenvalues of `I - λ ∂G/∂z` at the symmetric root of `z = λ a σ(z)` on both
logits of a two-player map with off-diagonal interaction `a`, found by Float64 Newton from zero: `1 ∓ λ a σ'(z)`.
"""
function symmetric_modes(interaction::Real, precision::Real)::NamedTuple
    pass = LinearGainPass([0 interaction; interaction 0] .* float(1), zeros(2))
    root = level_root(pass, zeros(2), precision)
    slope = inv(1 + exp(-first(root))) * inv(1 + exp(first(root)))
    return (; symmetric = 1 - precision * interaction * slope, antisymmetric = 1 + precision * interaction * slope)
end

"""`pass`'s gain at the logit `logit`, in Float64."""
threshold_gain(pass::ThresholdPass, logit::Real)::typeof(logit * float(1)) = pass.amplitude * float(1) / (1 + exp(pass.threshold * float(1) - logit))

"""
    rising_homotopy(before, after, precision, logit)

Whether the homotopy `z = λ ((1 - s) G₀(z) + s G₁(z))` from `before`'s root `logit` at precision `λ` reaches `s = 1`
with `s` rising: solved for `s`, the path is `s(z) = (z / λ - G₀) / (G₁ - G₀)`, here sampled upward in `z` from
`logit` in Float64 steps of 1e-4. A fall in `s` is a fold. Each map is one logit, with `G₁ > G₀` on the path.
"""
function rising_homotopy(before::ThresholdPass, after::ThresholdPass, precision::Real, logit::Real)::Bool
    # `float(1)` widens to Float64; the path starts at `s ≈ 0`, so any first sample rises.
    level, blend, logit_step = precision * float(1), typemin(float(1)), inv(float(10^4))
    for logit_now in range(logit * float(1); step = logit_step, length = 10^6)
        spread = threshold_gain(after, logit_now) - threshold_gain(before, logit_now)
        if !(spread > 0)
            throw(ArgumentError("the gains do not order on the path at logit $logit_now"))
        end
        next = (logit_now / level - threshold_gain(before, logit_now)) / spread
        if next >= 1
            return true
        end
        if next < blend
            return false
        end
        blend = next
    end
    throw(ArgumentError("the path does not reach s = 1 within 100 logits"))
end

"""
    SwapAt(game, swap_precision)

A `redraw` hook that returns `game` at every accepted point from precision `swap_precision` on, and `nothing`
before: the trace swaps once, since returning its current game swaps nothing.
"""
struct SwapAt{G, E <: Real}
    game::G
    swap_precision::E
end

function (hook::SwapAt{G})(::AbstractVector, precision::Real, ::Any)::Union{Nothing, G} where {G}
    if precision >= hook.swap_precision
        return hook.game
    end
    return nothing
end

"""A `keep_going` hook that stops the trace at its first accepted point at or past `swap_precision`."""
struct StopAt{E <: Real}
    swap_precision::E
end

(hook::StopAt)(::AbstractVector, precision::Real)::Bool = precision < hook.swap_precision

"""
A `NormalFormGains` pass that counts in `builds` the work arrays it hands out (`pass_workspace`). It has no call
without work arrays, so an evaluation that skips them throws a `MethodError`.
"""
struct CountingPass{G <: NormalFormGains}
    inner::G
    builds::Base.RefValue{Int}
end

function GameTracing.pass_workspace(pass::CountingPass, logits::AbstractVector)::NamedTuple
    pass.builds[] += 1
    return pass_workspace(pass.inner, logits)
end

function (pass::CountingPass)(
        work::W,
        gain::G,
        gain_slope::S,
        gain_precision_slope::P,
        logits::AbstractVector,
        precision::Real,
    )::Tuple{W, G, S, P} where {W <: NamedTuple, G <: AbstractVector, S <: AbstractMatrix, P <: AbstractVector}
    return pass.inner(work, gain, gain_slope, gain_precision_slope, logits, precision)
end

GameTracing.logit_ranges(pass::CountingPass)::Tuple = logit_ranges(pass.inner)
GameTracing.gap_bound(pass::CountingPass)::Real = gap_bound(pass.inner)

"""`game`'s branch residual `F(point, pin)` from one evaluation on fresh work arrays (`GameResidual`)."""
function branch_residual(game::LogitGame, point::AbstractVector{E}, pin::Real)::Vector{E} where {E <: Real}
    return GameTracing.GameResidual(game, GameTracing.BranchWorkspace(game, point))(point, pin)
end

"""`table` in Float32."""
function single_precision(table::AbstractArray{T, N})::Array{Float32, N} where {T <: Real, N}
    return Float32.(table)
end

"""Draws one random payoff table of scale 3 from `rng` for `players` players with two actions each."""
struct RandomTable{R <: AbstractRNG}
    rng::R
    players::Int
end

(table::RandomTable)(::Integer)::Array = 3 .* randn(table.rng, ntuple(Returns(2), table.players))

"""Moves a payoff table by Gaussian noise of scale 1/20 drawn from `rng`, as a redraw moves a sampled map."""
struct PayoffJitter{R <: AbstractRNG}
    rng::R
end

(jitter::PayoffJitter)(table::AbstractArray)::Array = table .+ randn(jitter.rng, size(table)...) ./ 20

"""`players` random payoff tables of scale 3 from `rng`."""
random_payoffs(rng::AbstractRNG, players::Integer)::Tuple = ntuple(RandomTable(rng, players), players)

"""
    random_swap(seed)

`(; swap, reference_before, reference_after)`: a random 3-player game from seed `seed` whose payoffs move by about 0.05
at half its target precision, as a mid-trace redraw moves a sampled map, and its two maps in Float64.
"""
function random_swap(seed::Integer)::NamedTuple
    rng = Xoshiro(1000 + seed)
    before = random_payoffs(rng, 3)
    after = map(PayoffJitter(rng), before)
    swap = MapSwap(NormalFormPass(map(single_precision, before)), NormalFormPass(map(single_precision, after)), 3.0f0, 1.5f0, 3)
    return (; swap, reference_before = NormalFormPass(before), reference_after = NormalFormPass(after))
end

"""
    probability_bound(pass, logits, precision, residual, slope)

The largest probability error of a root of `pass` at `precision` whose logit residual is `residual`, to first order:
the logit error `‖J⁻¹‖∞ residual`, `J = I - ∂(λG)/∂z` at the root `logits`, plus Float32's 32-ulp rounding of the
logits, times `slope`, the probabilities' largest sup-norm slope in the logits: 1/4 for one logistic logit, 1/2 for a
softmax over several.
"""
function probability_bound(pass::Any, logits::AbstractVector, precision::Real, residual::Real, slope::Real)::typeof(float(residual))
    jacobian = reference_system(pass, [logits; precision * float(1)]).jacobian[:, 1:(end - 1)]
    logit_error = opnorm(inv(jacobian), Inf32) * residual + 32 * eps(Float32) * maximum(abs, logits)
    return slope * logit_error
end

"""
A pass that adds deterministic noise of size `noise` to the gain at precisions in `(0, window)` and none past it: a branch
whose second solve at the opening's secant distance is rounding noise, and whose endpoint is the clean game's.
"""
struct WindowNoisePass{P, E <: Real}
    pass::P
    noise::E
    window::E
end

GameTracing.logit_ranges(pass::WindowNoisePass)::Union{Tuple, AbstractVector} = GameTracing.logit_ranges(pass.pass)
GameTracing.gap_bound(pass::WindowNoisePass)::Real = GameTracing.gap_bound(pass.pass) + abs(pass.noise)

function (pass::WindowNoisePass)(
        gain::G,
        gain_slope::S,
        gain_precision_slope::P,
        logits::AbstractVector{E},
        precision::Real,
    )::Tuple{G, S, P} where {E <: Real, G <: AbstractVector, S <: AbstractMatrix, P <: AbstractVector}
    gain, gain_slope, gain_precision_slope = pass.pass(gain, gain_slope, gain_precision_slope, logits, precision)
    if 0 < precision < pass.window
        for index in eachindex(gain)
            gain[index] += pass.noise * (E(hash((index, sum(logits))) % 65536) / 32768 - one(E))
        end
    end
    return gain, gain_slope, gain_precision_slope
end

"""`payoffs` as a game with static payoffs (`SArray`), as `logit_solve_lambda_batch` takes it."""
function static_game(payoffs::Tuple)::NormalFormGains
    return NormalFormGains(map(payoff -> SArray{Tuple{size(payoff)...}}(payoff), payoffs))
end

"""PathFollowing settings in `element` with the static solver a batch needs, and `options`."""
function batch_settings(element::Type{<:Real}; options...)::PathFollowingSettings
    return PathFollowingSettings{element}(; linear_solver = PathFollowing.StaticLU(), options...)
end

"""PathFollowing's `Tracer` of `gains` at its flat start under `settings`, as `logit_solve` starts it (`path_init`)."""
function started_tracer(gains::NormalFormGains, settings::PathFollowingSettings{E})::PathFollowing.Tracer{E} where {E <: Real}
    problem = GameTracing.path_problem(gains, zeros(E, logit_count(gains)))
    return unwrap(GameTracing.path_init(problem, settings, GameTracing.PositivePrecision(GameTracing.AcceptAll())))
end

"""
PathFollowing's step-control targets before the Gambit-strict defaults: a first correction of 1/4 and a turn of 1 rad.
The jump hunts of the branch tests ran at them; the defaults keep those branches without the checks the tests isolate.
"""
function previous_targets(element::Type)::NamedTuple
    return (; target_turn = 1, distance = StrategyDistance{element}(; target_correction = 1 // 4))
end
