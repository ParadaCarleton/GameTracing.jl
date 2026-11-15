"""
    GameTracing

Homotopy continuation for logit quantal response equilibria. `trace_endpoint` follows the
branch of a logit game from zero precision (infinite temperature), where the only equilibrium is the flat one,
up to a target precision with BifurcationKit's pseudo-arclength continuation (PALC), across folds. `logit_solve` and
its kin run on BifurcationKit (`BifurcationKitSettings`) or on PathFollowing (`PathFollowingSettings`).
Failures are values (`TraceFailure`), never exceptions.
"""
module GameTracing

export LogitGame,
    NormalFormGains,
    mixed_strategies,
    logit_count,
    pass_workspace,
    LogitQREProfile,
    logit_solve,
    logit_solve_branch,
    logit_solve_lambda,
    EngineSettings,
    BifurcationKitSettings,
    PathFollowingSettings,
    CorrectorDistance,
    StrategyDistance,
    LogitDistance,
    TraceCause,
    TraceFailure,
    Traced,
    TraceResult,
    StepControl,
    CorrectorSettings,
    TraceGeometry,
    StepDistance,
    AveragedDistance,
    SummedDistance,
    FisherDistance,
    FisherLogPrecisionDistance,
    BorderedSolver,
    MatrixSolve,
    BorderingSolve,
    trace_endpoint,
    branch_projection

include("failure.jl")
include("bordered.jl")
include("geometry.jl")
include("branch.jl")
include("normal_form.jl")
include("callbacks.jl")
include("trace.jl")
include("redraw.jl")
include("projection.jl")
include("logit_solve.jl")
include("path_following.jl")

end
