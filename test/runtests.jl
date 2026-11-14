using GameTracing
using Test

include("games.jl")
include("fixtures.jl")
include("arms.jl")
include("gambit.jl")

# A suite past 30 minutes is stuck: a separate process SIGQUITs it, printing every thread's backtrace.
const WATCHDOG = run(`sh -c "timeout 1800 tail --pid=$(getpid()) -f /dev/null || kill -QUIT $(getpid())"`; wait = false)

const GAME = LinearGainPass(Float32[0 1; 1 0], Float32[0.5f0, -0.25f0])
const TARGET_PRECISION = 2.0f0
const WIDTH = 2

# Every `*_tests.jl` file runs, in name order, so none can be left out; the code rules run last. One outer testset
# keeps a failing file from aborting the rest: a failed top-level testset throws as it ends.
# `Pkg.test(; test_args = ["name_tests.jl"])` runs only the named files, then the code rules.
const SELECTED_FILES = isempty(ARGS) ? sort(filter(endswith("_tests.jl"), readdir(@__DIR__))) : ARGS
@testset "GameTracing" begin
    for file in SELECTED_FILES
        include(joinpath(@__DIR__, file))
    end
    include("code_rules.jl")
end
