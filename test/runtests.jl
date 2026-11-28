using CondaPkg: CondaPkg
using GameTracing: GameTracing
using ParallelTestRunner: runtests

# Gambit's logit QRE is the test oracle, at the version checked by hand against a 50-digit trace. The pin goes into the
# test environment, not a package-level CondaPkg.toml that every user of GameTracing would read. It resolves here
# once, as workers that each wrote CondaPkg.toml would race.
CondaPkg.add("python"; version = "=3.12", resolve = false)
CondaPkg.add_pip("pygambit"; version = "==16.7.0")

# Each `*_tests.jl` file and the code rules run in a module of their own, spread over worker processes.
# `Pkg.test(; test_args = ["trace", "--jobs=2"])` runs only the files whose names start with `trace`, on 2 workers.
const TEST_FILES = [filter(endswith("_tests.jl"), readdir(@__DIR__)); "code_rules.jl"]
runtests(
    GameTracing, ARGS;
    testsuite = Dict(chopsuffix(file, ".jl") => :(include($(joinpath(@__DIR__, file)))) for file in TEST_FILES),
    init_worker_code = :(include($(joinpath(@__DIR__, "helpers.jl")))),
    init_code = :(using GameTracing; Main.TestHelpers.share_helpers!(@__MODULE__)),
)
