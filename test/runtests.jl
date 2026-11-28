using CondaPkg: CondaPkg
using GameTracing: GameTracing
using ParallelTestRunner: parse_args, runtests

# Gambit's logit QRE is the test oracle, at the version checked by hand against a 50-digit trace. The pin goes into the
# test environment, not a package-level CondaPkg.toml that every user of GameTracing would read. It resolves here
# once, as workers that each wrote CondaPkg.toml would race.
CondaPkg.add("python"; version = "=3.12", resolve = false)
CondaPkg.add_pip("pygambit"; version = "==16.7.0")

"""
The test files whose names start with one of `prefixes` (all of them for no prefix), and the code rules always.
Throws for a prefix no file has.
"""
function selected_files(prefixes::AbstractVector{<:AbstractString})::Vector{String}
    files = [filter(endswith("_tests.jl"), readdir(@__DIR__)); "code_rules.jl"]
    unmatched = filter(prefix -> !any(startswith(prefix), files), prefixes)
    isempty(unmatched) || throw(ArgumentError("no test file starts with $(join(unmatched, ", "))"))
    return filter(file -> file == "code_rules.jl" || isempty(prefixes) || any(prefix -> startswith(file, prefix), prefixes), files)
end

# Each selected file runs in a module of its own, spread over worker processes. `Pkg.test(; test_args = ["trace",
# "--jobs=2"])` runs the files whose names start with `trace`, then the code rules, on 2 workers.
runtests(
    GameTracing, parse_args(filter(startswith("--"), ARGS));
    testsuite = Dict(file => :(include($(joinpath(@__DIR__, file)))) for file in selected_files(filter(!startswith("--"), ARGS))),
    init_worker_code = :(include($(joinpath(@__DIR__, "helpers.jl")))),
    init_code = :(using GameTracing; Main.TestHelpers.share_helpers!(@__MODULE__)),
)
