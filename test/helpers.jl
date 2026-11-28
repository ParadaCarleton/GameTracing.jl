# The setup each test worker runs once. Its test files run in modules of their own that bind the helpers' names.

# A worker past 30 minutes is stuck: a separate process SIGQUITs it, printing every thread's backtrace.
const WATCHDOG = run(`sh -c "timeout 1800 tail --pid=$(getpid()) -f /dev/null || kill -QUIT $(getpid())"`; wait = false)

"""The games, fixtures and oracles the test files share, loaded once per worker so its files share compiled code."""
module TestHelpers

using GameTracing

include("games.jl")
include("fixtures.jl")
include("causes.jl")
include("rounding.jl")
include("arms.jl")
include("gambit.jl")

const GAME = LinearGainPass(Float32[0 1; 1 0], Float32[0.5f0, -0.25f0])
const TARGET_PRECISION = 2.0f0
const WIDTH = 2

"""Each name the helpers define or import, as the test files see it."""
const HELPER_NAMES = filter(
    name -> !startswith(String(name), '#') && name ∉ (:TestHelpers, :eval, :include),
    names(TestHelpers; all = true, imported = true),
)

"""Bind each helper name in `sandbox`, the module one test file runs in, unless it already names the same object."""
function share_helpers!(sandbox::Module)::Nothing
    shared = [
        Expr(:., name)
            for name in HELPER_NAMES
            if !(isdefined(sandbox, name) && getglobal(sandbox, name) === getglobal(TestHelpers, name))
    ]
    Core.eval(sandbox, Expr(:using, Expr(:(:), Expr(:., :Main, :TestHelpers), shared...)))
    return nothing
end

end
