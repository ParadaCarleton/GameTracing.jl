# Project.toml's url source loads main's PathFollowing. A checkout tests its own lib/PathFollowing only after
# `Pkg.develop(path = "lib/PathFollowing")`, as CI.yml does. This testset fails when the two sources differ.

"""The `.jl` files under `root`, as paths relative to it, sorted."""
function julia_files(root::AbstractString)::Vector{String}
    return sort!([relpath(joinpath(dir, name), root) for (dir, _, names) in walkdir(root) for name in names if endswith(name, ".jl")])
end

@testset "the loaded PathFollowing has the sources of lib/PathFollowing" begin
    shipped = joinpath(pkgdir(GameTracing), "lib", "PathFollowing", "src")
    loaded = joinpath(pkgdir(GameTracing.PathFollowing), "src")
    @test julia_files(loaded) == julia_files(shipped)
    @test all(file -> read(joinpath(loaded, file), String) == read(joinpath(shipped, file), String), julia_files(shipped))
end
