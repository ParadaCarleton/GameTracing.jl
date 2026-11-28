# Rewrite each url source of a Project.toml, and of every project it sources, as a path source to a git clone.
# julia-downgrade-compat restores path sources to the manifest it resolves; it drops url sources.
# Run as: julia .github/path_sources.jl PROJECT_FILE CLONE_DIRECTORY
using TOML: TOML

const Clones = Dict{Tuple{String, Union{Nothing, String}}, String}

"""The clone of `url` at `rev` (`nothing`: the default branch) in `root`, made once per url and rev in `clones`."""
function clone!(clones::Clones, url::String, rev::Union{Nothing, String}, root::String)::String
    return get!(clones, (url, rev)) do
        directory = joinpath(root, string(length(clones) + 1))
        run(`git clone --quiet $url $directory`)
        rev === nothing || run(`git -C $directory checkout --quiet $rev`)
        return directory
    end
end

"""
Rewrite `project_file`'s url sources as path sources to clones in `root`, then visit each project it sources.
`visited` holds the project files seen, so a cycle of sources ends.
"""
function path_sources!(clones::Clones, visited::Set{String}, project_file::String, root::String)::Nothing
    project_file in visited && return nothing
    push!(visited, project_file)
    project = TOML.parsefile(project_file)
    sources = get(project, "sources", Dict{String, Any}())
    sourced = String[]
    rewritten = false
    for (name, source) in sources
        if haskey(source, "url")
            rev = get(source, "rev", nothing)
            checkout = normpath(joinpath(clone!(clones, source["url"], rev, root), get(source, "subdir", ".")))
            println("$name: $(source["url"]) at $(something(rev, "its default branch")) becomes the path source $checkout")
            # Relative to the project, as the downgrade action writes the manifest's path; Pkg.test asserts the two agree.
            sources[name] = Dict("path" => relpath(checkout, dirname(project_file)))
            rewritten = true
        end
        # A source with a rev alone names a registered package: no project to visit.
        haskey(sources[name], "path") && push!(sourced, joinpath(dirname(project_file), sources[name]["path"], "Project.toml"))
    end
    rewritten && open(io -> TOML.print(io, project), project_file, "w")
    foreach(file -> path_sources!(clones, visited, normpath(file), root), sourced)
    return nothing
end

length(ARGS) == 2 || error("usage: julia path_sources.jl PROJECT_FILE CLONE_DIRECTORY")
path_sources!(Clones(), Set{String}(), abspath(ARGS[1]), abspath(ARGS[2]))
