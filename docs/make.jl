using Documenter, GameTracing

makedocs(;
    sitename = "GameTracing.jl",
    authors = "Carlos Parada",
    modules = [GameTracing],
    format = Documenter.HTML(canonical = "https://paradacarleton.github.io/GameTracing.jl/stable"),
    pages = ["Home" => "index.md", "API" => "api.md"],
    warnonly = false,
)

const REPOSITORY = "ParadaCarleton/GameTracing.jl"
# deploydocs skips the push without an error when its repo is not the one CI runs in.
repository = get(ENV, "GITHUB_REPOSITORY", REPOSITORY)
repository == REPOSITORY || error("docs/make.jl deploys to $REPOSITORY, but CI runs in $repository")
deploydocs(repo = "github.com/$REPOSITORY.git", push_preview = true)
