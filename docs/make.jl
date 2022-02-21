using PythonCallChainRules
using Documenter

DocMeta.setdocmeta!(PythonCallChainRules, :DocTestSetup, :(using PythonCallChainRules); recursive=true)

makedocs(;
    modules=[PythonCallChainRules],
    authors="rejuvyesh <mail@rejuvyesh.com> and contributors",
    repo="https://github.com/rejuvyesh/PythonCallChainRules.jl/blob/{commit}{path}#{line}",
    sitename="PythonCallChainRules.jl",
    format=Documenter.HTML(;
        prettyurls=get(ENV, "CI", "false") == "true",
        canonical="https://rejuvyesh.github.io/PythonCallChainRules.jl",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
    ],
)

deploydocs(;
    repo="github.com/rejuvyesh/PythonCallChainRules.jl",
    devbranch="main",
)
