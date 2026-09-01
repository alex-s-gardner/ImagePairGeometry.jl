using ImagePairGeometry
using Documenter

DocMeta.setdocmeta!(ImagePairGeometry, :DocTestSetup, :(using ImagePairGeometry); recursive=true)

makedocs(;
    modules=[ImagePairGeometry],
    authors="Alex Gardner <alex.s.gardner@jpl.nasa.gov>, and contributors",
    sitename="ImagePairGeometry.jl",
    format=Documenter.HTML(;
        canonical="https://alex-s-gardner.github.io/ImagePairGeometry.jl",
        edit_link="main",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
    ],
)

deploydocs(;
    repo="github.com/alex-s-gardner/ImagePairGeometry.jl",
    devbranch="main",
)
