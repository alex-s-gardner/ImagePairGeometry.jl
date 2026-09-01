using ImagePairGeometry
using Documenter
using Proj   # loads ImagePairGeometryProjExt, so its docstrings are part of the built docs

const PROJ_EXT = Base.get_extension(ImagePairGeometry, :ImagePairGeometryProjExt)

DocMeta.setdocmeta!(ImagePairGeometry, :DocTestSetup, :(using ImagePairGeometry); recursive=true)

makedocs(;
    modules=[ImagePairGeometry, PROJ_EXT],
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
