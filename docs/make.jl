using ImagePairGeometry
using Documenter
using Proj      # loads ImagePairGeometryProjExt
using Rasters, ArchGDAL, DimensionalData, DiskArrays  # loads ImagePairGeometryRastersExt

const PROJ_EXT = Base.get_extension(ImagePairGeometry, :ImagePairGeometryProjExt)
const RASTERS_EXT = Base.get_extension(ImagePairGeometry, :ImagePairGeometryRastersExt)

DocMeta.setdocmeta!(ImagePairGeometry, :DocTestSetup, :(using ImagePairGeometry); recursive=true)

makedocs(;
    modules=[ImagePairGeometry, PROJ_EXT, RASTERS_EXT],
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
