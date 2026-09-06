using ImagePairGeometry
using Documenter
using Rasters, ArchGDAL, DimensionalData, DiskArrays  # loads ImagePairGeometryRastersExt

const RASTERS_EXT = Base.get_extension(ImagePairGeometry, :ImagePairGeometryRastersExt)

DocMeta.setdocmeta!(ImagePairGeometry, :DocTestSetup, :(using ImagePairGeometry); recursive=true)

makedocs(;
    modules=[ImagePairGeometry, RASTERS_EXT],
    authors="Alex Gardner <alex.s.gardner@jpl.nasa.gov>, and contributors",
    sitename="ImagePairGeometry.jl",
    format=Documenter.HTML(;
        canonical="https://alex-s-gardner.github.io/ImagePairGeometry.jl",
        edit_link="main",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
        "Radar geometry" => "radar.md",
    ],
)

deploydocs(;
    repo="github.com/alex-s-gardner/ImagePairGeometry.jl",
    devbranch="main",
)
