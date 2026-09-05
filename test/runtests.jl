using ImagePairGeometry
using Test
using Aqua

include("npz.jl")

@testset verbose=true "ImagePairGeometry.jl" begin
    @testset "Code quality (Aqua.jl)" begin
        Aqua.test_all(ImagePairGeometry)
    end
    @time @testset "rounding" begin include("rounding.jl") end
    @time @testset "vecmath" begin include("vecmath.jl") end
    @time @testset "coregister" begin include("coregister.jl") end
    @time @testset "window" begin include("window.jl") end
    @time @testset "kernel" begin include("kernel.jl") end
    @time @testset "velocity conversion" begin include("velocity.jl") end
    @time @testset "radar numerics vs isce3" begin include("radar_numerics.jl") end
    @time @testset "radar coordinate vs reference" begin include("radar_coordinate.jl") end
    @time @testset "radar per-point kernel" begin include("radar_geometry.jl") end
    @time @testset "radar geogrid vs reference" begin include("radar_geogrid.jl") end
    @time @testset "geogrid vs reference" begin include("geogrid.jl") end
    @time @testset "blocks and threads" begin include("blocks.jl") end
    @time @testset "interpolated transform" begin include("interpolate.jl") end
    @time @testset "FastGeoProjections transform" begin include("fastgeoprojections.jl") end
    @time @testset "Rasters IO" begin include("rasters.jl") end
end
