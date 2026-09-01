using ImagePairGeometry
using Test
using Aqua

include("npz.jl")

@testset "ImagePairGeometry.jl" begin
    @testset "Code quality (Aqua.jl)" begin
        Aqua.test_all(ImagePairGeometry)
    end
    @time @testset "rounding" begin include("rounding.jl") end
    @time @testset "vecmath" begin include("vecmath.jl") end
    @time @testset "coregister" begin include("coregister.jl") end
    @time @testset "window" begin include("window.jl") end
    @time @testset "kernel" begin include("kernel.jl") end
    @time @testset "geogrid vs reference" begin include("geogrid.jl") end
    @time @testset "blocks and threads" begin include("blocks.jl") end
    @time @testset "AutoRIFT handoff" begin include("autorift.jl") end
    @time @testset "Rasters IO" begin include("rasters.jl") end
end
