using ImagePairGeometry
using Test
using Aqua

@testset "ImagePairGeometry.jl" begin
    @testset "Code quality (Aqua.jl)" begin
        Aqua.test_all(ImagePairGeometry)
    end
    @time @testset "rounding" begin include("rounding.jl") end
    @time @testset "vecmath" begin include("vecmath.jl") end
    @time @testset "coregister" begin include("coregister.jl") end
    @time @testset "window" begin include("window.jl") end
end
