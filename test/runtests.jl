using ImagePairGeometry
using Test
using Aqua

@testset "ImagePairGeometry.jl" begin
    @testset "Code quality (Aqua.jl)" begin
        Aqua.test_all(ImagePairGeometry)
    end
    # Write your tests here.
end
