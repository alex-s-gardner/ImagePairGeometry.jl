using ImagePairGeometry
using Test
using Aqua

include("npz.jl")

# `AutoRIFT` is not registered, so the test environment cannot depend on it and CI cannot load it.
# The handoff test runs when it is resolvable in the active environment — add it with
# `Pkg.develop(path = "…/AutoRIFT.jl")` in `test/` to exercise the extension locally.
const HAVE_AUTORIFT = Base.identify_package("AutoRIFT") !== nothing

@testset verbose=true "ImagePairGeometry.jl" begin
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
    @time @testset "interpolated transform" begin include("interpolate.jl") end
    if HAVE_AUTORIFT
        @time @testset "AutoRIFT handoff" begin include("autorift.jl") end
    else
        @info "Skipping the AutoRIFT handoff tests: AutoRIFT is not in this environment."
    end
    @time @testset "Rasters IO" begin include("rasters.jl") end
end
