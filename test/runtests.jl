using ImagePairGeometry
using Test
using Aqua

include("npz.jl")

@testset verbose=true "ImagePairGeometry.jl" begin
    @testset "Code quality (Aqua.jl)" begin
        # `persistent_tasks` re-resolves this package in a clean environment, which cannot satisfy
        # the FastGeoProjections `[compat]` bound while no release carrying the point-operator API is
        # registered — CI supplies it from `main` instead, and a fresh resolve does not inherit that.
        # Re-enable once that release exists.
        Aqua.test_all(ImagePairGeometry; persistent_tasks = false)
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
    # The SLCDatasets extension. Asked by trying to import rather than by looking, since the reader is
    # an unregistered weak dependency and `Pkg.test`'s sandbox is not where it is added.
    if (try
            @eval import SLCDatasets
            true
        catch
            false
        end)
        @time @testset "radar types from SLCDatasets" begin include("radar_slcdatasets.jl") end
    else
        @info "skipping the SLCDatasets extension test; SLCDatasets could not be loaded"
    end
end
