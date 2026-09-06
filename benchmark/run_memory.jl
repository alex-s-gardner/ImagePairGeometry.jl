include(joinpath(@__DIR__, "radar_setup.jl"))
using Printf, BenchmarkTools
using ImagePairGeometry: INT_BANDS, FLOAT_BANDS, allocate_geometry, window_geotransform

println("== per-grid-point storage ==")
@printf("output: %d Int32 bands + %d Float64 bands = %d B/pt\n",
        length(INT_BANDS), length(FLOAT_BANDS),
        4*length(INT_BANDS) + 8*length(FLOAT_BANDS))
@printf("inputs: 12 Float64 rasters                = %d B/pt\n", 12*8)

for n in (1_000, 10_000, 40_000)
    tot = (4*11 + 8*8 + 12*8) * n^2
    @printf("grid %6d^2 = %.2e pts: %8.2f GiB resident if unblocked\n", n, Float64(n)^2, tot/2^30)
end

println("\n== allocation of the result alone ==")
for n in (256, 1024)
    win = CartesianIndices((1:n, 1:n))
    c = bench_case(64)
    b = @benchmark allocate_geometry($win, window_geotransform($c.grid, $win), $c.grid.crs,
                                     nodata_from(0.0), $c.coord) samples=5 evals=1
    @printf("n=%5d: %8.2f ms  %8.2f MiB\n", n, minimum(b).time/1e6, minimum(b).memory/2^20)
end

println("\n== footprint_bounds (160 rdr2geo solves) ==")
c = bench_case(64)
b = @benchmark footprint_bounds($c.tf.forward, $c.coord)
@printf("footprint_bounds: %8.3f ms  %6d allocs %8d B\n",
        minimum(b).time/1e6, minimum(b).allocs, minimum(b).memory)

println("\n== blocked, peak allocation vs unblocked ==")
n = 256
c = bench_case(n)
src = InMemoryInputs(c.inputs, c.win)
for bs in ((n, n), (128, 128), (64, 64))
    b = @benchmark pairgeometry_blocked($c.grid, $c.pair, $src; transform = $c.tf,
        window = $c.win, blocksize = $bs, ntasks = 1, nodata = nodata_from(0.0)) samples=3 evals=1 seconds=90
    m = minimum(b)
    @printf("blocksize %s: %8.1f ms  %8.2f MiB  %6d allocs\n", bs, m.time/1e6, m.memory/2^20, m.allocs)
end

println("\n== threading scaling (nthreads = $(Threads.nthreads())) ==")
for nt in (1, 2, 4, 8)
    nt > Threads.nthreads() && continue
    b = @benchmark pairgeometry_blocked($c.grid, $c.pair, $src; transform = $c.tf,
        window = $c.win, blocksize = (64, 64), ntasks = $nt, nodata = nodata_from(0.0)) samples=3 evals=1 seconds=120
    @printf("ntasks=%2d: %8.1f ms\n", nt, minimum(b).time/1e6)
end
