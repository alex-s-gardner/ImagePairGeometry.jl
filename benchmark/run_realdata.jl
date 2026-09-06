# The radar path against the compiled reference, on real data: agreement and wall clock.
#
# Every other script here measures synthetic inputs, because they need no data on disk. This one
# measures the case a user actually runs — a real NISAR acquisition over the ITS_LIVE 120 m grid — so it
# needs `test/reference/gen_radar_realdata.py` run first and its output directory named:
#
#     micromamba run -n geogrid-ref python test/reference/gen_radar_realdata.py \
#         --reference <ref.json> --secondary <sec.json> \
#         --params ~/data/autorift/tests/params --outdir <dir>
#     IPG_REALDATA_DIR=<dir> julia --project=benchmark -t 8 benchmark/run_realdata.jl
#
# The reference's own kernel time is in that directory's `run.json`, so the comparison needs no second
# Python run. Results are written to `julia_run.json` beside it: the numbers cost minutes to produce
# and prose is a poor place to keep them.

using ImagePairGeometry
using ImagePairGeometry: INT_BANDS, FLOAT_BANDS, reference_files, nodata_from, mapgrid,
                         fast_transform, Orbit, RadarCoordinate, CoregisteredPair,
                         LookLeft, LookRight, incidence_angle
using ArchGDAL
using Dates
using JSON3
using Printf
using Rasters
using StaticArrays

const RA_EXT = Base.get_extension(ImagePairGeometry, :ImagePairGeometryRastersExt)
using .RA_EXT: RasterInputs

const REALDATA_DIR = get(ENV, "IPG_REALDATA_DIR", "")
const PARAMS_DIR = get(ENV, "IPG_PARAMS_DIR", expanduser("~/data/autorift/tests/params"))

isempty(REALDATA_DIR) && error("""
    set IPG_REALDATA_DIR to a directory produced by test/reference/gen_radar_realdata.py""")

hexparse(v) = v isa Number ? Float64(v) : parse(Float64, v.hex)
cf_epoch(units::AbstractString) = DateTime(replace(strip(split(units, "since")[2]), " " => "T"))

const RUN = JSON3.read(read(joinpath(REALDATA_DIR, "run.json"), String))
const REF = JSON3.read(read(joinpath(REALDATA_DIR, "reference_metadata.json"), String))
const SEC = JSON3.read(read(joinpath(REALDATA_DIR, "secondary_metadata.json"), String))

function build_case()
    g, orb = REF.geometry, REF.orbit
    epoch = cf_epoch(orb.epoch)
    orbit = Orbit(; time = [hexparse(v) for v in orb.time],
                  position = [SVector{3,Float64}(hexparse.(p)...) for p in orb.position],
                  velocity = [SVector{3,Float64}(hexparse.(v)...) for v in orb.velocity])
    offset = (epoch - DateTime(Date(epoch))).value / 1000
    look = lowercase(REF.identification.look_direction) == "left" ? LookLeft : LookRight
    kw = (; orbit, starting_range = hexparse(g.starting_range),
          dr = hexparse(g.range_pixel_spacing), sensing_start = hexparse(g.sensing_start),
          prf = hexparse(g.prf), nsamples = g.nsamples, nlines = g.nlines, look_side = look,
          wavelength = hexparse(g.wavelength), orbit_epoch_offset = offset)
    coord = RadarCoordinate(; kw..., incidence_angle = incidence_angle(; kw...))
    dt = (cf_epoch(SEC.orbit.epoch) - epoch).value / 1000 +
         hexparse(SEC.geometry.sensing_start) - hexparse(g.sensing_start)
    pair = CoregisteredPair(coord; dt)

    dem = Raster(joinpath(PARAMS_DIR, "NPS_0120m_h.tif"); lazy = true, missingval = nothing)
    from_file = mapgrid(dem)
    grid = MapGrid(geotransform = from_file.geotransform, size = from_file.size,
                   crs = Int(RUN.grid.epsg))
    w = RUN.window
    win = CartesianIndices((w.pOff + 1:(w.pOff + w.pCount), w.lOff + 1:(w.lOff + w.lCount)))

    param(n) = Raster(joinpath(PARAMS_DIR, "NPS_0120m_$n.tif"); lazy = true, missingval = nothing)
    src = RasterInputs(dem = dem, dhdx = param("dhdx"), dhdy = param("dhdy"),
                       vx = param("vx"), vy = param("vy"),
                       srx = param("vxSearchRange"), sry = param("vySearchRange"),
                       csminx = param("xMinChipSize"), csminy = param("yMinChipSize"),
                       csmaxx = param("xMaxChipSize"), csmaxy = param("yMaxChipSize"),
                       ssm = param("StableSurface"))
    return (; grid, pair, coord, win, src)
end

# The radar path takes its sentinel from the reference-velocity band, which the ITS_LIVE rasters
# declare as -32767.
function vx_nodata()
    return ArchGDAL.read(joinpath(PARAMS_DIR, "NPS_0120m_vx.tif")) do ds
        v = ArchGDAL.getnodatavalue(ArchGDAL.getband(ds, 1))
        v === nothing ? nothing : Float64(v)
    end
end

const C = build_case()
const ND = nodata_from(vx_nodata())

compute(ntasks) = pairgeometry_blocked(C.grid, C.pair, C.src;
                                       transform = () -> fast_transform(C.grid.crs, 4326),
                                       window = C.win, ntasks = ntasks, nodata = ND)

npts = length(C.win)
@printf("grid      : %d x %d = %d points at %g m, EPSG %d\n",
        RUN.window.pCount, RUN.window.lCount, npts, RUN.grid.spacing, RUN.grid.epsg)
@printf("reference : %s\n", REF.source)
@printf("secondary : %s\n", SEC.source)
@printf("interval  : %.6f days\n", C.pair.dt / 86400)
@printf("incidence : %.6f rad (reference %.6f)\n",
        C.coord.incidence_angle, hexparse(RUN.radar.incidence_angle))

# Warm up, and anchor the timings to a result that agrees: a fast wrong answer is the failure mode
# this repository exists to catch.
r0 = compute(Threads.nthreads())
@printf("\nnvalid    : %d / %d (%.2f%%)\n", nvalid(r0), npoints(r0), 100 * nvalid(r0) / npoints(r0))

function band_agreement(r)
    readband(file, b) = ArchGDAL.read(joinpath(REALDATA_DIR, file)) do ds
        permutedims(ArchGDAL.read(ArchGDAL.getband(ds, b)), (1, 2))
    end
    int_worst = Dict{String,Tuple{Int,Int}}()
    float_worst = Dict{String,Float64}()
    for (file, fields) in reference_files(r)
        isfile(joinpath(REALDATA_DIR, file)) || continue
        for (bi, f) in enumerate(fields)
            want = readband(file, bi)
            got = getfield(r, f)
            if f in INT_BANDS
                d = Int.(got) .- Int.(Int32.(want))
                int_worst[String(f)] = (count(!=(0), d), isempty(d) ? 0 : maximum(abs, d))
            else
                worst = 0.0
                for k in eachindex(got, want)
                    w, g = Float64(want[k]), got[k]
                    (w == -32767.0 || g == -32767.0) && continue
                    worst = max(worst, abs(g - w) / max(abs(w), 1.0))
                end
                float_worst[String(f)] = worst
            end
        end
    end
    return int_worst, float_worst
end

int_worst, float_worst = band_agreement(r0)
println("\n== agreement with the compiled reference ==")
for (band, (n, m)) in sort(collect(int_worst); by = kv -> -kv[2][1])
    @printf("  %-16s %8d points differ, max %d index\n", band, n, m)
end
for (band, w) in sort(collect(float_worst); by = last, rev = true)
    @printf("  %-16s %.3e relative\n", band, w)
end

# Threading, not repetition: each count is timed once. The kernel over 4.7M points takes seconds, and
# the run-to-run spread is far below the gap between task counts, so repeating it would buy noise
# rather than precision.
println("\n== wall clock, by task count ==")
ref_kernel = RUN.times_s.kernel
times = Dict{String,Float64}()
for ntasks in (1, 2, 4, 8)
    ntasks <= Threads.nthreads() || continue
    t0 = time()
    r = compute(ntasks)
    el = time() - t0
    times["ntasks_$ntasks"] = el
    @printf("  ntasks=%d  %7.3f s  %6.2f us/pt  %5.2fx the reference\n",
            ntasks, el, 1e6 * el / npts, ref_kernel / el)
end
@printf("  reference  %7.3f s  %6.2f us/pt  (compiled C++, single-threaded)\n",
        ref_kernel, 1e6 * ref_kernel / npts)

record = Dict(
    "reference" => REF.source,
    "secondary" => SEC.source,
    "grid" => Dict("epsg" => RUN.grid.epsg, "spacing" => RUN.grid.spacing,
                   "pCount" => RUN.window.pCount, "lCount" => RUN.window.lCount),
    "npoints" => npts,
    "nvalid" => nvalid(r0),
    "nthreads" => Threads.nthreads(),
    "times" => times,
    "reference_kernel_s" => ref_kernel,
    "int_bands" => Dict(k => [v[1], v[2]] for (k, v) in int_worst),
    "float_bands" => float_worst,
    "all_int_bitwise" => all(v[1] == 0 for v in values(int_worst)),
    "worst_float" => maximum(values(float_worst)),
)
open(joinpath(REALDATA_DIR, "julia_run.json"), "w") do io
    JSON3.pretty(io, record)
end
println("\nwrote ", joinpath(REALDATA_DIR, "julia_run.json"))
