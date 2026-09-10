# The golden runs on disk: which cases exist, and the parameters each one's geogrid was run with.
#
# A run directory holds the nine `window_*.tif` geogrid outputs, a `capture/` directory of the arrays
# Python autoRIFT was handed, and `capture.log` — the container's stdout, which is the only record of
# the parameters `runGeogrid` was called with. This file turns that log into a `GoldenRun`.
#
# **The log is a lossy record, and that governs the whole design here.** `GeogridOptical.cpp` prints
# with C++ default `ostream` formatting, which is `%g`: six significant figures. That is enough for
# an integer index and not enough for a coordinate. Measured on the cases on disk:
#
#     X-direction coordinate: 437992  15          the pan asset's origin is 437992.5
#     X-direction coordinate: -2.12881e+06  15    ...is -2128815.0, so 5 m are gone
#     Y-direction coordinate: 7.70004e+06  -10    ...is 7700040.0, recoverable only by luck
#
# A half-pixel error in the image origin moves every pixel index by half a pixel, which is exactly
# the failure `AutoRIFT.jl/tools/golden/intermediate.jl` documents as invisible in a median: zero
# residual under uniform motion, growing with the velocity gradient. So the image geometry is *not*
# taken from the log. `image_geometry` reads it from the authoritative source per platform, and the
# log's printed value serves only as a cross-check that the authoritative one rounds to it.
#
# What the log *is* authoritative for is everything integral or exact: the EPSG code, the origin
# index and dimensions of the geogrid window, the nodata sentinel, and the parameter-raster URLs. The
# repeat interval prints as `4.1472e+06` — exact in six figures because it is a whole number of days
# in seconds — and is cross-checked against the product's own `date_dt`.
#
# Two further parsing hazards, both found on the runs on disk rather than anticipated:
#
#   * **Container stdout interleaves.** In `S1A_IW_SLC__1SSV_20240618T025528` a download progress
#     line lands mid-number: `Range: 803581  2.3295Using granule search...`. A regex that takes
#     "the rest of the line" silently yields `2.3295` for a `dr` of `2.32956`. Every numeric field is
#     therefore matched with an anchored pattern that must consume the whole value, and a line that
#     does not match wholly is an error rather than a partial parse.
#   * **The block header can be overwritten.** In `S1A_IW_SLC__1SSV_20240618T025533` the log holds
#     ` parameters: ` where every other radar run holds `Radar parameters: ` — a carriage return from
#     a progress line ate the word. So a block is identified by the fields inside it, not by its
#     header.

using ArchGDAL
using ImagePairGeometry: REFERENCE_FILES
using JSON3
using Printf

# The AutoRIFT.jl harness, reused rather than reimplemented: `manifest.json` is the authoritative
# job-to-product mapping for all 22 cases, and `xchg.jl` reads the `.abx` captures. Sibling checkout
# by default, since neither package depends on the other.
const GOLDEN_AUTORIFT = get(ENV, "IPG_GOLDEN_AUTORIFT",
                            normpath(joinpath(@__DIR__, "..", "..", "..", "AutoRIFT.jl")))

isdir(GOLDEN_AUTORIFT) || error("""
    the golden harness needs AutoRIFT.jl beside this repository for `manifest.json` and `xchg.jl`,
    and $GOLDEN_AUTORIFT is not a directory. Set IPG_GOLDEN_AUTORIFT to its checkout.""")

include(joinpath(GOLDEN_AUTORIFT, "tools", "golden", "manifest.jl"))
include(joinpath(GOLDEN_AUTORIFT, "tools", "ab", "xchg.jl"))

const PARAMS_CACHE = get(ENV, "IPG_GOLDEN_PARAMS", joinpath(CACHE, "params_cache"))

# Where a reference run lands. The same two lines as `AutoRIFT.jl/tools/golden/reference.jl`, repeated
# rather than included: that file also carries the container driver, and pulling it in would make
# reading a run on disk depend on Docker being installed.
runs_dir(c::GoldenCase) = joinpath(CACHE, "runs", c.product)
run_dir(c::GoldenCase, n::Integer) = joinpath(runs_dir(c), string(n))

# Every geogrid output uses one sentinel, and the log states it. Asserted rather than assumed: the
# whole comparison is "which points are missing", so a wrong sentinel makes every band agree
# trivially on the points that matter least.
const EXPECTED_NODATA = -32767.0

"""
    GoldenRun

One geogrid run on disk: where it is, and the parameters it was run with.

`image` is `nothing` until [`image_geometry`](@ref) resolves it, because the log does not carry it to
full precision — see this file's header. Everything else here is read from the log and is exact.

`window` is the grid window in the DEM's index space, one-based: the log's `Origin index (in DEM) of
geogrid` plus its `Dimensions of geogrid`. That is the same convention
`test/radar_itslive_product.jl` builds from a `run.json`, and it is what makes the comparison
band-aligned with the reference rather than starting at the DEM's own corner.
"""
struct GoldenRun
    case::GoldenCase
    dir::String
    run::Int
    radar::Bool
    epsg::Int
    dt::Float64
    chip_size_0::Float64
    grid_spacing::Float64
    nodata::Float64
    window::CartesianIndices{2}
    dem_url::String
    param_urls::Dict{String,String}
    # Optical: the pixel size the log prints, used to cross-check the authoritative geometry.
    printed_origin::Union{NTuple{2,Float64},Nothing}
    printed_spacing::Union{NTuple{2,Float64},Nothing}
    printed_size::Union{NTuple{2,Int},Nothing}
    # Radar: the `Radar parameters:` block, plus `output.txt`'s pixel sizes.
    radar_params::Union{NamedTuple,Nothing}
    orbit_files::Vector{String}
end

Base.show(io::IO, r::GoldenRun) = print(io, "GoldenRun(", short_name(r), ", run ", r.run,
                                        r.radar ? ", radar" : ", optical", ")")

"""
    short_name(r) -> String

A name for a run that fits a report column: the case's first granule, cut at its acquisition
timestamp. A full product name is 100 characters and two of them do not fit a terminal line.

The cut keeps the timestamp because that is what distinguishes the two `S1A_IW_SLC__1SSV_20240618`
cases and the two `LT05` ones, which agree on everything before it.
"""
short_name(r::GoldenRun) = short_name(r.case)

function short_name(c::GoldenCase)
    g = first(split(c.product, "_X_"))
    # Cut at the first acquisition timestamp: what follows it is orbit, take and product IDs on a
    # Sentinel granule and processing dates on a Landsat one. A Landsat name has no `T`, so it keeps
    # its own second date field, which is what distinguishes two acquisitions of one path/row.
    m = match(r"^(.*?_\d{8}T\d{6})", g)
    m === nothing || return String(m.captures[1])
    return length(g) <= 41 ? String(g) : String(first(g, 41))
end

# ---------------------------------------------------------------------------
# Numeric fields, parsed so a truncated line is an error
# ---------------------------------------------------------------------------
#
# `NUMBER` matches a complete C++ `%g` output and nothing else: optional sign, digits, optional
# fraction, optional exponent. The pattern is used with a following `\s` or end-of-line so that a
# progress line glued onto the value (`2.3295Using granule search...`) fails to match rather than
# yielding its prefix.

const NUMBER = raw"[-+]?[0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?"

"""
    logfields(text, label, n) -> Vector{Float64}

The `n` numbers on the line `label` introduces, or `nothing` when the label is absent.

Throws when the line is present but does not hold exactly `n` complete numbers — a truncated value
is the failure mode this exists to catch, and a partially parsed geometry is worse than no geometry.
"""
function logfields(text::AbstractString, label::AbstractString, n::Integer)
    # `label` is a literal, and several labels contain regex metacharacters (`(`, `.`).
    pat = Regex("^" * escape_regex(label) * raw":\s*" *
                join(fill("(" * NUMBER * ")", n), raw"\s+") * raw"\s*$", "m")
    m = match(pat, text)
    m === nothing || return [parse(Float64, c) for c in m.captures]

    # Present but unparseable: say which line, since the cause is upstream noise in the log rather
    # than anything a caller controls.
    loose = match(Regex("^" * escape_regex(label) * ":.*\$", "m"), text)
    loose === nothing && return nothing
    error("""the log line for "$label" does not hold $n complete numbers:
                 $(strip(loose.match))
             This is container stdout interleaving a progress line into the value; the run's log
             cannot be used and the geometry must come from another source.""")
end

escape_regex(s::AbstractString) = replace(s, r"([\\^$.|?*+()\[\]{}])" => s"\\\1")

"""
    logfields!(text, label, n) -> Vector{Float64}

[`logfields`](@ref), erroring when the label is absent. For a field the comparison cannot proceed
without: a silently defaulted `dt` or nodata is a wrong answer in every band.
"""
function logfields!(text::AbstractString, label::AbstractString, n::Integer)
    v = logfields(text, label, n)
    v === nothing && error("the log has no \"$label\" line, which this comparison requires")
    return v
end

"""
    trylogfields(text, label, n) -> Union{Vector{Float64},Nothing}

[`logfields`](@ref) returning `nothing` where it would throw, for a field with another source.

Used only where a fallback exists and is checked against this value when both are intact. A field
with no other source goes through `logfields!`, so a truncated line stays an error.
"""
trylogfields(text, label, n) = try
    logfields(text, label, n)
catch
    nothing
end

"""
    leading_number(text, label) -> Float64

The first number on the line `label` introduces, whatever follows it.

For a value whose *own* digits are intact on a line a progress message truncated further along. The
number must still be followed by whitespace, so a value the noise landed inside of is not accepted:
`803581  2.3295Using...` yields `803581`, and a hypothetical `80358Using...` matches nothing.
"""
function leading_number(text::AbstractString, label::AbstractString)
    m = match(Regex("^" * escape_regex(label) * raw":\s*(" * NUMBER * raw")\s", "m"), text)
    m === nothing && error("the log has no parseable leading number on the \"$label\" line")
    return parse(Float64, m.captures[1])
end

"""
    band_nodata(dir) -> Float64

The nodata value the run's output rasters declare, asserted equal across all of them.

The rasters rather than the log: they are what the comparison reads, and one run's log has the value
truncated by an interleaved progress line.
"""
function band_nodata(dir::AbstractString)
    seen = Dict{Float64,String}()
    for (file, _) in REFERENCE_FILES
        path = joinpath(dir, file)
        isfile(path) || continue
        ArchGDAL.read(path) do ds
            for b in 1:ArchGDAL.nraster(ds)
                v = ArchGDAL.getnodatavalue(ArchGDAL.getband(ds, b))
                v === nothing && continue
                seen[Float64(v)] = "$file band $b"
            end
        end
    end
    isempty(seen) && error("no output raster in $dir declares a nodata value")
    length(seen) == 1 || error(
        "the output rasters in $dir declare more than one nodata value: " *
        join(("$v ($where)" for (v, where) in seen), ", "))
    return first(keys(seen))
end

"""
    output_txt_resolution(dir) -> (ground_range, azimuth)

The ground pixel sizes from a radar run's `output.txt`, or `(nothing, nothing)` without one.

More digits than the log's `%g` prints, which matters because these divide the search extent.
"""
function output_txt_resolution(dir::AbstractString)
    path = joinpath(dir, "output.txt")
    isfile(path) || return (nothing, nothing)
    text = read(path, String)
    grab(k) = (m = match(Regex("^" * k * raw":\s*(" * NUMBER * raw")\s*$", "m"), text);
               m === nothing ? nothing : parse(Float64, m.captures[1]))
    return (grab("X_res"), grab("Y_res"))
end

"""
    geogrid_dims(text) -> (nx, ny)

The `Dimensions of geogrid: 2341 x 2351` line. Separated by `x` rather than whitespace, so it needs
its own pattern rather than [`logfields`](@ref).
"""
function geogrid_dims(text::AbstractString)
    m = match(Regex(raw"^Dimensions of geogrid:\s*([0-9]+)\s*x\s*([0-9]+)\s*$", "m"), text)
    m === nothing && error("the log has no \"Dimensions of geogrid\" line")
    return (parse(Int, m.captures[1]), parse(Int, m.captures[2]))
end

# The raster each `GeometryInputs` field reads, as the ITS_LIVE parameter set names it. Used only to
# repair a URL container stdout truncated; the log is the source everywhere it is intact.
const PARAM_RASTER = Dict("dhdx" => "dhdx", "dhdy" => "dhdy", "vx" => "vx0", "vy" => "vy0",
                          "srx" => "vxSearchRange", "sry" => "vySearchRange",
                          "csminx" => "xMinChipSize", "csminy" => "yMinChipSize",
                          "csmaxx" => "xMaxChipSize", "csmaxy" => "yMaxChipSize",
                          "ssm" => "StableSurface")

"""
    repair_url(url, dem, field) -> String

`url` if it is a raster path, or the one `field` names on the DEM's own tile if stdout truncated it.

One run's log holds `SPS_0Polarization hh` where the search-range URL belongs: a progress line landed
inside the token. The repair is sound because the truncated text still shows the tile prefix and the DEM
URL — intact, and asserted so by `param_urls` — carries the directory, the tile and the resolution, so
only the quantity's own name is supplied. A repair that disagreed with the surviving prefix is refused
rather than trusted.
"""
function repair_url(url::AbstractString, dem::AbstractString, field::AbstractString)
    endswith(url, ".tif") && return String(url)
    name = get(PARAM_RASTER, field, nothing)
    name === nothing && error("cannot repair the truncated URL \"$url\" for `$field`")
    # `.../v001/SPS_0120m_h.tif` gives the directory and the `SPS_0120m` stem.
    m = match(r"^(.*/)([A-Z]+_\d+m)_h\.tif$", dem)
    m === nothing && error(
        "the log's `$field` URL is truncated to \"$url\" and the DEM URL \"$dem\" does not match " *
        "the `<tile>_<res>_h.tif` form the repair needs")
    dir, stem = m.captures[1], m.captures[2]
    # The surviving text must be consistent with the repair, and there are two ways it can be. Usually a
    # prefix of the URL survives — `.../v001/SPS_0Polarization hh` keeps the directory and part of the
    # stem — and it has to match. Sometimes nothing of the URL survives at all, because the noise
    # replaced the whole token: the second of a pair is left as the bare `hh` of a `Polarization hh`
    # line. That is repairable only because the field determines the raster and the *pair's* first half,
    # repaired against the DEM, determines the tile — so the repair is accepted only when the sibling
    # agrees, which `param_urls` arranges by repairing the pair in order.
    want = dir * stem * "_" * name * ".tif"
    # The surviving text has to agree with the repair as far as it goes. It is not a *prefix* of it: the
    # noise begins mid-token, so `.../v001/SPS_0Polarization` agrees through `.../v001/SPS_0` and then
    # diverges. What is checked is that the agreement covers the directory, which is what fixes the tile
    # and the resolution — everything after it is the raster name the field determines. A token the noise
    # replaced entirely, leaving the bare `hh` of a `Polarization hh` line, carries no directory and is
    # accepted on the field alone.
    common = 0
    for (a, b) in zip(url, want)
        a == b || break
        common += 1
    end
    (common >= length(dir) || !occursin('/', url)) || error(
        "the truncated `$field` URL \"$url\" agrees with the repair \"$want\" for only $common " *
        "characters, short of the $(length(dir)) that fix the tile, so the repair would be a guess")
    return want
end

"""
    param_urls(text) -> (dem, Dict)

The parameter-raster URLs the run read, keyed by the `GeometryInputs` field they feed.

Read from the log rather than built from a naming rule, because the rule would be wrong: the
reference velocity rasters are `vx0`/`vy0` where every other quantity drops the digit, and a run
against `vx` instead reads a *different* velocity field — a plausible one, so the error would surface
as a small offset bias rather than as a failure.
"""
function param_urls(text::AbstractString)
    one(label) = begin
        m = match(Regex("^" * escape_regex(label) * raw":\s*(\S+)\s*$", "m"), text)
        m === nothing ? nothing : String(m.captures[1])
    end
    two(label) = begin
        m = match(Regex("^" * escape_regex(label) * raw":\s*(\S+)\s+(\S+)\s*$", "m"), text)
        m === nothing ? nothing : (String(m.captures[1]), String(m.captures[2]))
    end

    dem = one("DEM")
    dem === nothing && error("the log names no DEM")
    endswith(dem, ".tif") || error(
        "the log's DEM URL is \"$dem\", which is not a raster path — container stdout truncated it, " *
        "and the DEM names the tile every other raster is repaired against")
    urls = Dict{String,String}("dem" => dem)
    for (label, fields) in (("Slopes", ("dhdx", "dhdy")), ("Velocities", ("vx", "vy")),
                            ("Search Range", ("srx", "sry")),
                            ("Chip Size Min", ("csminx", "csminy")),
                            ("Chip Size Max", ("csmaxx", "csmaxy")))
        pair = two(label)
        pair === nothing && continue
        urls[fields[1]] = repair_url(pair[1], dem, fields[1])
        urls[fields[2]] = repair_url(pair[2], dem, fields[2])
    end
    ssm = one("Stable Surface Mask")
    ssm === nothing || (urls["ssm"] = repair_url(ssm, dem, "ssm"))
    return dem, urls
end

# ---------------------------------------------------------------------------
# Reading one run
# ---------------------------------------------------------------------------

"""
    goldenrun(c::GoldenCase, n::Integer) -> GoldenRun

The run `n` of case `c`, read from its `capture.log`.

A run is identified as radar by the presence of the `Range:`/`Azimuth:` pair rather than by a
`Radar parameters:` header, since container stdout can overwrite the header — one run on disk carries
` parameters: ` where the rest carry the full word.
"""
function goldenrun(c::GoldenCase, n::Integer)
    dir = run_dir(c, n)
    log = joinpath(dir, "capture.log")
    isfile(log) || error("no capture.log in $dir")
    text = read(log, String)

    epsg = Int(only(logfields!(text, "EPSG", 1)))
    dt = only(logfields!(text, "Repeat Time", 1))
    chip0 = only(logfields!(text, "Smallest Allowable Chip Size in m", 1))
    spacing = only(logfields!(text, "Grid spacing in m", 1))
    # The sentinel comes from the output rasters' own band metadata, with the log as a cross-check.
    # The rasters are authoritative — they are what the comparison reads — and one run's log has the
    # value truncated by an interleaved progress line, so the log cannot be the only source.
    nodata = band_nodata(dir)
    logged = trylogfields(text, "Output Nodata Value", 1)
    logged === nothing || only(logged) == nodata || error(
        "run $n of $(c.product) declares nodata $(only(logged)) in its log but " *
        "$nodata on its output bands; the comparison is a statement about which points are " *
        "missing, so these must agree")
    nodata == EXPECTED_NODATA || error(
        "run $n of $(c.product) uses nodata $nodata, not $EXPECTED_NODATA; every band " *
        "comparison is a statement about which points are missing, so this cannot be defaulted")

    origin = logfields!(text, "Origin index (in DEM) of geogrid", 2)
    nx, ny = geogrid_dims(text)
    pOff, lOff = Int(origin[1]), Int(origin[2])
    window = CartesianIndices((pOff + 1:(pOff + nx), lOff + 1:(lOff + ny)))

    dem_url, urls = param_urls(text)

    # Radar and optical carry different geometry blocks, and the range/azimuth pair is what
    # distinguishes them — not the `Radar parameters:` header, which a progress line can overwrite.
    # `Azimuth:` appears only in the radar block, and it is intact on every radar run on disk.
    # Detection rests on it alone rather than on `Range:` too, since one run has `Range:` truncated.
    az = trylogfields(text, "Azimuth", 2)
    radar = az !== nothing

    printed_origin = printed_spacing = printed_size = nothing
    radar_params = nothing
    if radar
        dims = logfields!(text, "Dimensions", 2)
        inc = only(logfields!(text, "Incidence Angle", 1))
        # The starting range is the first number on `Range:`, and it is intact even on the run whose
        # second number a progress line truncated — so it is read with a leading-anchored pattern
        # rather than the whole-line one.
        starting_range = leading_number(text, "Range")

        # `Range:` prints the starting range and the range sample spacing on one line, and on one run
        # an interleaved progress line truncates the spacing to `2.3295` from `2.32956`. The SLC
        # reader prints the same quantity on its own line, so either can supply it and the two are
        # cross-checked whenever both are intact. A silently truncated `dr` would scale every range
        # index by 1 + 2e-5, which is a fraction of a pixel near the near edge and whole pixels at
        # the far one — a gradient across the swath rather than a visible failure.
        pair = trylogfields(text, "Range", 2)
        reader = trylogfields(text, " -- Slant range spacing in meters", 1)
        dr = if pair !== nothing
            reader === nothing || pair[2] == only(reader) || error(
                "run $n of $(c.product): `Range:` gives dr = $(pair[2]) but the SLC reader " *
                "gives $(only(reader)); these are the same quantity and must agree")
            pair[2]
        elseif reader !== nothing
            only(reader)
        else
            error("""run $n of $(c.product): the `Range:` line is truncated and the SLC reader's
                     `Slant range spacing in meters` is absent, so the range sample spacing cannot
                     be recovered from this log.""")
        end

        # The ground pixel sizes, which `output.txt` carries to more digits than the log's `%g`.
        px, py = output_txt_resolution(dir)
        radar_params = (; starting_range, dr, aztime = az[1], prf = az[2],
                        nsamples = Int(dims[1]), nlines = Int(dims[2]), incidence_deg = inc,
                        ground_range_size = px, azimuth_size = py)
    else
        xc = logfields!(text, "X-direction coordinate", 2)
        yc = logfields!(text, "Y-direction coordinate", 2)
        dims = logfields!(text, "Dimensions", 2)
        printed_origin = (xc[1], yc[1])
        printed_spacing = (xc[2], yc[2])
        printed_size = (Int(dims[1]), Int(dims[2]))
    end

    orbits = sort(filter(f -> endswith(f, ".EOF"), readdir(dir; join = true)))

    return GoldenRun(c, dir, n, radar, epsg, dt, chip0, spacing, nodata, window,
                     dem_url, urls, printed_origin, printed_spacing, printed_size,
                     radar_params, orbits)
end

"""
    goldenruns(; name = nothing) -> Vector{GoldenRun}

Every run on disk, or those of the one case whose product name contains `name`.

Runs are matched to cases by *product name*, taken from the run directory. Not by granule or date: a
scene can appear in two pairs — `LC08_L1TP_060018_20130330_20200912_02_T1` does — so either would be
ambiguous.
"""
function goldenruns(; name = nothing)
    all = cases()
    wanted = name === nothing ? all : cases(name)
    out = GoldenRun[]
    for c in wanted
        d = runs_dir(c)
        isdir(d) || continue
        for sub in sort(readdir(d))
            n = tryparse(Int, sub)
            n === nothing && continue
            isfile(joinpath(d, sub, "capture.log")) || continue
            push!(out, goldenrun(c, n))
        end
    end
    return out
end

"""
    coverage() -> Nothing

Print which of the 22 golden cases have a run on disk, and what each has. The inventory a report
needs in order to state what is *not* covered.
"""
function coverage()
    @printf("%-42s %-10s %-20s %s\n", "case", "platform", "runs", "capture arrays")
    for c in cases()
        d = runs_dir(c)
        if !isdir(d)
            @printf("%-42s %-10s %-20s %s\n", short_name(c), c.platform, "-", "no run directory")
            continue
        end
        ns = sort([n for n in (tryparse(Int, s) for s in readdir(d)) if n !== nothing])
        # Per run, so a run with no capture reads as `0` rather than shifting the column. Only run
        # 200 carries one on several cases; the others compare geogrid output alone.
        caps = [isdir(joinpath(d, string(n), "capture")) ?
                length(readdir(joinpath(d, string(n), "capture"))) : 0 for n in ns]
        @printf("%-42s %-10s %-20s %s\n", short_name(c), c.platform, join(ns, ","),
                join(caps, ","))
    end
    return nothing
end
