# A minimal reader for the `.npz` fixture archives.
#
# The reference fixtures store their arrays as raw little-endian Float64 in a compressed npz —
# exact to the bit, and around twenty times smaller than the same values written as JSON text. An
# npz is a zip of `.npy` members, and a `.npy` is a short ASCII header followed by the raw buffer.
# Reading the one dtype the fixtures use takes less code than taking on a dependency, and keeps the
# test environment to Test, Aqua, JSON3, Proj and StaticArrays.
#
# Deliberately narrow: little-endian Float64, C order, zip entries either stored or deflated.
# Anything else throws rather than being guessed at.

using CodecZlib: DeflateDecompressor, transcode
using ImagePairGeometry
using JSON3
using Proj
using Test

"""
    parse_npy(bytes) -> Array{Float64}

The array in a `.npy` buffer.

Accepts only little-endian Float64 in C order, which is what `gen_geogrid.py` writes. The header is
a Python dict literal; only `descr`, `fortran_order` and `shape` are read from it.
"""
function parse_npy(bytes::Vector{UInt8})
    length(bytes) > 10 || throw(ArgumentError("npy buffer too short: $(length(bytes)) bytes"))
    bytes[1:6] == UInt8[0x93, 'N', 'U', 'M', 'P', 'Y'] ||
        throw(ArgumentError("not an npy buffer: bad magic"))
    major = bytes[7]
    hlen, hstart = if major == 1
        Int(bytes[9]) | (Int(bytes[10]) << 8), 11
    else
        (Int(bytes[9]) | (Int(bytes[10]) << 8) | (Int(bytes[11]) << 16) |
         (Int(bytes[12]) << 24)), 13
    end
    header = String(bytes[hstart:hstart + hlen - 1])

    occursin("'<f8'", header) || throw(ArgumentError(
        "npz fixtures must be little-endian Float64, got header: $header"))
    occursin("'fortran_order': False", header) || throw(ArgumentError(
        "npz fixtures must be in C order, got header: $header"))

    m = match(r"'shape':\s*\(([^)]*)\)", header)
    m === nothing && throw(ArgumentError("no shape in npy header: $header"))
    dims = [parse(Int, strip(s)) for s in split(m.captures[1], ',') if !isempty(strip(s))]

    data = reinterpret(Float64, bytes[hstart + hlen:end])
    n = prod(dims; init = 1)
    length(data) == n || throw(ArgumentError(
        "npy payload is $(length(data)) elements, header says $n"))
    # C order: the last dimension varies fastest, so read into reversed dims. Callers apply
    # `permutedims` to get the orientation this package uses.
    return Array(reshape(data, reverse(dims)...))
end

"""
    load_npz(path) -> Dict{String,Array{Float64}}

Every member of an `.npz` archive, keyed by name with the `.npy` suffix removed.

Reads the zip central directory rather than scanning for local headers, so a member whose local
header omits its size (as a streamed writer may) still reads correctly.
"""
function load_npz(path::AbstractString)
    bytes = read(path)
    n = length(bytes)

    u16(i) = UInt16(bytes[i]) | (UInt16(bytes[i + 1]) << 8)
    u32(i) = UInt32(bytes[i]) | (UInt32(bytes[i + 1]) << 8) |
             (UInt32(bytes[i + 2]) << 16) | (UInt32(bytes[i + 3]) << 24)

    # End-of-central-directory record, searched from the back past any comment.
    eocd = 0
    for i in (n - 21):-1:max(1, n - 65557)
        if u32(i) == 0x06054b50
            eocd = i
            break
        end
    end
    eocd == 0 && throw(ArgumentError("$path is not a zip archive: no end-of-central-directory"))

    nentries = Int(u16(eocd + 10))
    cd = Int(u32(eocd + 16)) + 1

    out = Dict{String,Array{Float64}}()
    p = cd
    for _ in 1:nentries
        u32(p) == 0x02014b50 || throw(ArgumentError("bad central directory entry in $path"))
        method = Int(u16(p + 10))
        csize = Int(u32(p + 20))
        namelen = Int(u16(p + 28))
        extralen = Int(u16(p + 30))
        commentlen = Int(u16(p + 32))
        lho = Int(u32(p + 42)) + 1
        name = String(bytes[p + 46:p + 46 + namelen - 1])

        # Local header: fixed 30 bytes, then the name and extra field.
        u32(lho) == 0x04034b50 || throw(ArgumentError("bad local header for $name in $path"))
        lnamelen = Int(u16(lho + 26))
        lextralen = Int(u16(lho + 28))
        dstart = lho + 30 + lnamelen + lextralen
        raw = bytes[dstart:dstart + csize - 1]

        payload = if method == 0
            raw
        elseif method == 8
            transcode(DeflateDecompressor, raw)
        else
            throw(ArgumentError("unsupported zip compression method $method for $name in $path"))
        end

        out[replace(name, r"\.npy$" => "")] = parse_npy(Vector{UInt8}(payload))
        p += 46 + namelen + extralen + commentlen
    end
    return out
end

# ---------------------------------------------------------------------------
# Shared fixture assembly.
#
# `geogrid.jl` and `blocks.jl` both build inputs from the same `geogrid.json` / `geogrid_arrays.npz`
# pair, and the 12-field `GeometryInputs` call is the same in each. Written once here — the file both
# already include — so a fixture schema change is one edit rather than three.

"""
    fixture_inputs(arrays, case, window) -> GeometryInputs

The `GeometryInputs` for a fixture case, over `window`.

A band absent from the case's `input_names` becomes `nothing`, which is how the driver learns not to
compute the outputs depending on it.

`parse_npy` reverses the C-order dimensions, so a member arrives with the x index varying along the
first axis — the orientation this package uses — and needs no transpose.
"""
function fixture_inputs(arrays::Dict{String,Array{Float64}}, case, window)
    present = Set(String.(case.input_names))
    band(k) = k in present ? arrays["$(case.name)/input/$k"][window] : nothing
    return GeometryInputs(dem = band("dem"),
                          dhdx = band("dhdx"), dhdy = band("dhdy"),
                          vx = band("vx"), vy = band("vy"),
                          srx = band("srx"), sry = band("sry"),
                          csminx = band("csminx"), csminy = band("csminy"),
                          csmaxx = band("csmaxx"), csmaxy = band("csmaxy"),
                          ssm = band("ssm"))
end

"""
    fixture_scene(case) -> NamedTuple

The grid, image coordinate system and pair a fixture case describes, plus a `transform` thunk.

`transform` builds a fresh `TransformPair` each call: for a same-CRS case an `IdentityTransform`, and
otherwise a PROJ pair. A thunk rather than a value because a threaded run needs one PROJ context per
task, and because `footprint_bounds` and the run itself each want one.
"""
function fixture_scene(case)
    img, dem = case.image, case.dem
    coord = ProjectedCoordinate(
        origin = (Float64(img.origin[1]), Float64(img.origin[2])),
        spacing = (Float64(img.spacing[1]), Float64(img.spacing[2])),
        size = (Int(img.size[1]), Int(img.size[2])))
    grid = MapGrid(geotransform = ntuple(i -> Float64(dem.geotransform[i]), 6),
                   size = (Int(dem.size[1]), Int(dem.size[2])), crs = Int(dem.epsg))
    # The fixture drives the reference by setting `startingX`/`startingY` directly, so the pair here
    # is the image against itself and contributes only the interval.
    fp = ImageFootprint(origin = coord.origin, spacing = coord.spacing, size = coord.size)
    pair = coregister(fp, fp; dt = Float64(case.dt))
    same = Int(img.epsg) == Int(dem.epsg)
    # Each call builds its own PROJ context, since a threaded run calls this once per task and PROJ
    # documents a context as usable from one thread at a time. Building on the shared global context
    # instead corrupts its SQLite handle, which surfaces as a `bad parameter or other API misuse` error
    # from whichever task lost the race.
    #
    # Contexts are deliberately never destroyed: a `Transformation`'s finalizer calls `proj_destroy` on
    # its `PJ*`, which must not run after its context is freed.
    transform = function ()
        same && return transform_pair(IdentityTransform())
        ctx = Proj.proj_context_create()
        # `Proj.__init__` points only the *global* context at the bundled `proj.db`, so a self-created
        # one cannot find the database unless told where it is.
        Proj.proj_context_set_search_paths(1, [Proj.PROJ_DATA[]], ctx)
        # Grids fetched over the network would make results depend on what happened to be cached.
        Proj.proj_context_set_enable_network(false, ctx)
        g, i = "EPSG:$(Int(dem.epsg))", "EPSG:$(Int(img.epsg))"
        return TransformPair(Proj.Transformation(g, i; ctx),
                             Proj.Transformation(i, g; ctx))
    end
    return (; coord, grid, pair, transform, same,
            params = GeometryParams(chip_size_0 = Float64(case.chip_size_0)))
end

"""
    GEOGRID_FIXTURE, GEOGRID_ARRAYS

The recorded reference outputs: the scalars from `reference/geogrid.json` and the band arrays from
`reference/geogrid_arrays.npz`.

Read once here rather than per test file, since more than one suite drives its cases from them.
"""
const GEOGRID_FIXTURE = JSON3.read(read(joinpath(@__DIR__, "reference", "geogrid.json"), String))
const GEOGRID_ARRAYS = load_npz(joinpath(@__DIR__, "reference", "geogrid_arrays.npz"))

"""
    setup_case(name) -> NamedTuple

Everything needed to run the fixture case called `name`: its grid, pair, image coordinate system,
window, parameters and input rasters, plus a `makepair` thunk.

`makepair` is a thunk, called once per task: a PROJ transformation wraps a `PJ*` on a context that
PROJ documents as usable from one thread at a time, and building two concurrently on the shared
global context corrupts its SQLite handle.
"""
function setup_case(name)
    c = only(filter(x -> x.name == name, collect(GEOGRID_FIXTURE.cases)))
    s = fixture_scene(c)
    win = grid_window(s.grid, footprint_bounds(s.transform(), s.coord))
    # The two EPSG codes as integers. `grid.crs` holds a `GeoFormat` and a `ProjectedCoordinate`
    # carries no CRS at all, so a test building its own transform reads them from here.
    return (; s.grid, s.pair, s.coord, win, s.params, s.same,
            grid_epsg = Int(c.dem.epsg), image_epsg = Int(c.image.epsg),
            inputs = fixture_inputs(GEOGRID_ARRAYS, c, win), makepair = s.transform)
end
