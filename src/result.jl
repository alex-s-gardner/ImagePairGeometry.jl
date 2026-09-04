# The result: one array per reference output band.
#
# Named after the reference's nine files and their bands, so writing them out and comparing them
# against the reference are both one-liners, and so a transposed pair is a compile error rather
# than a silent axis swap.
#
# Struct of arrays rather than an array of structs: every consumer wants a whole band at once — the
# GeoTIFF writer, the bitwise comparison, the correlator that takes the grid as two matrices.
#
# `Int32` where the reference writes `GDT_Int32`, because that is what bit-exactness is asserted
# against; widening to `Int` would put the `-32767` sentinel in two representations.

"""
    PairGeometry

Per-grid-point geometry of an image pair, one array per output quantity.

All arrays share the axes of the grid window the result was computed over.

# Integer fields, sentinel `-32767`
- `location_x`, `location_y`: zero-based pixel index of the grid point in the image.
- `offset_x`, `offset_y`: expected displacement in pixels, from the reference velocity field.
- `search_x`, `search_y`: search half-extent in pixels.
- `chip_min_x`, `chip_min_y`, `chip_max_x`, `chip_max_y`: chip size bounds in pixels.
- `stable_surface`: stable-surface mask.

# Float fields, sentinel `-32767.0`
- `off2vx_dx`, `off2vx_dy`, `off2vy_dx`, `off2vy_dy`: the operator converting a pixel displacement
  to a map velocity. Displacement `(dx, dy)` gives `vx = off2vx_dx * dx + off2vx_dy * dy` and
  likewise for `vy`, after scaling the displacement by the scale factors.
- `off2vx_dr`, `off2vy_dr`: velocity per pixel of displacement along the image's *own* axes, the
  radar path's third off2vel band (`geogridRadar.cpp:1185-1187`). Left at their sentinel on the
  projected path, which writes two-band off2vel files.
- `scale_x`, `scale_y`: ratio of true ground distance to nominal pixel spacing.

# Metadata
- `geotransform`, `crs`: georeferencing of the window. The CRS is a
  `GeoFormatTypes.GeoFormat` or `nothing`, inherited from the grid; read it with
  `GeoInterface.crs`.
- `window`: the grid indices covered.
- `nodata`: the policy the sentinels come from.
- `coordinate`: the [`AbstractImageCoordinate`](@ref) the geometry was computed for. Carried as a type
  parameter so consumers that differ between the two paths — the output band layout, the sign of the
  correlator's y prior — dispatch on it rather than inferring or being told.

Field names map to the reference's files as `location` → `window_location.tif`, `offset` →
`window_offset.tif`, `search` → `window_search_range.tif`, `chip_min`/`chip_max` →
`window_chip_size_min.tif`/`_max.tif`, `stable_surface` →
`window_stable_surface_mask.tif`, `off2vx`/`off2vy` → `window_rdr_off2vel_x_vec.tif`/`_y_vec.tif`,
`scale` → `window_scale_factor.tif`.
"""
struct PairGeometry{I<:AbstractMatrix{Int32},F<:AbstractMatrix{Float64},C,
                    K<:AbstractImageCoordinate}
    location_x::I
    location_y::I
    offset_x::I
    offset_y::I
    search_x::I
    search_y::I
    chip_min_x::I
    chip_min_y::I
    chip_max_x::I
    chip_max_y::I
    stable_surface::I

    off2vx_dx::F
    off2vx_dy::F
    off2vy_dx::F
    off2vy_dy::F
    off2vx_dr::F
    off2vy_dr::F
    scale_x::F
    scale_y::F

    geotransform::NTuple{6,Float64}
    crs::C
    window::CartesianIndices{2}
    nodata::NoDataPolicy
    coordinate::K
end

"""
    INT_BANDS

Names of the `Int32` fields of [`PairGeometry`](@ref), in the reference's band order.
"""
const INT_BANDS = (:location_x, :location_y, :offset_x, :offset_y, :search_x, :search_y,
                   :chip_min_x, :chip_min_y, :chip_max_x, :chip_max_y, :stable_surface)

"""
    FLOAT_BANDS

Names of the `Float64` fields of [`PairGeometry`](@ref), in the reference's band order.
"""
const FLOAT_BANDS = (:off2vx_dx, :off2vx_dy, :off2vy_dx, :off2vy_dy, :off2vx_dr, :off2vy_dr,
                     :scale_x, :scale_y)

"""
    REFERENCE_FILES

Each of the reference's nine output files, with the [`PairGeometry`](@ref) fields holding its
bands in order — the **projected** path's band layout.

The basis for writing GeoTIFFs that a downstream reader consuming the reference's output can read
unchanged, and for comparing band by band against it.

The two off2vel files have two bands here and three on the radar path
(`geogridOptical.cpp:461` versus `geogridRadar.cpp:634,652`). That difference is not cosmetic: a
reader written against the projected output indexes bands positionally, so writing three where the
reference writes two would break it. Use [`reference_files`](@ref) to get the layout for a given
coordinate system rather than assuming this one.
"""
const REFERENCE_FILES = (
    "window_location.tif" => (:location_x, :location_y),
    "window_offset.tif" => (:offset_x, :offset_y),
    "window_search_range.tif" => (:search_x, :search_y),
    "window_chip_size_min.tif" => (:chip_min_x, :chip_min_y),
    "window_chip_size_max.tif" => (:chip_max_x, :chip_max_y),
    "window_stable_surface_mask.tif" => (:stable_surface,),
    "window_rdr_off2vel_x_vec.tif" => (:off2vx_dx, :off2vx_dy),
    "window_rdr_off2vel_y_vec.tif" => (:off2vy_dx, :off2vy_dy),
    "window_scale_factor.tif" => (:scale_x, :scale_y),
)

"""
    RADAR_REFERENCE_FILES

[`REFERENCE_FILES`](@ref) with the radar path's three-band off2vel layout.

Band 3 of each is the velocity per pixel of displacement along that image axis
(`geogridRadar.cpp:1185-1187`), which the projected path has no equivalent of.
"""
const RADAR_REFERENCE_FILES = (
    "window_location.tif" => (:location_x, :location_y),
    "window_offset.tif" => (:offset_x, :offset_y),
    "window_search_range.tif" => (:search_x, :search_y),
    "window_chip_size_min.tif" => (:chip_min_x, :chip_min_y),
    "window_chip_size_max.tif" => (:chip_max_x, :chip_max_y),
    "window_stable_surface_mask.tif" => (:stable_surface,),
    "window_rdr_off2vel_x_vec.tif" => (:off2vx_dx, :off2vx_dy, :off2vx_dr),
    "window_rdr_off2vel_y_vec.tif" => (:off2vy_dx, :off2vy_dy, :off2vy_dr),
    "window_scale_factor.tif" => (:scale_x, :scale_y),
)

"""
    reference_files(c::AbstractImageCoordinate) -> Tuple

The output file and band layout for a coordinate system: [`REFERENCE_FILES`](@ref) for a projected
image, [`RADAR_REFERENCE_FILES`](@ref) for a radar one.

Dispatching rather than branching on a flag, so a caller cannot write the wrong band count for the
path it is on.
"""
reference_files(::ProjectedCoordinate) = REFERENCE_FILES
reference_files(::RadarCoordinate) = RADAR_REFERENCE_FILES

"""
    reference_files(g::PairGeometry) -> Tuple

The layout for a result, from the coordinate system it was computed for.

This is how [`write_geotiffs`](@ref) picks a band count. Getting it wrong is silent — a reader built
against the reference's projected output indexes bands positionally, so an extra band shifts
everything after it — which is why the result carries its coordinate rather than the layout being
inferred from the bands. Inferring it from whether `off2vx_dr` is all-sentinel misclassifies a radar
result computed without a slope raster, since the whole slope branch that fills that band is skipped.
"""
reference_files(g::PairGeometry) = reference_files(g.coordinate)

GeoInterface.crs(r::PairGeometry) = r.crs

"""
    velocity_conversion(g::PairGeometry; coordinate = g.coordinate) -> NamedTuple

The parts of `g` that turn a measured pixel displacement into a map velocity.

A correlator returns a displacement in pixels; these are the per-point quantities that convert it,
gathered in one place so a caller need not know which fields of a [`PairGeometry`](@ref) participate.

# Fields
- `off2vx`, `off2vy`: the two-by-two operator, each as `(dx, dy)` coefficient arrays.
- `scale`: the scale factors, as `(x, y)`.
- `stable_surface`: the stable-surface mask, `true` where stable. Points that are missing or
  outside the image are `false`.
- `chip_scale_y`: median ratio of the y to the x chip-size bound, the aspect ratio the reference
  derives as `ScaleChipSizeY` (`testautoRIFT.py:376`). `NaN` where no chip-size band is present.
- `dy_sign`: the factor a measured y displacement needs before use. See below.
- `nodata`: the sentinel marking a missing entry in the float arrays.

Given a displacement `(dx, dy)` in pixels, in the convention where `+y` points north:

```julia
c = velocity_conversion(g)
vx = c.off2vx.dx .* (dx .* c.scale.x) .+ c.off2vx.dy .* (dy .* c.scale.y)
vy = c.off2vy.dx .* (dx .* c.scale.x) .+ c.off2vy.dy .* (dy .* c.scale.y)
```

`dy_sign` is [`y_displacement_sign`](@ref) of the coordinate system: the factor a measured y
displacement needs before the expressions above, `-1.0` on the radar path and `+1.0` on the projected
one. Applying it is the caller's step, since the caller is what holds the measurement, but the value
is supplied here so it need not be re-derived.
"""
function velocity_conversion(g::PairGeometry; coordinate = g.coordinate)
    sentinel = Int32(g.nodata.output)
    valid = g.location_x .!= sentinel

    stable = [(v && m != sentinel && m != 0) for (v, m) in zip(valid, g.stable_surface)]

    # The reference takes this over points where both bounds are present (`testautoRIFT.py:376`).
    ratios = Float64[]
    for (a, b) in zip(g.chip_min_x, g.chip_min_y)
        (a != sentinel && b != sentinel && a != 0) && push!(ratios, b / a)
    end
    chip_scale_y = isempty(ratios) ? NaN : _median(ratios)

    return (off2vx = (dx = g.off2vx_dx, dy = g.off2vx_dy),
            off2vy = (dx = g.off2vy_dx, dy = g.off2vy_dy),
            scale = (x = g.scale_x, y = g.scale_y),
            stable_surface = stable,
            chip_scale_y = chip_scale_y,
            dy_sign = y_displacement_sign(coordinate),
            nodata = g.nodata.output)
end

# Sorted copy: `Statistics` is a standard library but not a declared dependency, and this is the
# only call.
function _median(v::Vector{Float64})
    s = sort(v)
    n = length(s)
    return isodd(n) ? s[(n + 1) ÷ 2] : (s[n ÷ 2] + s[n ÷ 2 + 1]) / 2
end

Base.size(r::PairGeometry) = size(r.location_x)
Base.axes(r::PairGeometry) = axes(r.location_x)
Base.eachindex(r::PairGeometry) = eachindex(r.location_x)

"""
    allocate_geometry(window, geotransform, crs, nodata, coordinate) -> PairGeometry

A [`PairGeometry`](@ref) sized to `window`, with every band filled with its sentinel.

Prefilling means a band the inputs do not support — the reference writes no file at all in that
case — is uniformly nodata rather than uninitialized, and a point skipped for being out of bounds
needs no explicit write.
"""
function allocate_geometry(window::CartesianIndices{2}, geotransform::NTuple{6,Float64},
                           crs, nodata::NoDataPolicy, coordinate::AbstractImageCoordinate)
    sz = size(window)
    # `Val` so the tuple lengths are compile-time constants. With a plain `length(INT_BANDS)` the
    # result is `Tuple{Vararg{Matrix{Int32}}}`, of unknown length, and splatting that into the
    # constructor becomes a call `juliac --trim` cannot resolve.
    ints = ntuple(_ -> fill(Int32(nodata.output), sz), Val(length(INT_BANDS)))
    floats = ntuple(_ -> fill(nodata.output, sz), Val(length(FLOAT_BANDS)))
    return PairGeometry(ints..., floats..., geotransform, crs, window, nodata, coordinate)
end

"""
    npoints(r::PairGeometry) -> Int

Number of grid points in the result, valid or not.
"""
npoints(r::PairGeometry) = length(r.location_x)

"""
    nvalid(r::PairGeometry) -> Int

Number of grid points that mapped inside the image, i.e. whose location is not the sentinel.
"""
nvalid(r::PairGeometry) = count(!=(Int32(r.nodata.output)), r.location_x)

function Base.show(io::IO, ::MIME"text/plain", r::PairGeometry)
    nx, ny = size(r)
    v = nvalid(r)
    println(io, "PairGeometry: $(nx)x$(ny) grid points, $v valid ",
            "($(round(100 * v / max(npoints(r), 1); digits = 1))%)")
    println(io, "  window: ", r.window)
    println(io, "  crs:    ", r.crs)
    print(io, "  nodata: ", r.nodata.output)
end
