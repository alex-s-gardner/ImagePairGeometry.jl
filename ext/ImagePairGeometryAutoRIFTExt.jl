module ImagePairGeometryAutoRIFTExt

# Handing a `PairGeometry` to the correlator.
#
# This is a contract negotiation, not a type conversion, and each clause is a place a silent error
# could hide:
#
# Index base. Geogrid's pixel index is zero-based — `round((x - startingX) / XSize)`, bounds-tested
# against `0` and `nPixels - 1` (`geogridOptical.cpp:723-724,775`). `AutoRIFT.PointSet` carries
# one-based positions, so every index gains 1.
#
# The half pixel. `autoRIFT.py:890` stores `round(xGrid) + 0.5`, and AutoRIFT.jl reproduces that at
# correlation time in `_shift_points` for every pyramid level alike (`src/multichip.jl:440-446`).
# So the `+ 0.5` must *not* be applied here: doing it twice moves every search centre a pixel.
#
# Missing values. Geogrid marks a point outside the image with `-32767`; AutoRIFT skips a point whose
# search radius is zero (`src/points.jl:127-128`). Passing the sentinel through as a radius would
# make it negative, and `gridpoints`' margin logic sizes itself from `maximum(radius)`, so it would
# be mis-sized rather than caught.
#
# What does not fit. `PointSet` holds ten fields; geogrid produces eighteen numbers per point. The
# displacement-to-velocity operator, the scale factors, the stable-surface mask and the y chip-size
# bounds have nowhere to go, and downstream velocity conversion needs them — so they are returned
# alongside rather than dropped.
#
# Version requirement. The per-point chip-size bounds go through `pointset`'s `chip_size_min_x` and
# `chip_size_max_x` keywords, which exist on no released AutoRIFT. `AutoRIFT` is unregistered, so
# `[compat]` cannot express the requirement: against a checkout whose `PointSet` lacks those fields
# this loads and then throws a `MethodError` on the first call.

using ImagePairGeometry
using ImagePairGeometry: PairGeometry, chip_size_pixels, ProjectedCoordinate,
                         RadarCoordinate
using AutoRIFT

"""
    AutoRIFT.pointset(g::PairGeometry; chip_size = nothing, chip_size_0 = 240.0,
                      pixel_size = nothing) -> PointSet{2}

The search grid in `g`, as an `AutoRIFT.PointSet`.

Pixel positions become one-based (geogrid's are zero-based) and a point that fell outside the image
gets a search radius of zero, which is how AutoRIFT marks a point to skip.

The half-pixel offset the reference bakes into its grid is *not* applied: AutoRIFT.jl adds it at
correlation time for every pyramid level, so applying it here as well would displace every search
centre by a pixel.

`chip_size` sets the base chip extent in pixels. Given `pixel_size` instead, it is derived as the
reference does — `ceil(chip_size_0 / pixel_size / 4) * 4` — from a chip size in meters. One of the
two is required, since neither is recoverable from `g`.

Only part of `g` fits a `PointSet`. Use [`velocity_conversion`](@ref) for the operator, the scale
factors and the mask, which downstream velocity conversion needs.
"""
function AutoRIFT.pointset(g::PairGeometry; chip_size = nothing, chip_size_0 = 240.0,
                           pixel_size = nothing, coordinate = nothing)
    base = if chip_size !== nothing
        Int(chip_size)
    elseif pixel_size !== nothing
        chip_size_pixels(chip_size_0, pixel_size)
    else
        throw(ArgumentError(
            "pointset needs the base chip extent: pass `chip_size` in pixels, or `pixel_size` in " *
            "meters to derive it from `chip_size_0`. Neither is recoverable from a PairGeometry, " *
            "which stores chip size bounds but not the base."))
    end

    sentinel = Int32(g.nodata.output)
    valid = g.location_x .!= sentinel

    # One-based, and a skipped point still needs a coordinate: `PointSet` has no missing value, so
    # its position is arbitrary and its zero radius is what excludes it.
    x = [v ? Float64(l + 1) : 1.0 for (v, l) in zip(valid, g.location_x)]
    y = [v ? Float64(l + 1) : 1.0 for (v, l) in zip(valid, g.location_y)]

    # A radius is zero where the point is invalid, where the search extent itself is missing, or
    # where geogrid computed no extent at all — each meaning "do not search here".
    rad(band) = [(v && b != sentinel && b > 0) ? Int(b) : 0 for (v, b) in zip(valid, band)]
    rx = rad(g.search_x)
    ry = rad(g.search_y)

    # An absent search-range raster leaves the band uniformly sentinel, which would skip every
    # point. That is a missing input rather than a grid of skips, so say so.
    if all(iszero, rx) && !isempty(rx)
        throw(ArgumentError(
            "every search radius is zero, so no point would be correlated. The PairGeometry has " *
            "no search-range band — it was computed without `srx`/`sry` (and their required " *
            "`dhdx`/`dhdy`). Supply them, or build the PointSet with an explicit radius."))
    end

    prior(band, flip) = [(v && b != sentinel) ? flip * Float64(b) : 0.0
                        for (v, b) in zip(valid, band)]

    # The radar path negates the y prior; the projected path does not. `testautoRIFT.py:405-407` does
    # it under `if optflag == 0`, and its own comment says why: "convert azimuth offset to vertical
    # offset as used in autoRIFT convention". Azimuth increases along the track while AutoRIFT's `Dy`
    # points down a north-up raster, so the two disagree in sign. A projected image's `+y`-down
    # convention already matches, and negating there would send the correlator the wrong way.
    dy_flip = _prior_sign(coordinate)

    # Chip-size bounds are per point; zero means unbounded in AutoRIFT, which is what a missing
    # bound should mean.
    bound(band) = [(v && b != sentinel && b > 0) ? Int(b) : 0 for (v, b) in zip(valid, band)]

    return AutoRIFT.pointset(x, y;
                             search_radius_x = rx, search_radius_y = ry,
                             chip_size = base,
                             dx_prior = prior(g.offset_x, 1.0),
                             dy_prior = prior(g.offset_y, dy_flip),
                             chip_size_min_x = bound(g.chip_min_x),
                             chip_size_max_x = bound(g.chip_max_x))
end

# Which sign the y prior and the returned y displacement carry, by coordinate system.
#
# A dispatch rather than a keyword flag: the two paths disagree, the disagreement is silent — a wrong
# sign sends the correlator searching the wrong way and returns a plausible velocity — and the answer
# is a property of the coordinate system rather than a choice.
#
# `nothing` throws. A `PairGeometry` does not carry the coordinate that produced it, so there is no
# value to infer from, and defaulting to either path would be wrong half the time.
_prior_sign(::ProjectedCoordinate) = 1.0
_prior_sign(::RadarCoordinate) = -1.0
_prior_sign(::Nothing) = throw(ArgumentError(
    "pass `coordinate` — the `ProjectedCoordinate` or `RadarCoordinate` the geometry was computed " *
    "for. The y sign differs between them: the radar path negates the azimuth prior to reach " *
    "AutoRIFT's north-up convention (`testautoRIFT.py:405-407`) and the projected path does not. " *
    "A `PairGeometry` does not record which it came from, and guessing would be wrong half the time."))
_prior_sign(x) = throw(ArgumentError(
    "`coordinate` must be a ProjectedCoordinate or a RadarCoordinate, got $(typeof(x))"))

"""
    velocity_conversion(g::PairGeometry; coordinate) -> NamedTuple

The parts of `g` an `AutoRIFT.PointSet` cannot hold, which converting a correlated displacement to
a map velocity needs.

# Fields
- `off2vx`, `off2vy`: the two-by-two operator, each as `(dx, dy)` coefficient arrays.
- `scale`: the scale factors, as `(x, y)`.
- `stable_surface`: the stable-surface mask, `true` where stable. Points that are missing or
  outside the image are `false`.
- `chip_scale_y`: median ratio of the y to the x chip-size bound, the aspect ratio the reference
  derives as `ScaleChipSizeY` (`testautoRIFT.py:376`). `NaN` where no chip-size band is present.
- `dy_sign`: the factor a displacement from the correlator needs before use — `-1.0` on the radar
  path, `+1.0` on the projected one. See below.
- `nodata`: the sentinel marking a missing entry in the float arrays.

Given a displacement `(dx, dy)` in pixels, in AutoRIFT's convention where `+y` points north:

```julia
c = velocity_conversion(g)
vx = c.off2vx.dx .* (dx .* c.scale.x) .+ c.off2vx.dy .* (dy .* c.scale.y)
vy = c.off2vy.dx .* (dx .* c.scale.x) .+ c.off2vy.dy .* (dy .* c.scale.y)
```

`dy_sign` is the factor to apply to a displacement coming *back* from the correlator before using it
here, and it is the counterpart of the prior's negation in [`pointset`](@ref): `-1.0` on the radar path
and `+1.0` on the projected one, matching `testautoRIFT.py:790-791`, which negates under the same
`optical_flag == 0` guard as `:405-407`. Applying it is the caller's step, because the caller is what
holds the correlator's output — but the value is supplied here so it need not be re-derived.
"""
function velocity_conversion(g::PairGeometry; coordinate = nothing)
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
            dy_sign = _prior_sign(coordinate),
            nodata = g.nodata.output)
end

# Sorting a copy rather than taking a dependency on Statistics for one call.
function _median(v::Vector{Float64})
    s = sort(v)
    n = length(s)
    return isodd(n) ? s[(n + 1) ÷ 2] : (s[n ÷ 2] + s[n ÷ 2 + 1]) / 2
end

export velocity_conversion

end
