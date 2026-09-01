# The driver: the per-point loop, and the inputs it reads.
#
# The reference reads one grid row at a time (`geogridOptical.cpp:519-...`, a `RasterIO` call per
# row per raster) and writes one row at a time. That is already streaming, so blocking is a matter
# of choosing the unit rather than restructuring: here the loop runs over an arbitrary window of the
# grid, and `pairgeometry` over a whole window is exactly the same computation as over each of its
# sub-windows. Points are independent, so neither blocking nor threading can change a result.
#
# Inputs arrive as plain matrices covering the window. Reading them from disk lives in an extension,
# so the loop itself pulls in no IO dependency and stays trim-compilable.

"""
    GeometryInputs(; dem, dhdx = nothing, dhdy = nothing, vx = nothing, vy = nothing,
                     srx = nothing, sry = nothing, csminx = nothing, csminy = nothing,
                     csmaxx = nothing, csmaxy = nothing, ssm = nothing)

The per-grid-point input rasters, covering the window being computed.

Only `dem` is required. Each other quantity is `nothing` when not supplied, and the outputs
depending on it are then left as nodata — matching the reference, which writes no file at all for
those bands.

Dependencies among them, as the reference enforces them:

- Slope (`dhdx`, `dhdy`) gates the displacement-to-velocity operator, the scale factors, the
  expected offset and the search extent. Without it the reference writes none of them, since all
  four need the surface normal.
- Velocity (`vx`, `vy`) gates the expected offset; search range (`srx`, `sry`) gates the search
  extent. Both additionally require slope.
- Chip size and the stable-surface mask are independent of slope.

All arrays must share axes with each other and with the window.
"""
struct GeometryInputs{D,S,V,R,Cn,Cx,M}
    dem::D
    dhdx::S
    dhdy::S
    vx::V
    vy::V
    srx::R
    sry::R
    csminx::Cn
    csminy::Cn
    csmaxx::Cx
    csmaxy::Cx
    ssm::M
end

function GeometryInputs(; dem, dhdx = nothing, dhdy = nothing, vx = nothing, vy = nothing,
                        srx = nothing, sry = nothing, csminx = nothing, csminy = nothing,
                        csmaxx = nothing, csmaxy = nothing, ssm = nothing)
    ax = axes(dem)
    for (name, a) in ((:dhdx, dhdx), (:dhdy, dhdy), (:vx, vx), (:vy, vy), (:srx, srx),
                      (:sry, sry), (:csminx, csminx), (:csminy, csminy), (:csmaxx, csmaxx),
                      (:csmaxy, csmaxy), (:ssm, ssm))
        a === nothing && continue
        axes(a) == ax || throw(DimensionMismatch(
            "GeometryInputs field `$name` has axes $(axes(a)), expected $ax to match `dem`"))
    end
    # Paired rasters are meaningless alone: the reference reads the second only when the first is
    # given, so a lone one would be silently ignored.
    for (a, b, na, nb) in ((dhdx, dhdy, :dhdx, :dhdy), (vx, vy, :vx, :vy), (srx, sry, :srx, :sry),
                           (csminx, csminy, :csminx, :csminy),
                           (csmaxx, csmaxy, :csmaxx, :csmaxy))
        (a === nothing) == (b === nothing) || throw(ArgumentError(
            "GeometryInputs `$na` and `$nb` must be given together, got " *
            "$na = $(a === nothing ? "nothing" : "an array"), " *
            "$nb = $(b === nothing ? "nothing" : "an array")"))
    end
    (vx === nothing || dhdx !== nothing) || throw(ArgumentError(
        "GeometryInputs `vx`/`vy` require `dhdx`/`dhdy`: the expected offset needs the surface " *
        "normal, and the reference writes no offset without a slope raster"))
    (srx === nothing || dhdx !== nothing) || throw(ArgumentError(
        "GeometryInputs `srx`/`sry` require `dhdx`/`dhdy`: the search extent needs the surface " *
        "normal, and the reference writes no search range without a slope raster"))
    return GeometryInputs(dem, dhdx, dhdy, vx, vy, srx, sry, csminx, csminy, csmaxx, csmaxy, ssm)
end

"""
    GeometryParams(; chip_size_0 = 240.0, scaling = SearchRangeScaling())

Scalar parameters of the computation.

- `chip_size_0`: reference chip size in meters, 240.0 in production (`testGeogrid.py:358`).
- `scaling`: the search-range inflation, see [`SearchRangeScaling`](@ref).
"""
struct GeometryParams{T<:Real}
    chip_size_0::Float64
    scaling::SearchRangeScaling{T}
end

GeometryParams(; chip_size_0 = 240.0, scaling = SearchRangeScaling()) =
    GeometryParams(Float64(chip_size_0), scaling)

"""
    pairgeometry(grid, pair, inputs; transform = IdentityTransform(), window = nothing,
                 params = GeometryParams(), nodata = nodata_from(nothing)) -> PairGeometry

Geometry of `pair` at every point of `grid` covered by `window`.

`grid` is the target [`MapGrid`](@ref), `pair` a [`CoregisteredPair`](@ref), and `inputs` a
[`GeometryInputs`](@ref) whose arrays cover `window`. `transform` maps grid coordinates to image
coordinates — an [`IdentityTransform`](@ref), a `TransformPair`, or a `Proj.Transformation`, whose
inverse is derived when only one direction is given.

`window` defaults to the whole grid intersected with the pair's footprint, via
[`footprint_bounds`](@ref) and [`grid_window`](@ref).

Every point is independent, so computing a window in pieces gives the same answer as computing it
whole, so blocking and threading cost nothing in fidelity.

# Example

```jldoctest
julia> using ImagePairGeometry

julia> grid = MapGrid(geotransform = (0.0, 10.0, 0.0, 100.0, 0.0, -10.0), size = (10, 10));

julia> a = ImageFootprint(origin = (5.0, 95.0), spacing = (5.0, -5.0), size = (16, 16));

julia> pair = coregister(a, a; dt = 86400.0);

julia> win = grid_window(grid, footprint_bounds(IdentityTransform(), pair.coordinate))
CartesianIndices((1:8, 1:8))

julia> r = pairgeometry(grid, pair, GeometryInputs(dem = zeros(size(win))); window = win);

julia> size(r), nvalid(r)
((8, 8), 64)
```
"""
function pairgeometry(grid::MapGrid, pair::CoregisteredPair, inputs::GeometryInputs;
                      transform = IdentityTransform(), window = nothing,
                      params::GeometryParams = GeometryParams(),
                      nodata::NoDataPolicy = nodata_from(nothing))
    tf = transform_pair(transform)
    coord = pair.coordinate
    win = window === nothing ? grid_window(grid, footprint_bounds(tf.forward, coord)) : window

    size(inputs.dem) == size(win) || throw(DimensionMismatch(
        "GeometryInputs arrays are $(size(inputs.dem)) but the window is $(size(win)); the " *
        "inputs must cover exactly the window being computed"))

    result = allocate_geometry(win, window_geotransform(grid, win), grid.crs, nodata)
    _fill_geometry!(result, grid, coord, pair.dt, inputs, tf, params, nodata, win)
    return result
end

# Split from `pairgeometry` so the loop specializes on the concrete input and transform types
# rather than on the keyword-argument call site.
function _fill_geometry!(r::PairGeometry, grid::MapGrid, coord::ProjectedCoordinate, dt::Float64,
                         inp::GeometryInputs, tf::TransformPair, params::GeometryParams,
                         nd::NoDataPolicy, win::CartesianIndices{2})
    has_slope = inp.dhdx !== nothing
    has_vel = inp.vx !== nothing
    has_sr = inp.srx !== nothing
    has_csmin = inp.csminx !== nothing
    has_csmax = inp.csmaxx !== nothing
    has_ssm = inp.ssm !== nothing

    # Interval-dependent search-range inflation is per pair, not per point.
    sr_scale = searchrange_scale(params.scaling, dt)

    # Chip size in pixels depends only on the image's pixel size.
    pix_x = chip_size_pixels(params.chip_size_0, xsize(coord))
    pix_y = chip_size_pixels(params.chip_size_0, ysize(coord))

    out = Int32(nd.output)
    fout = nd.output

    for (k, idx) in enumerate(win)
        i, j = idx.I
        gx, gy = gridpoint_center(grid, i, j)
        gz = Float64(inp.dem[k])

        normal = has_slope ? surface_normal(inp.dhdx[k], inp.dhdy[k]) : NO_NORMAL
        g = pointgeometry(tf, gx, gy, gz, coord, normal)

        xind, yind = pixel_index(g, coord)
        # Out of bounds leaves every band at its sentinel, as prefilled.
        inbounds(xind, yind, coord) || continue

        r.location_x[k] = cround32(xind)
        r.location_y[k] = cround32(yind)

        if has_slope
            if has_vel
                vx = Float64(inp.vx[k])
                # The reference tests only the x component, and treats a missing velocity as a
                # zero offset rather than as nodata.
                if ismissingval(nd, vx)
                    r.offset_x[k] = Int32(0)
                    r.offset_y[k] = Int32(0)
                else
                    vel = close_slope_parallel(vx, Float64(inp.vy[k]), normal)
                    ox, oy = pixel_offset(vel, g, dt)
                    r.offset_x[k] = cround32(ox)
                    r.offset_y[k] = cround32(oy)
                end
            end

            # The operator is singular for a surface perpendicular to the image plane.
            if cross_check(g) > 1.0
                o = offset_to_velocity(g, coord, dt)
                r.off2vx_dx[k] = o[1]
                r.off2vx_dy[k] = o[2]
                r.off2vy_dx[k] = o[3]
                r.off2vy_dy[k] = o[4]
            end

            sfx, sfy = scale_factors(g, coord)
            r.scale_x[k] = sfx
            r.scale_y[k] = sfy

            if has_sr
                # Scaled and clamped before the nodata test, which is why a sentinel at the
                # positive end survives as an apparently valid range. See REFERENCE.md.
                s1x = scaled_searchrange(params.scaling, inp.srx[k], sr_scale)
                s1y = scaled_searchrange(params.scaling, inp.sry[k], sr_scale)
                if ismissingval(nd, s1x) || s1x == 0
                    r.search_x[k] = Int32(0)
                    r.search_y[k] = Int32(0)
                else
                    sr1 = close_slope_parallel(s1x, s1y, normal)
                    sr2 = close_slope_parallel(-s1x, s1y, normal)
                    sx, sy = search_pixels(sr1, sr2, g, coord, dt)
                    r.search_x[k] = cround32(sx)
                    r.search_y[k] = cround32(sy)
                end
            end
        end

        if has_csmin
            cx = Float64(inp.csminx[k])
            if ismissingval(nd, cx)
                r.chip_min_x[k] = out
                r.chip_min_y[k] = out
            else
                # Truncated, not rounded — unlike every other integer output here.
                px, py = chip_pixels(cx, Float64(inp.csminy[k]), params.chip_size_0, pix_x, pix_y)
                r.chip_min_x[k] = ctrunc32(px)
                r.chip_min_y[k] = ctrunc32(py)
            end
        end

        if has_csmax
            cx = Float64(inp.csmaxx[k])
            if ismissingval(nd, cx)
                r.chip_max_x[k] = out
                r.chip_max_y[k] = out
            else
                px, py = chip_pixels(cx, Float64(inp.csmaxy[k]), params.chip_size_0, pix_x, pix_y)
                r.chip_max_x[k] = ctrunc32(px)
                r.chip_max_y[k] = ctrunc32(py)
            end
        end

        if has_ssm
            m = Float64(inp.ssm[k])
            r.stable_surface[k] = ismissingval(nd, m) ? out : ctrunc32(m)
        end
    end

    # Bands the inputs do not support stay at their sentinel; the reference writes no file for them.
    has_slope || (fill!(r.scale_x, fout); fill!(r.scale_y, fout))
    return r
end
