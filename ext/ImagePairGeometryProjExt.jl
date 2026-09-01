module ImagePairGeometryProjExt

# PROJ transforms, built so a threaded run does not corrupt them.
#
# `Proj.Transformation` wraps a `PJ*` created on a PROJ context. A context owns an SQLite connection
# to `proj.db`, and PROJ documents a context as usable from one thread at a time. Constructing two
# transformations concurrently on the shared global context is enough to break it — the failure is
# not a wrong number but
#
#   PROJError: proj_create_operations: SQLite error [ code = 21, msg = no more rows available ]
#
# from inside `proj_create_crs_to_crs`, i.e. a corrupted statement handle.
#
# So a threaded run needs one context per task, and each context needs its search paths set: `Proj`
# points only the *global* context at its bundled `proj.db`, so a self-created context cannot find
# the database and every transformation on it fails to build.
#
# Contexts are deliberately never destroyed. A `Transformation`'s finalizer calls `proj_destroy` on
# its `PJ*`, and that must not run after its context is freed; letting the process own the contexts
# avoids ordering a teardown against the garbage collector for a resource whose per-task cost is
# a few hundred kilobytes.

using ImagePairGeometry
using ImagePairGeometry: TransformPair, AbstractTransformFactory
using Proj

"""
    ProjTransformFactory(grid_crs, image_crs)

A factory building a [`TransformPair`](@ref) between two CRSs, one PROJ context per call.

Pass it as `pairgeometry_blocked`'s `transform`: it is called once per task, so each task gets a
transformation on a context it alone uses. Calling it is not cheap — about 1.7 ms, dominated by a
cold `proj.db` cache — which is why it is per task rather than per block.

`grid_crs` and `image_crs` are anything `Proj.Transformation` accepts: `"EPSG:3413"`, an integer
EPSG code, a WKT string, or a `GeoFormatTypes` object.

The pair's forward direction is grid to image, matching the kernel; `footprint_bounds` takes the
inverse from the pair itself.

```julia
factory = ProjTransformFactory(3413, 32624)
r = pairgeometry_blocked(grid, pair, source; transform = factory, ntasks = 8)
```

For a serial run `factory()` gives a pair directly, and a same-CRS run should use
`IdentityTransform()` instead — it is exact and skips PROJ entirely.
"""
struct ProjTransformFactory{G,I} <: AbstractTransformFactory
    grid_crs::G
    image_crs::I
end

_crs_string(x::Integer) = "EPSG:$x"
_crs_string(x) = x

function (f::ProjTransformFactory)()
    ctx = Proj.proj_context_create()
    # Without this the context cannot find the bundled `proj.db`: `Proj.__init__` sets the search
    # path on the global context only.
    Proj.proj_context_set_search_paths(1, [Proj.PROJ_DATA[]], ctx)
    # Grids fetched over the network would make results depend on what happened to be cached.
    Proj.proj_context_set_enable_network(false, ctx)
    g, i = _crs_string(f.grid_crs), _crs_string(f.image_crs)
    return TransformPair(Proj.Transformation(g, i; ctx, always_xy = false),
                        Proj.Transformation(i, g; ctx, always_xy = false))
end

"""
    ImagePairGeometry.proj_transform(grid_crs, image_crs) -> TransformPair

A [`TransformPair`](@ref) between two CRSs on a fresh PROJ context.

For a serial run. A threaded run wants [`ProjTransformFactory`](@ref) so each task builds its own,
since a context is not safe to use from two threads at once.

`always_xy = false` reproduces the reference, which builds its transforms with
`osr.CoordinateTransformation` on SRSs from `ImportFromEPSG` — authority axis order. For
projected-to-projected pairs the two agree, so this is fidelity to the reference rather than a
behavioral choice.
"""
ImagePairGeometry.proj_transform(grid_crs, image_crs) =
    ProjTransformFactory(grid_crs, image_crs)()

export ProjTransformFactory

end
