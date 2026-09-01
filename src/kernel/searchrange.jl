# Search-range scaling: the one place `dt` enters nonlinearly.
#
# A search range given in m/yr converts to pixels by the same linear factor as any other velocity,
# but before that conversion the reference inflates it for short-interval pairs
# (`geogridOptical.cpp:701-712`). The reasoning is that a pair days apart has a displacement
# dominated by noise and coregistration error rather than by ice motion, so the search must be
# wider than the reference velocity implies; by `dt_unity` days the inflation is gone.
#
# The scale falls linearly in `dt`, is floored at 1, and the result is then clamped to a fixed
# window. Two consequences that rule out factoring `dt` out of the search range:
#
#   - The clamp is applied per axis, so it changes the *direction* of the search vector, not just
#     its length. The slope closure downstream consumes that direction.
#   - Output is not monotone in `dt`: a 91-day pair can get a wider pixel search than a 182-day
#     one, because the inflation falls faster than the displacement grows.

"""
    SearchRangeScaling(; dt_unity = 182.0, max_scale = 5.0, lower = 0.0, upper = 20000.0)

Parameters of the interval-dependent search-range inflation.

# Fields
- `dt_unity`: interval in days at which the scale reaches 1, i.e. no inflation.
- `max_scale`: scale approached as the interval goes to zero.
- `lower`, `upper`: bounds the scaled range is clamped to, in the same units as the input
  (m/yr).

Defaults are the reference's (`GeogridOptical.py:358-362`: `srs_dt_unity = 182`,
`srs_max_scale = 5`, `srs_min_search = 0`, `srs_max_search = 20000`).
"""
struct SearchRangeScaling{T<:Real}
    dt_unity::T
    max_scale::T
    lower::T
    upper::T
end

function SearchRangeScaling(; dt_unity = 182.0, max_scale = 5.0, lower = 0.0, upper = 20000.0)
    T = promote_type(typeof(dt_unity), typeof(max_scale), typeof(lower), typeof(upper))
    return SearchRangeScaling{T}(T(dt_unity), T(max_scale), T(lower), T(upper))
end

"""
    searchrange_scale(p::SearchRangeScaling, dt) -> Float64

Inflation factor applied to a search range for a pair separated by `dt` seconds.

Reproduces `geogridOptical.cpp:701-704` including its division order:

```c
std::max(max_factor * ((dt_unity - 1) * max_factor + (max_factor - 1)
         - (max_factor - 1) * dt / 24.0 / 3600.0) / ((dt_unity - 1) * max_factor), 1.0)
```

Falls linearly from `max_scale` toward 1 as `dt` grows, floored at 1 — so a pair longer than
`dt_unity` days is not inflated, and never shrunk.

```jldoctest
julia> using ImagePairGeometry: SearchRangeScaling, searchrange_scale

julia> p = SearchRangeScaling();

julia> round(searchrange_scale(p, 1 * 86400.0), digits = 4)
5.0

julia> round(searchrange_scale(p, 91 * 86400.0), digits = 4)
3.011

julia> searchrange_scale(p, 182 * 86400.0)
1.0

julia> searchrange_scale(p, 365 * 86400.0)
1.0
```
"""
@inline function searchrange_scale(p::SearchRangeScaling, dt::Real)
    dt_unity = Float64(p.dt_unity)
    max_factor = Float64(p.max_scale)
    days = Float64(dt) / 24.0 / 3600.0
    scale = max_factor * ((dt_unity - 1) * max_factor + (max_factor - 1) -
                          (max_factor - 1) * days) / ((dt_unity - 1) * max_factor)
    return max(scale, 1.0)
end

"""
    scaled_searchrange(p::SearchRangeScaling, sr, scale) -> Float64

One axis of a search range, inflated by `scale` then clamped to `p.lower`/`p.upper`.

Matches `geogridOptical.cpp:701-711`, where the clamp is `min(max(sr, lower), upper)`.

The clamp is the reason the reference's nodata handling for search range is partly ineffective.
Applied *before* the nodata comparison, it maps a negative sentinel such as the DEM's `-32767` to
`lower` — normally `0.0`, which the downstream `== 0` test does catch — but maps the search-range
rasters' own `+32767` to `upper`, an apparently valid 20000 m/yr range. Reproduced; see
`REFERENCE.md`.
"""
@inline function scaled_searchrange(p::SearchRangeScaling, sr::Real, scale::Real)
    return min(max(Float64(sr) * Float64(scale), Float64(p.lower)), Float64(p.upper))
end
