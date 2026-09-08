# Raster IO

Available when `Rasters`, `ArchGDAL`, `DimensionalData` and `DiskArrays` are loaded — all four, since the
extension needs each of them. The core depends on no IO stack, which is what keeps a caller who wants only
the kernel from acquiring GDAL with it, and what makes the kernel `juliac --trim`-compilable.

What lives here is everything that touches a file: building a [`MapGrid`](@ref) or an
[`ImageFootprint`](@ref) from a raster's geotransform, reading each block's inputs from disk so a grid too
large to materialize never is, sampling a DEM for the radar solve, and writing the nine output GeoTIFFs
with the band order, data types and nodata values the reference uses.

```@autodocs
Modules = [Base.get_extension(ImagePairGeometry, :ImagePairGeometryRastersExt)]
Order = [:type, :function]
```
