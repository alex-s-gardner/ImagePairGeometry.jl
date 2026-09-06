# Streaming the output

Blocking bounds the *inputs*: [`readblock`](@ref) hands back a view or a windowed read, so the eleven
input rasters are never materialized whole. The result is a separate matter. The 19 output bands are
108 bytes per grid point of the window whatever the block size — a 40000×40000 polar grid is around
300 GiB — so a run that large needs its output bounded too.

A `sink` does that. Each block is handed to it as the block completes and the buffer is reused, so
peak output memory is one block per task:

```julia
paths = pairgeometry_blocked(grid, pair, source; transform = tf, ntasks = 8,
                             sink = ImagePairGeometry.GeoTIFFOutputs(outdir))
```

`GeoTIFFOutputs` (defined when `Rasters` is loaded) writes the same files
[`write_geotiffs`](@ref) does, incrementally, and returns their paths.

[`BlockCallback`](@ref) streams to anything else:

```julia
n = Threads.Atomic{Int}(0)
pairgeometry_blocked(grid, pair, source; transform = tf, ntasks = 8,
                     sink = BlockCallback((block, g) -> Threads.atomic_add!(n, nvalid(g))))
n[]     # valid points over the whole window
```

The callback runs concurrently from every task, and its `g` views a buffer the task reuses for its
next block — so it must consume `g` rather than retain it.

The default sink is [`InMemoryOutputs`](@ref), which collects the whole window and returns the
[`PairGeometry`](@ref). Every band is bit-identical whichever sink is used, since blocking and
threading cannot change a point that depends only on its own inputs.

## Which bands a run produces

[`supported_bands`](@ref) reports the [`PairGeometry`](@ref) fields the given inputs can produce a
value for; every other field is uniformly nodata. A sink creating storage per band uses this to create
none for a band that would be written and never touched, which is what makes `GeoTIFFOutputs` write
the same *set* of files `write_geotiffs` does.

## Writing a sink

Implement all five functions. [`prepare_sink`](@ref) runs once before any block;
[`sink_taskstate`](@ref) once inside each task, and is where per-task state belongs;
[`blockdest`](@ref) and [`commitblock!`](@ref) run per block; [`finish_sink`](@ref) once after every
task has joined, and its return value is what the run returns.

```@autodocs
Modules = [ImagePairGeometry]
Order = [:type, :function]
Pages = ["sink.jl"]
```
