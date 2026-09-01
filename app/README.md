# Standalone compilation

`kernel_main.jl` is an entry point that exercises the projected-path kernel with no IO, no PROJ and
no Rasters, so it can be compiled to a standalone binary with `juliac --trim`. Building it is a
check on type stability that is stronger than any static analysis: `--trim=safe` refuses any call it
cannot resolve statically, so a success means the whole path from `coregister` through
`pairgeometry` is concretely inferred.

Requires Julia 1.12 or later, where `juliac` ships with Julia.

```bash
cd app
julia --project=.. \
  "$(dirname "$(which julia)")/../share/julia/juliac/juliac.jl" \
  --output-exe kernelapp --trim=safe --experimental kernel_main.jl
./kernelapp     # prints the valid-point count
```

Produces a ~1.4 MB executable. Only the core is trimmable: the extensions pull in GDAL and PROJ,
whose initialization is not statically resolvable, which is the reason IO lives in extensions rather
than in `src/`.

One thing this build has already caught: `allocate_geometry` used `ntuple` with a runtime length,
giving a `Tuple{Vararg{...}}` of unknown size whose splat into the constructor could not be resolved.
`Val(length(...))` fixed it. Nothing in the test suite or in JET's report flagged that — the trim
verifier is a distinct check, not a duplicate of them.
