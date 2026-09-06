# Benchmarks

Not part of the package, and not run by CI. Each script is standalone:

```
julia --project=benchmark benchmark/<script>.jl
```

`*_perf.jl` are the standing cost breakdowns — per-point kernel cost by stage, the projected path's
cost share, blocked and threaded scaling, memory. `radar_setup.jl` builds the radar fixture's
acquisition on a grid of a chosen size, and the `run_*.jl` scripts measure against it: whole-kernel
cost and its breakdown, the statistical profile, storage scaling, and the accuracy-versus-iterations
sweeps behind `GEO2RDR_ITERATIONS`, `RANGE_DOPPLER_ITERATIONS` and `WarmStart`.

`run_items23.jl`, `run_bitwise.jl`, `run_ceiling.jl` and `run_sepsum.jl` record measurements a source
comment or commit message cites: which orbit-interpolation variants preserve the tested 1-ULP position
agreement with isce3, which loop-invariant hoists are worth taking once inlined, and the bitwise gate
behind storing the Hermite separation sums on the `Orbit`. A change that revisits any of them should
re-run it rather than trust the number in prose.

None of these depend on a projection library. `fast_transform` is the only transform the package
ships, so it is what they measure; the reference fixtures were generated through PROJ and `test/`
asserts against it, but nothing here does.
