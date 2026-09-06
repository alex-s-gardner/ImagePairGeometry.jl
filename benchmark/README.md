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

`run_items23.jl`, `run_bitwise.jl` and `run_ceiling.jl` record measurements a source comment cites:
which orbit-interpolation variants preserve the tested 1-ULP position agreement with isce3, and which
loop-invariant hoists are worth taking once inlined. A change that revisits either should re-run them
rather than trust the numbers in the comment.
