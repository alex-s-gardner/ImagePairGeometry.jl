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

`run_chebyshev.jl` is the gate on `chebyshev_orbit`: the interpolant's accuracy over the whole orbit
domain, the cost on a window, and every output band against the default Hermite path. Its accuracy
figure depends on the orbit by six orders of magnitude, so with `IPG_REALDATA_DIR` set it measures a
real orbit alongside the analytic one. The band
comparison is the part that decides whether the option is safe to offer, so a change touching either
interpolant should re-run it.

`run_items23.jl`, `run_bitwise.jl`, `run_ceiling.jl` and `run_sepsum.jl` record measurements a source
comment or commit message cites: which orbit-interpolation variants preserve the tested 1-ULP position
agreement with isce3, which loop-invariant hoists are worth taking once inlined, and the bitwise gate
behind storing the Hermite separation sums on the `Orbit`. A change that revisits any of them should
re-run it rather than trust the number in prose.

`run_realdata.jl` is the exception to everything above: it measures a real NISAR acquisition over the
ITS_LIVE 120 m grid rather than synthetic inputs, so it needs data on disk and does not run standalone.
`test/reference/gen_radar_realdata.py` produces the reference outputs it compares against and the
`run.json` holding the reference's own kernel time, so the comparison needs no second Python run:

```
IPG_REALDATA_DIR=<dir> julia --project=benchmark -t 8 benchmark/run_realdata.jl
```

It records band agreement beside the timings and writes both to `julia_run.json` in that directory.

It reports two timings, because the reference's timer covers reading the twelve inputs and writing the
nine outputs as well as the kernel. The matched number is the comparable one; the kernel-only number is
what a change to the kernel moves. Timing the kernel alone against the reference's total overstates the
difference by about 1.6x.

None of the others depend on a projection library. `fast_transform` is the only transform the package
ships, so it is what they measure; the reference fixtures were generated through PROJ and `test/`
asserts against it, but nothing here does.
