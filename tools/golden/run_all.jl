# The whole sweep, in the order the layers depend on each other.
#
#     julia --project=tools/golden -t 8 tools/golden/run_all.jl
#
# Layers 0 and 2 come before 1 and 3 deliberately. The self-gate establishes the comparator reports
# differences at all, and the handoff needs no parameter rasters — so both run in seconds and a failure
# in either says the later layers' inputs cannot be trusted. Layer 1 is the long pole, and Layer 3 needs
# its geometry.
#
# Each layer's own script runs standalone and takes a case name; this exists to run them in order and
# print one table. A red layer does not stop the sweep: they test different things, and knowing which
# ones are red at once is more useful than the first one.

using Printf

const HERE = @__DIR__

"""
    layer(name, script, args) -> Bool

Run one layer in a fresh process and report whether it passed.

A fresh process per layer rather than `include`, because the layer scripts share include chains and
several define the same helper names — `main_optical` and `main_radar` both pull in `params.jl` and
`compare.jl`. Loading them into one session would have the last definition win silently.
"""
function layer(name::AbstractString, script::AbstractString, args::Vector{String} = String[])
    @printf("\n%s\n=== %s\n%s\n", "="^78, name, "="^78)
    path = joinpath(HERE, script)
    threads = "-t" * string(max(1, Threads.nthreads()))
    cmd = `$(Base.julia_cmd()) --project=$HERE $threads $path $args`
    ok = success(pipeline(cmd; stdout = stdout, stderr = stderr))
    @printf("--- %s: %s\n", name, ok ? "pass" : "FAIL")
    return ok
end

results = Pair{String,Bool}[]
push!(results, "0  self-gate" => layer("Layer 0 — the harness gates on itself", "selftest.jl"))
push!(results, "2  handoff" => layer("Layer 2 — geogrid output to the correlator's inputs", "handoff.jl"))
push!(results, "1  optical" => layer("Layer 1 — geogrid, projected path", "geogrid_optical.jl"))
push!(results, "1  radar" => layer("Layer 1 — geogrid, radar path", "geogrid_radar.jl"))
push!(results, "3  pointset" => layer("Layer 3 — AutoRIFT.pointset against the captures", "pointset.jl"))

@printf("\n%s\n%-28s %s\n%s\n", "="^78, "layer", "result", "="^78)
for (name, ok) in results
    @printf("%-28s %s\n", name, ok ? "pass" : "FAIL")
end
allok = all(last, results)
println()
println(allok ? "every layer agrees within its gates" : "at least one layer is outside its gates")
exit(allok ? 0 : 1)
