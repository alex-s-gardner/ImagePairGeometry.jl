# The three-vector primitives, checked for exact values and for the properties that motivate
# them existing at all rather than calling LinearAlgebra.

using ImagePairGeometry: dot3, cross3, norm3, unitvec3
using StaticArrays: SVector
using LinearAlgebra: dot, cross, norm, normalize
using Test

v3(a, b, c) = SVector{3,Float64}(a, b, c)

@testset "exact values" begin
    @test dot3(v3(1, 2, 3), v3(4, 5, 6)) === 32.0
    @test cross3(v3(1, 0, 0), v3(0, 1, 0)) === v3(0, 0, 1)
    @test cross3(v3(0, 1, 0), v3(1, 0, 0)) === v3(0, 0, -1)
    @test norm3(v3(3, 4, 0)) === 5.0
    @test unitvec3(v3(3, 4, 0)) === v3(0.6, 0.8, 0.0)
    @test unitvec3(v3(-30, 0, 0)) === v3(-1.0, 0.0, 0.0)
end

@testset "agrees with LinearAlgebra where both are exact" begin
    for (a, b) in ((v3(1, 2, 3), v3(4, 5, 6)), (v3(-1.5, 0.25, 8), v3(2, -4, 0.5)))
        @test dot3(a, b) == dot(a, b)
        @test cross3(a, b) == cross(a, b)
        @test norm3(a) == norm(a)
    end
end

@testset "unitvec3 divides, it does not scale by the reciprocal" begin
    # The reason this is not `normalize`. `x / n` and `x * (1 / n)` differ in the last bit for
    # most inputs, and the reference divides (`geogridOptical.cpp:1026-1032`).
    differs = 0
    for i in 1:2000
        v = v3(i * 1.7 - 900, i * -0.31 + 12, i * 0.077)
        u = unitvec3(v)
        n = norm3(v)
        @test u === v3(v[1] / n, v[2] / n, v[3] / n)
        recip = v * (1 / n)
        differs += any(reinterpret(UInt64, u[k]) != reinterpret(UInt64, recip[k]) for k in 1:3)
    end
    # Not a property we need, just evidence the distinction is real rather than theoretical.
    @test differs > 0
end

@testset "identity-transform vectors are exact" begin
    # Under an identity CRS transform the shifted difference is exactly the spacing, so the axis
    # unit vectors and scale factors come out exact. Relied on by the fast path.
    for spacing in (10.0, 15.0, 30.0, 60.0, 120.0, -30.0, -120.0)
        d = v3(spacing, 0, 0)
        @test norm3(d) === abs(spacing)
        @test unitvec3(d) === v3(sign(spacing), 0.0, 0.0)
        @test norm3(d) / abs(spacing) === 1.0
    end
end

@testset "zero vector yields NaN, as the reference does" begin
    # Reachable: the surface normal is all-zeros when no slope raster is given
    # (`geogridOptical.cpp:761-764`). The reference does not guard it and neither do we.
    u = unitvec3(v3(0, 0, 0))
    @test all(isnan, u)
    @test norm3(v3(0, 0, 0)) === 0.0
end

@testset "type stable and non-allocating" begin
    a, b = v3(1, 2, 3), v3(4, 5, 6)
    @test @inferred(dot3(a, b)) === 32.0
    @test @inferred(cross3(a, b)) isa SVector{3,Float64}
    @test @inferred(norm3(a)) isa Float64
    @test @inferred(unitvec3(a)) isa SVector{3,Float64}
    @test @allocated(dot3(a, b)) == 0
    @test @allocated(cross3(a, b)) == 0
    @test @allocated(unitvec3(a)) == 0
end
