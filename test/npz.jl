# A minimal reader for the `.npz` fixture archives.
#
# The reference fixtures store their arrays as raw little-endian Float64 in a compressed npz —
# exact to the bit, and around twenty times smaller than the same values written as JSON text. An
# npz is a zip of `.npy` members, and a `.npy` is a short ASCII header followed by the raw buffer.
# Reading the one dtype the fixtures use takes less code than taking on a dependency, and keeps the
# test environment to Test, Aqua, JSON3, Proj and StaticArrays.
#
# Deliberately narrow: little-endian Float64, C order, zip entries either stored or deflated.
# Anything else throws rather than being guessed at.

using CodecZlib: DeflateDecompressor, transcode
using Test

"""
    parse_npy(bytes) -> Array{Float64}

The array in a `.npy` buffer.

Accepts only little-endian Float64 in C order, which is what `gen_geogrid.py` writes. The header is
a Python dict literal; only `descr`, `fortran_order` and `shape` are read from it.
"""
function parse_npy(bytes::Vector{UInt8})
    length(bytes) > 10 || throw(ArgumentError("npy buffer too short: $(length(bytes)) bytes"))
    bytes[1:6] == UInt8[0x93, 'N', 'U', 'M', 'P', 'Y'] ||
        throw(ArgumentError("not an npy buffer: bad magic"))
    major = bytes[7]
    hlen, hstart = if major == 1
        Int(bytes[9]) | (Int(bytes[10]) << 8), 11
    else
        (Int(bytes[9]) | (Int(bytes[10]) << 8) | (Int(bytes[11]) << 16) |
         (Int(bytes[12]) << 24)), 13
    end
    header = String(bytes[hstart:hstart + hlen - 1])

    occursin("'<f8'", header) || throw(ArgumentError(
        "npz fixtures must be little-endian Float64, got header: $header"))
    occursin("'fortran_order': False", header) || throw(ArgumentError(
        "npz fixtures must be in C order, got header: $header"))

    m = match(r"'shape':\s*\(([^)]*)\)", header)
    m === nothing && throw(ArgumentError("no shape in npy header: $header"))
    dims = [parse(Int, strip(s)) for s in split(m.captures[1], ',') if !isempty(strip(s))]

    data = reinterpret(Float64, bytes[hstart + hlen:end])
    n = prod(dims; init = 1)
    length(data) == n || throw(ArgumentError(
        "npy payload is $(length(data)) elements, header says $n"))
    # C order: the last dimension varies fastest, so read into reversed dims. Callers apply
    # `permutedims` to get the orientation this package uses.
    return Array(reshape(data, reverse(dims)...))
end

"""
    load_npz(path) -> Dict{String,Array{Float64}}

Every member of an `.npz` archive, keyed by name with the `.npy` suffix removed.

Reads the zip central directory rather than scanning for local headers, so a member whose local
header omits its size (as a streamed writer may) still reads correctly.
"""
function load_npz(path::AbstractString)
    bytes = read(path)
    n = length(bytes)

    u16(i) = UInt16(bytes[i]) | (UInt16(bytes[i + 1]) << 8)
    u32(i) = UInt32(bytes[i]) | (UInt32(bytes[i + 1]) << 8) |
             (UInt32(bytes[i + 2]) << 16) | (UInt32(bytes[i + 3]) << 24)

    # End-of-central-directory record, searched from the back past any comment.
    eocd = 0
    for i in (n - 21):-1:max(1, n - 65557)
        if u32(i) == 0x06054b50
            eocd = i
            break
        end
    end
    eocd == 0 && throw(ArgumentError("$path is not a zip archive: no end-of-central-directory"))

    nentries = Int(u16(eocd + 10))
    cd = Int(u32(eocd + 16)) + 1

    out = Dict{String,Array{Float64}}()
    p = cd
    for _ in 1:nentries
        u32(p) == 0x02014b50 || throw(ArgumentError("bad central directory entry in $path"))
        method = Int(u16(p + 10))
        csize = Int(u32(p + 20))
        namelen = Int(u16(p + 28))
        extralen = Int(u16(p + 30))
        commentlen = Int(u16(p + 32))
        lho = Int(u32(p + 42)) + 1
        name = String(bytes[p + 46:p + 46 + namelen - 1])

        # Local header: fixed 30 bytes, then the name and extra field.
        u32(lho) == 0x04034b50 || throw(ArgumentError("bad local header for $name in $path"))
        lnamelen = Int(u16(lho + 26))
        lextralen = Int(u16(lho + 28))
        dstart = lho + 30 + lnamelen + lextralen
        raw = bytes[dstart:dstart + csize - 1]

        payload = if method == 0
            raw
        elseif method == 8
            transcode(DeflateDecompressor, raw)
        else
            throw(ArgumentError("unsupported zip compression method $method for $name in $path"))
        end

        out[replace(name, r"\.npy$" => "")] = parse_npy(Vector{UInt8}(payload))
        p += 46 + namelen + extralen + commentlen
    end
    return out
end
