using Test, AFK

_eltype = Float32
_test_atype = (AFK.gpu() >= 0 ? AFK.KnetArray{_eltype} : Array{_eltype})

@time include("primitive.jl")
