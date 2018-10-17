using Test, AFK

_eltype = Float32
_test_atype = (AFK.gpu() >= 0 ? AFK.KnetArray{_eltype} : Array{_eltype})

_INPUTdim = 10; _OUTPUTdim = 2;

@time include("primitive.jl")
