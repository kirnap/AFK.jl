module AFK

# Knet functionalities
using Knet
Parameter = Knet.param

include("core.jl")
include("primitive.jl"); export Multiply


end # module
