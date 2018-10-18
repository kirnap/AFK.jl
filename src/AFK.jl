module AFK

include("Knetapi.jl");   export Parameter
include("core.jl")
include("primitive.jl"); export Multiply, Embed, BatchMul, Linear, Dense
include("compound.jl");  export MLP

end # module
