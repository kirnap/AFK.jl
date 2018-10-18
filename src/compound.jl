# Compound Layers with combination of primitive, recurrent and convolutional layers
"""
MLP is a bunch of stacked Dense Layers, i.e. MLP(;input::Int, hiddens::Array{Int}, output::Int)
"""
struct MLP
    layers::Tuple{Vararg{Layer}}
end

function MLP(;hiddens::Array, input::Int, output::Int, o...)
    ls = Layer[]
    for h in hiddens
        l = Dense(;input=input, output=h, activation=relu)
        input = h
        push!(ls, l)
    end
    push!(ls, Dense(;input=input, output=output)) # no need non-linear activity for final layer
    return MLP(Tuple(ls))
end

function (m::MLP)(x)
    for l in m.layers
        x = l(x)
    end
    return x
end
