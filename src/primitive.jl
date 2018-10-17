"""
    Multiply(input=inputDimension, output=outputDimension; winit=xavier, o...)

Matrix multiplication layer with a matrix initialized from `winit` keyword argument.

Input:  InputDimension by BatchSize `x` matrix where Batchsize ∈ {1, 2,.... N}

Output: Projection of a vector `x` from `inputDimension` to `outputDimension`

# Keywords
* `input=inputDimension`   : input matrix (vector) first dimension
* `output=outputDimension` : output matrix (vector) first dimension
* `winit=xaiver`           : distribution for weight initialization
"""
struct Multiply <: Layer
    w # weight
end

function Multiply(;input::Int, output::Int, winit=xavier, o...)
    w = Parameter(output, input; init=winit, o...)
    return Multiply(w)
end

(m::Multiply)(x) = m.w * x
(m::Multiply)(x::Array{T}) where T<:Integer = m.w[:, x]


"""

"""
Embed = Multiply
