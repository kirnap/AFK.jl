"""
    Multiply(input=inputDimension, output=outputDimension; winit=xavier, o...)

Matrix multiplication layer with a matrix.
    (m::Multiply)(x) = m.w * x

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
    Embed(input=inputDimension, output=outputDimension; winit=xavier, o...)

Lookup layer to encode sparse one-hot vector into dense continous vector.
    (m::Embed)(x) = m.w[:, x]

Input:  An integer array containing 1 locations of one-hot vectors

Output: Input-indexed columns of layer's matrix (m.w)

# Keywords
* `input=inputDimension`   : One-hot vector's dimension
* `output=outputDimension` : Dense vector's dimension
* `winit=xaiver`           : Distribution for weight initialization

"""
Embed = Multiply


"""
    BatchMul(;input=inputDimension, output=outputDimension, winit=xavier, o...)
Matrix tensor multiplication layer with a 2D matrix.

Input: N-dimensional tensor containing instances in the first dimension 

Output: Multiplication of all instances in the raws input. If `flatte=true` output is reduced to 2D matrix, otherwise input size is preserved except first dimension

# Keywords
* `input=inputDimension`   : Input tensor's  1st dimension
* `output=outputDimension` : Output tensor's 1st dimension
* `winit=xaiver`           : Distribution for weight initialization
* `flatten=true`           : Reduces the output tensor into 2D matrix 

"""
struct BatchMul <: Layer
    w # weight
end

function BatchMul(;input::Int, output::Int, winit=xavier, o...)
    w = Parameter(output, input; init=winit, o...)
    return BatchMul(w)
end

function (m::BatchMul)(x; flatten=false)
    xdims  = size(x)
    x_flat = reshape(x, xdims[1], prod(xdims[2:end]))
    yout = m.w * x_flat
    if flatten
        return yout
    end
    return reshape(yout, size(yout, 1), xdims[2:end]...)
end


"""
    Linear(;input=inputDimension, output=outputDimension, winit=xavier, binit=zeros, o...)
Linear layer applying usual linear transformation `w * x .+ b`.

Input:  inputDimension by BatchSize `x` matrix where Batchsize ∈ {1, 2,.... N}

Output: outputDimension by BatchSize Batchsize ∈ {1, 2,.... N}

# Keywords
* `input=inputDimension`   : Input tensor's  1st dimension
* `output=outputDimension` : Output tensor's 1st dimension
* `winit=xaiver`           : Distribution for weight initialization
* `binit=zeros`            : Distribution for bias initialization

"""
struct Linear
    w::Multiply
    b # bias
end

function Linear(;input::Int, output::Int, winit=xavier, binit=zeros, o...)
    w = Multiply(;input=input, output=output, o...)
    b = Parameter(output; init=binit, o...)
    return Linear(w, b)
end

(m::Linear)(x) = m.w(x) .+ m.b
