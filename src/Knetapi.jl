# Export Knet functionalities
using Knet

Parameter = Knet.param

export relu, sigm, invx, elu, selu # activation
export @diff, grad, value          # gradient
export KnetArray                   # datatype
