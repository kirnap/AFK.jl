# Export Knet functionalities
using Knet

Parameter = Knet.param

export @diff, grad, value, update!, params  # gradient
export relu, sigm, invx, elu, selu          # activation
export KnetArray                            # datatype
