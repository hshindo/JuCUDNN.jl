workspace()
using JuCUDA
using JuCUDNN

ndevices()

x = curand(Float32,10,5,4,3)
Array(x)
y = activation!(CUDNN_ACTIVATION_SIGMOID, x, similar(x))
Array(y)
