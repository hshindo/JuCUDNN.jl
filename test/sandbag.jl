workspace()
using JuCUDA
using JuCUDNN

ndevices()

x = curand(Float32,10,5,4,3)
Array(x)
y = activation!(CUDNN_ACTIVATION_SIGMOID, x, similar(x))
Array(y)

x = curand(Float32,5,5,1,1)
y = pooling(x,(3,3),(1,1),(2,2),CUDNN_POOLING_MAX)
dy=y
dx=zeros(x)
∇pooling!(y,dy,x,(3,3),(1,1),(2,2),CUDNN_POOLING_MAX,dx)
Array(dx)
