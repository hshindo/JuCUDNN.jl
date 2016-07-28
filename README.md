# JuCUDNN.jl
NVIDIA [cuDNN](https://github.com/hshindo/Merlin.jl) wrapper for Julia.

* [CUDNN.jl](https://github.com/JuliaGPU/CUDNN.jl) is based on `CudaArray` in [Cudart.jl](https://github.com/JuliaGPU/CUDArt.jl)
* JuCuDNN.jl is based on `CuArray` in [JuCUDA.jl].

## Installation
```julia
julia> Pkg.clone("https://github.com/hshindo/JuCUDA.jl.git")
julia> Pkg.clone("https://github.com/hshindo/JuCUDNN.jl.git")
julia> Pkg.update()
```

## Types
```julia
```

## Activation
Supported modes: `relu`, `clipped_relu`, `sigmoid`, `tanh`
```julia
x = curand(Float32,10,5,4,3)
Array(x)
y = activation!(CUDNN_ACTIVATION_SIGMOID, x, similar(x))
Array(y)
```

## Convolution
```julia
x = curand(Float32,5,4,3,2)
w = curand(Float32,2,2,3,4)
y = convolution(x, w, (0,0), (1,1))
Array(y)

x = curand(Float32,5,4,3,2)
w = curand(Float32,2,2,3,1)
y = convolution(x, w, (0,0), (1,1))
dy = y
dx = zeros(x)
∇convolution_data!(x,w,dy,(0,0),(1,1),dx)
Array(dx)
dw = zeros(w)
∇convolution_filter!(x,dy,(0,0),(1,1),dw)
Array(dw)
db = similar(dy,1,1,1,1)
∇convolution_bias!(x,dy,db)
Array(db)
```

## Dropout
```julia
x = curand(Float32,5,4,3,2)
Array(x)
y, s, ssize, rspace, rsize = dropout(x, 0.5)
Array(y)
dy = y
dx = zeros(x)
∇dropout!(x, dy, 0.5, s, ssize, rspace, rsize, dx)
Array(dx)
```

## Pooling
```julia
x = curand(Float32,5,5,1,1)
y = pooling(x, (3,3), (1,1), (2,2), CUDNN_POOLING_MAX)
dy = y
dx = zeros(x)
∇pooling!(y, dy, x, (3,3), (1,1), (2,2), CUDNN_POOLING_MAX,dx)
Array(dx)
```

## Softmax
```julia
x = curand(Float32,5,4,3,2)
y = softmax(x,CUDNN_SOFTMAX_FAST)
dy = y
dx = zeros(x)
∇softmax!(x, CUDNN_SOFTMAX_FAST, y, dy, dx)
Array(dx)
```

## LRN
```julia
x = curand(Float32,5,4,3,2)
y = lrn(x)
dy = y
dx = zeros(x)
∇lrn!(x, y, dy, dx)
Array(dx)
```

## BatchNorm
```julia
x = curand(Float32,5,4,3,2)
factor = 0.9
epsilon = 0.001
scale = CuArray(fill(Float32(1.0),(1,1,3,1)))
bias = CuArray(fill(Float32(0.0),(1,1,3,1)))
mean = similar(scale)
invvar = similar(scale)
y, rmean, rinvvar, smean, sinvvar = batchnorm_training(
    CUDNN_BATCHNORM_SPATIAL, x, scale, bias, factor, mean, invvar, epsilon)
Array(y)
z = batchnorm_inference(
    CUDNN_BATCHNORM_SPATIAL, x, scale, bias, rmean, rinvvar, epsilon)
Array(z)
dy = y
dx = zeros(x)
rscale = similar(scale)
rbias = similar(scale)
∇batchnorm!(
    CUDNN_BATCHNORM_SPATIAL, x, dy, scale, rscale, rbias, epsilon, smean, sinvvar, dx)
Array(dx)
```

## Others
```julia
```
