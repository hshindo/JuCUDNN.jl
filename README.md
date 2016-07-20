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

## Convolution
```julia
```

## Bias
```julia
```

## Pooling
```julia
x = curand(Float32,5,5,1,1)
y = pooling(x,(3,3),(1,1),(2,2),CUDNN_POOLING_MAX)
dy=y
dx=zeros(x)
∇pooling!(y,dy,x,(3,3),(1,1),(2,2),CUDNN_POOLING_MAX,dx)
Array(dx)
```

## Activation
```julia
```

## Others
```julia
```
