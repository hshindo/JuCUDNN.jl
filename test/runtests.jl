using JuCUDA
using JuCUDNN
if VERSION >= v"0.5-"
    using Base.Test
else
    using BaseTestNext
    const Test = BaseTestNext
end

@testset "activation" for i = 1:5
    x = CuArray(Float32, 10, 5, 3, 2)
    desc = ActivationDesc(CUDNN_ACTIVATION_SIGMOID)
    y = similar(x)
    CUDNN.activation!(desc, x, y)
end

@testset "convolution" for i = 1:5
    x = CuArray(Float32, 10, 5, 3, 2)
    desc = ConvolutionDesc()
    y = similar(x)
    CUDNN.activation!(desc, x, y)
end
