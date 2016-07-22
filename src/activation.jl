export activation!, ∇activation!
export
    CUDNN_ACTIVATION_SIGMOID, CUDNN_ACTIVATION_RELU,
    CUDNN_ACTIVATION_TANH, CUDNN_ACTIVATION_CLIPPED_RELU

#const activation_dict = Dict(:sigmoid=>CUDNN_ACTIVATION_SIGMOID)

function activation_desc(mode, relu_nanopt, relu_ceiling)
    p = Ptr{Void}[0]
    cudnnCreateActivationDescriptor(p)
    cudnnSetActivationDescriptor(p[1], mode, relu_nanopt, relu_ceiling)
    p[1]
end

"""
    activation!(mode, x::CuArray, y::CuArray)

reluNanOpt: whether propagates NaN or not
reluCeiling: floating point number to specify the clipping threashod
when the activation mode is set to CUDNN_ACTIVATION_CLIPPED_RELU.
"""
function activation!{T}(mode, x::CuArray{T}, y::CuArray{T};
    relu_nanopt=CUDNN_NOT_PROPAGATE_NAN, relu_ceiling=1.0, alpha=1.0, beta=0.0)

    h = gethandle(device(x))
    adesc = activation_desc(mode, relu_nanopt, relu_ceiling)
    xdesc = tensor_desc(x)
    ydesc = tensor_desc(y)
    cudnnActivationForward(h, adesc, T[alpha], xdesc, x, T[beta], ydesc, y)

    cudnnDestroyActivationDescriptor(adesc)
    cudnnDestroyTensorDescriptor(xdesc)
    cudnnDestroyTensorDescriptor(ydesc)
    y
end

function ∇activation!{T}(mode, y::CuArray{T}, dy::CuArray{T}, x::CuArray{T}, dx::CuArray{T};
    relu_nanopt=CUDNN_NOT_PROPAGATE_NAN, relu_ceiling=1.0, alpha=1.0, beta=0.0)

    h = gethandle(device(x))
    adesc = activation_desc(mode, relu_nanopt, relu_ceiling)
    ydesc = tensor_desc(y)
    dydesc = tensor_desc(y)
    xdesc = tensor_desc(x)
    dxdesc = tensor_desc(dx)
    cudnnActivationBackward(h, adesc, T[alpha], ydesc, y, dydesc, dy, xdesc, x,
    T[beta], dxdesc, dx)

    cudnnDestroyActivationDescriptor(adesc)
    cudnnDestroyTensorDescriptor(ydesc)
    cudnnDestroyTensorDescriptor(dydesc)
    cudnnDestroyTensorDescriptor(xdesc)
    cudnnDestroyTensorDescriptor(dxdesc)
    dx
end
