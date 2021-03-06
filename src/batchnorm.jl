export
    batchnorm_inference, batchnorm_inference!,
    batchnorm_training, batchnorm_training!,
    ∇batchnorm!
export
    CUDNN_BATCHNORM_PER_ACTIVATION, CUDNN_BATCHNORM_SPATIAL # mode

function batchnorm_inference(mode, x::CuArray, scale::CuArray, bias::CuArray,
    estmean::CuArray, estvar::CuArray, epsilon; alpha=1.0, beta=0.0)
    
    batchnorm_inference!(mode, x, similar(x), scale, bias, estmean, estvar, epsilon,
        alpha=alpha, beta=beta)
end

function batchnorm_training(mode, x::CuArray, scale::CuArray, bias::CuArray,
    averagefactor, epsilon; alpha=1.0, beta=0.0)
    
    y = similar(x)
    runmean = zeros(scale)
    runvar = zeros(scale)
    savemean = similar(scale)
    saveinvvar = similar(scale)

    batchnorm_training!(mode, x, y, scale, bias, averagefactor, runmean, runvar,
        epsilon, savemean, saveinvvar, alpha=alpha, beta=beta)

    y, runmean, runvar, savemean, saveinvvar
end

function batchnorm_inference!{T}(mode, x::CuArray{T}, y::CuArray{T},
    scale::CuArray{T}, bias::CuArray{T}, estmean::CuArray{T}, estvar::CuArray{T},
    epsilon; alpha=1.0, beta=0.0)

    h = gethandle(device(x))
    xdesc = tensor_desc(x)
    ydesc = tensor_desc(y)
    sbmvdesc = tensor_desc(scale)

    cudnnBatchNormalizationForwardInference(h, mode, T[alpha], T[beta], xdesc, x,
        ydesc, y, sbmvdesc, scale, bias, estmean, estvar, Cdouble(epsilon))

    cudnnDestroyTensorDescriptor(xdesc)
    cudnnDestroyTensorDescriptor(ydesc)
    cudnnDestroyTensorDescriptor(sbmvdesc)
    y
end

function batchnorm_training!{T}(mode, x::CuArray{T}, y::CuArray{T},
    scale::CuArray{T}, bias::CuArray{T}, averagefactor, runmean::CuArray{T},
    runvar::CuArray{T}, epsilon, savemean::CuArray{T}, saveinvvar::CuArray{T};
    alpha=1.0, beta=0.0)

    h = gethandle(device(x))
    xdesc = tensor_desc(x)
    ydesc = tensor_desc(y)
    sbmvdesc = tensor_desc(scale)

    cudnnBatchNormalizationForwardTraining(h, mode, T[alpha], T[beta], xdesc, x,
        ydesc, y, sbmvdesc, scale, bias, Cdouble(averagefactor), runmean, runvar,
	Cdouble(epsilon), savemean, saveinvvar)

    cudnnDestroyTensorDescriptor(xdesc)
    cudnnDestroyTensorDescriptor(ydesc)
    cudnnDestroyTensorDescriptor(sbmvdesc)
end

function ∇batchnorm!{T}(mode, x::CuArray{T}, dy::CuArray{T},
    scale::CuArray{T}, resultscale::CuArray{T}, resultbias::CuArray{T},
    epsilon, savemean::CuArray{T}, saveinvvar::CuArray{T}, dx::CuArray{T};
    alphadata=1.0, betadata=0.0, alphaparam=1.0, betaparam=0.0)

    h = gethandle(device(x))
    xdesc = tensor_desc(x)
    dydesc = tensor_desc(dy)
    dxdesc = tensor_desc(dx)
    sbdesc = tensor_desc(scale)

    cudnnBatchNormalizationBackward(h, mode, T[alphadata], T[betadata], T[alphaparam],
        T[betaparam], xdesc, x, dydesc, dy, dxdesc, dx, sbdesc, scale, resultscale,
	resultbias, Cdouble(epsilon), savemean, saveinvvar)

    cudnnDestroyTensorDescriptor(xdesc)
    cudnnDestroyTensorDescriptor(dydesc)
    cudnnDestroyTensorDescriptor(dxdesc)
    cudnnDestroyTensorDescriptor(sbdesc)
end
