export
    CUDNN_BATCHNORM_PER_ACTIVATION, CUDNN_BATCHNORM_SPATIAL # mode

function batchnorm_inference(mode, scale, bias, mean, variance;
    alpha=1.0, beta=0.0)

    h = gethandle(device(x))
    cudnnBatchNormalizationForwardInference(h, T[alpha], T[beta], xdesc, x, ydesc, y,
    bnScaleBiasMeanVarDesc, T[scale])
end
