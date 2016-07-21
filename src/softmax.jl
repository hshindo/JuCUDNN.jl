export softmax, softmax!, ∇softmax!
export CUDNN_SOFTMAX_FAST, CUDNN_SOFTMAX_ACCURATE, CUDNN_SOFTMAX_LOG # algorithm
export CUDNN_SOFTMAX_MODE_INSTANCE, CUDNN_SOFTMAX_MODE_CHANNEL # mode

function softmax!{T}(x::CuArray{T}, mode, y::CuArray{T};
  algo=CUDNN_SOFTMAX_ACCURATE, alpha=1.0, beta=0.0)

  h = gethandle(device(x))
  xdesc = tensor_desc(x)
  ydesc = tensor_desc(y)
  cudnnSoftmaxForward(h, algo, mode, T[alpha], xdesc, x, T[beta], ydesc, y)

  cudnnDestroyTensorDescriptor(xdesc)
  cudnnDestroyTensorDescriptor(ydesc)
  y
end

function softmax(x::CuArray, mode; algo=CUDNN_SOFTMAX_ACCURATE, alpha=1.0, beta=0.0)
  y = similar(x)
  softmax!(x, mode, y, algo=algo, alpha=alpha, beta=beta)
end

function ∇softmax!{T}(x::CuArray{T}, mode, y::CuArray{T}, dy::CuArray{T}, dx::CuArray{T};
  algo=CUDNN_SOFTMAX_ACCURATE, alpha=1.0, beta=0.0)

  h = gethandle(device(x))
  ydesc = tensor_desc(y)
  dydesc = tensor_desc(dy)
  dxdesc = tensor_desc(dx)
  cudnnSoftmaxBackward(h, algo, mode, T[alpha], ydesc, y, dydesc, dy,
    T[beta], dxdesc, dx)

  cudnnDestroyTensorDescriptor(ydesc)
  cudnnDestroyTensorDescriptor(dydesc)
  cudnnDestroyTensorDescriptor(dxdesc)
  dx
end
