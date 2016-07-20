export pooling, pooling!, ∇pooling!
export
  CUDNN_POOLING_MAX,
  CUDNN_POOLING_AVERAGE_COUNT_INCLUDE_PADDING,
  CUDNN_POOLING_AVERAGE_COUNT_EXCLUDE_PADDING
export
  CUDNN_NOT_PROPAGATE_NAN,
  CUDNN_PROPAGATE_NAN

function pooling_desc(mode, window, padding, stride, maxpoolingNanOpt)
  N = length(window)
  p = Ptr{Void}[0]
  cudnnCreatePoolingDescriptor(p)
  cwindow = Cint[window[i] for i=N:-1:1]
  cpadding = Cint[padding[i] for i=N:-1:1]
  cstride = Cint[stride[i] for i=N:-1:1]
  cudnnSetPoolingNdDescriptor(p[1], mode, maxpoolingNanOpt, N, cwindow, cpadding, cstride)
  p[1]
end

function pooling!{T}(x::CuArray{T}, window, padding, stride, mode, y::CuArray{T};
  maxpoolingNanOpt=CUDNN_NOT_PROPAGATE_NAN, alpha=1.0, beta=0.0)

  h = gethandle(device(x))
  xdesc = tensor_desc(x)
  poolingdesc = pooling_desc(mode, window, padding, stride, maxpoolingNanOpt)
  ydesc = tensor_desc(y)
  cudnnPoolingForward(h, poolingdesc, T[alpha], xdesc, x, T[beta], ydesc, y)

  cudnnDestroyTensorDescriptor(xdesc)
  cudnnDestroyPoolingDescriptor(poolingdesc)
  cudnnDestroyTensorDescriptor(ydesc)
  y
end

function pooling(x::CuArray, window, padding, stride, mode;
  maxpoolingNanOpt=CUDNN_NOT_PROPAGATE_NAN, alpha=1.0, beta=0.0)

  N = length(window)
  outdims = Array(Int, N)
  for i = 1:N
    outdims[i] = (size(x,i) + 2*padding[i] - window[i]) ÷ stride[i] + 1
  end
  y = similar(x, outdims..., size(x,N+1), size(x,N+2))
  pooling!(x, window, padding, stride, mode, y, maxpoolingNanOpt=maxpoolingNanOpt, alpha=alpha, beta=beta)
end

function ∇pooling!{T}(y::CuArray{T}, dy::CuArray{T}, x::CuArray{T},
  window, padding, stride, mode, dx::CuArray{T};
  maxpoolingNanOpt=CUDNN_NOT_PROPAGATE_NAN, alpha=1.0, beta=0.0)

  h = gethandle(device(x))
  poolingdesc = pooling_desc(mode, window, padding, stride, maxpoolingNanOpt)
  ydesc = tensor_desc(y)
  dydesc = tensor_desc(dy)
  xdesc = tensor_desc(x)
  dxdesc = tensor_desc(dx)
  cudnnPoolingBackward(h, poolingdesc, T[alpha], ydesc, y, dydesc, dy, xdesc, x,
    T[beta], dxdesc, dx)

  cudnnDestroyPoolingDescriptor(poolingdesc)
  cudnnDestroyTensorDescriptor(ydesc)
  cudnnDestroyTensorDescriptor(dydesc)
  cudnnDestroyTensorDescriptor(xdesc)
  cudnnDestroyTensorDescriptor(dxdesc)
end
