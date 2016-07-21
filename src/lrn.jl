export lrn, lrn!, ∇lrn!

function lrn_desc(n, lrnalpha, lrnbeta, k)
  p = Ptr{Void}[0]
  cudnnCreateLRNDescriptor(p)
  cn = Cuint(n)
  clrnalpha = Cdouble(lrnalpha)
  clrnbeta = Cdouble(lrnbeta)
  ck = Cdouble(k)
  cudnnSetLRNDescriptor(p[1], cn, clrnalpha, clrnbeta, ck)
  p[1]
end

function lrn!{T}(x::CuArray{T}, y::CuArray{T};
  mode=CUDNN_LRN_CROSS_CHANNEL_DIM1, n=5, lrnalpha=1e-4, lrnbeta=0.75, k=2.0, alpha=1.0, beta=0.0)

  h = gethandle(device(x))
  lrndesc = lrn_desc(n, lrnalpha, lrnbeta, k)
  xdesc = tensor_desc(x)
  ydesc = tensor_desc(y)

  cudnnLRNCrossChannelForward(h, lrndesc, mode, T[alpha], xdesc, x, T[beta], ydesc, y)

  cudnnDestroyLRNDescriptor(lrndesc)
  cudnnDestroyTensorDescriptor(xdesc)
  cudnnDestroyTensorDescriptor(ydesc)
  y
end

function lrn(x::CuArray;
  mode=CUDNN_LRN_CROSS_CHANNEL_DIM1, n=5, lrnalpha=1e-4, lrnbeta=0.75, k=2.0, alpha=1.0, beta=0.0)

  y=similar(x)
  lrn!(x, y, mode=mode, n=n, lrnalpha=lrnalpha, lrnbeta=lrnbeta, k=k, alpha=alpha, beta=beta)
end

function ∇lrn!{T}(x::CuArray{T}, y::CuArray{T}, dy::CuArray{T}, dx::CuArray{T};
  mode=CUDNN_LRN_CROSS_CHANNEL_DIM1, n=5, lrnalpha=1e-4, lrnbeta=0.75, k=2.0, alpha=1.0, beta=0.0)

  h = gethandle(device(x))
  lrndesc = lrn_desc(n, lrnalpha, lrnbeta, k)
  ydesc = tensor_desc(y)
  dydesc = tensor_desc(dy)
  xdesc = tensor_desc(x)
  dxdesc = tensor_desc(dx)

  cudnnLRNCrossChannelBackward(h, lrndesc, mode, T[alpha], ydesc, y,
    dydesc, dy, xdesc, x, T[beta], dxdesc, dx)

  cudnnDestroyLRNDescriptor(lrndesc)
  cudnnDestroyTensorDescriptor(ydesc)
  cudnnDestroyTensorDescriptor(dydesc)
  cudnnDestroyTensorDescriptor(xdesc)
  cudnnDestroyTensorDescriptor(dxdesc)
  y
end
