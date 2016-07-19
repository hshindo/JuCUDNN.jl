export PoolingDesc, pooling!, pooling, ∇pooling!
export CUDNN_POOLING_MAX
export CUDNN_POOLING_AVERAGE_COUNT_INCLUDE_PADDING
export CUDNN_POOLING_AVERAGE_COUNT_EXCLUDE_PADDING

type PoolingDesc{N}
  ptr::Ptr{Void}
  window::NTuple{N,Int}
  padding::NTuple{N,Int}
  stride::NTuple{N,Int}
end

function PoolingDesc{N}(mode, window::NTuple{N,Int}, padding::NTuple{N,Int}, stride::NTuple{N,Int};
  maxpooling_nanopt=CUDNN_NOT_PROPAGATE_NAN)
  p = Ptr{Void}[0]
  cudnnCreatePoolingDescriptor(p)
  cwindow = Cint[window[i] for i=N:-1:1]
  cpadding = Cint[padding[i] for i=N:-1:1]
  cstride = Cint[stride[i] for i=N:-1:1]
  cudnnSetPoolingNdDescriptor(p[1], mode, maxpooling_nanopt, N, cwindow, cpadding, cstride)
  desc = PoolingDesc(p[1], window, padding, stride)
  finalizer(desc, cudnnDestroyPoolingDescriptor)
  desc
end

Base.unsafe_convert(::Type{Ptr{Void}}, desc::PoolingDesc) = desc.ptr

function pooling!{T}(x::CuArray{T}, y::CuArray{T}, poolingdesc; alpha=1.0, beta=0.0)
  h = gethandle(x.dev)
  xdesc = TensorDesc(x)
  ydesc = TensorDesc(y)
  cudnnPoolingForward(h, poolingdesc, T[alpha], xdesc, x, T[beta], ydesc, y)
end

function pooling(x::CuArray, poolingdesc; alpha=1.0)
  N = length(poolingdesc.window)
  w, p, s = poolingdesc.window, poolingdesc.padding, poolingdesc.stride
  outdims = Array(Int, N)
  for i = 1:N
    outdims[i] = (size(x,i) + 2*p[i] - w[i]) ÷ s[i] + 1
  end
  y = similar(x, outdims..., size(x,N+1), size(x,N+2))
  pooling!(x, y, poolingdesc, alpha)
end

function ∇pooling!{T}(poolingdesc, y::CuArray{T}, dy::CuArray{T},
  x::CuArray{T}, dx::CuArray{T}; alpha=1.0, beta=0.0)
  h = gethandle(x.dev)
  ydesc = TensorDesc(y)
  dydesc = TensorDesc(dy)
  xdesc = TensorDesc(x)
  dxdesc = TensorDesc(dx)
  cudnnPoolingBackward(h, poolingdesc, ydesc, y, dydesc, dy, xdesc, x,
    T[beta], dxdesc, dx)
end
