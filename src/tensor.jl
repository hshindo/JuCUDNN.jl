export
  CUDNN_NOT_PROPAGATE_NAN, CUDNN_PROPAGATE_NAN,

  CUDNN_OP_TENSOR_ADD, CUDNN_OP_TENSOR_MUL,
  CUDNN_OP_TENSOR_MIN, CUDNN_OP_TENSOR_MAX

function tensor_desc{T,N}(a::CuArray{T,N})
  if N < 4
    # might be inefficient
    s = [1,1,1,1]
    for i = 1:N
      s[i] = size(a, i)
    end
    a = reshape(a, tuple(s...))
  end
  csize = Cint[size(a,i) for i=ndims(a):-1:1]
  cstrides = Cint[stride(a,i) for i=ndims(a):-1:1]
  p = Ptr{Void}[0]
  cudnnCreateTensorDescriptor(p)
  cudnnSetTensorNdDescriptor(p[1], datatype(T), ndims(a), csize, cstrides)
  p[1]
end

"""
This function copies the scaled data from one tensor to another tensor with a different
layout. See CUDNN manual for details.
"""
function transform(x::CuArray)
  h = gethandle(x.dev)
  throw("Not implemented yet.")
end

"""
y = alpha*x + beta*y
"""
function add!{T}(x::CuArray{T}, y::CuArray{T}; alpha=1.0, beta=1.0)
  h = gethandle(x.dev)
  xdesc, ydesc = tensor_desc(x), tensor_desc(y)
  cudnnAddTensor(h, T[alpha], xdesc, x, T[beta], ydesc, y)
  cudnnDestroyTensorDescriptor(xdesc)
  cudnnDestroyTensorDescriptor(ydesc)
  y
end

"""
C = op ( alpha1[0] * A, alpha2[0] * B ) + beta[0] * C
"""
function op!{T}(op, A::CuArray{T}, B::CuArray{T}, C::CuArray{T};
  alpha1=1.0, alpha2=1.0, beta=0.0, nan_opt=CUDNN_NOT_PROPAGATE_NAN)

  h = gethandle(A.dev)
  p = Ptr{Void}[0]
  cudnnCreateOpTensorDescriptor(p)
  opdesc = p[1]
  cudnnSetOpTensorDescriptor(opdesc, op, datatype(T), nan_opt)
  adesc, bdesc, cdesc = tensor_desc(A), tensor_desc(B), tensor_desc(C)
  cudnnOpTensor(h, opdesc, T[alpha1], adesc, A, T[alpha2], bdesc, B, T[beta], cdesc, C)

  cudnnDestroyTensorDescriptor(adesc)
  cudnnDestroyTensorDescriptor(bdesc)
  cudnnDestroyTensorDescriptor(cdesc)
  C
end

function set!{T}(y::CuArray{T}, value)
  h = gethandle(y.dev)
  ydesc = tensor_desc(y)
  cudnnSetTensor(h, ydesc, y, T[value])
  cudnnDestroyTensorDescriptor(ydesc)
  y
end

function scale!{T}(y::CuArray{T}, alpha)
  h = gethandle(y.dev)
  ydesc = tensor_desc(y)
  cudnnScaleTensor(h, ydesc, y, T[alpha])
  cudnnDestroyTensorDescriptor(ydesc)
  y
end
