type TensorDesc
  ptr::Ptr{Void}

  function TensorDesc{T}(a::CuArray{T})
    csize = Cint[size(a,i) for i=ndims(a):-1:1]
    cstrides = Cint[stride(a,i) for i=ndims(a):-1:1]
    p = Ptr{Void}[0]
    cudnnCreateTensorDescriptor(p)
    desc = new(p[1])
    finalizer(desc, cudnnDestroyTensorDescriptor)
    cudnnSetTensorNdDescriptor(desc, datatype(T), ndims(a), csize, cstrides)
    desc
  end
end

Base.unsafe_convert(::Type{Ptr{Void}}, td::TensorDesc) = td.ptr
