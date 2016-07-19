type ActivationDesc
  ptr::Ptr{Void}
end

function ActivationDesc(mode, relu_nanopt=CUDNN_NOT_PROPAGATE_NAN, relu_ceiling=1.0)
  p = Ptr{Void}[0]
  cudnnCreateActivationDescriptor(p)
  cudnnSetActivationDescriptor(p[1], mode, relu_nanopt, relu_ceiling)
  desc = ActivationDesc(p[1])
  finalizer(desc, cudnnDestroyActivationDescriptor)
  desc
end

Base.unsafe_convert(::Type{Ptr{Void}}, desc::ActivationDesc) = desc.ptr
