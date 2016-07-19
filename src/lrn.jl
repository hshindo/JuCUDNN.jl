type LRNDescriptor
  ptr::Ptr{Void}
end

function LRNDescriptor(n::Float64, alpha::Float64, beta::Float64, k::Float64)
  p = Ptr{Void}[0]
  cudnnCreateLRNDescriptor(p)
  cudnnSetLRNDescriptor(p[1], n, alpha, beta, k)
  desc = new(p[1])
  finalizer(desc, cudnnDestroyLRNDescriptor)
  desc
end

Base.unsafe_convert(::Type{Ptr{Void}}, desc::LRNDescriptor) = desc.ptr

function crosschanel()
  cudnnLRNCrossChannelForward()
end
