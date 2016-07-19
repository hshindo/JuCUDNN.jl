export dropout

type DropoutDesc
  ptr::Ptr{Void}
end

function DropoutDesc(dropout::Float64)
  p = Ptr{Void}[0]
  cudnnCreateDropoutDescriptor(p)
  cudnnSetDropoutDescriptor(p[1], h)

  desc = new(p[1])
  finalizer(desc, cudnnDestroyDropoutDescriptor)
  desc
end

Base.unsafe_convert(::Type{Ptr{Void}}, desc::DropoutDesc) = desc.ptr

function dropout()
  p = Ptr{Void}[0]
  cudnnCreateDropoutDescriptor(p)
  dropoutdesc = p[1]

  xdesc = TensorDesc(x)
  ydesc = TensorDesc(y)

  cudnnDropoutForward(h, dropoutdesc, xdesc, x, ydesc, y, reservespace, reservesize)

  cudnnDestroyDropoutDescriptor(dropoutdesc)
end
