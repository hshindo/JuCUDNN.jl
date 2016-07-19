type Handle
  ptr::Ptr{Void}

  function Handle()
    p = Ptr{Void}[0]
    cudnnCreate(p)
    h = new(p[1])
    finalizer(h, cudnnDestroy)
    h
  end
end

Base.unsafe_convert(::Type{Ptr{Void}}, h::Handle) = h.ptr

const handles = Dict{Int,Handle}()

function gethandle(dev::Int)
  if !haskey(handles, dev)
    h = Handle()
    handles[dev] = h
    h
  else
    handles[dev]
  end
end
