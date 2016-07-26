export dropout, dropout!, ∇dropout!

dropout(x::CuArray, droprate) = dropout!(x, similar(x), droprate)

function dropout!(x::CuArray, y::CuArray, droprate)
    h = gethandle(device(x))
    p = Ptr{Void}[0]
    cudnnCreateDropoutDescriptor(p)
    dropoutdesc = p[1]

    # TODO: make `states` to be initialized once for each device.
    statessize_p = Cint[0]
    cudnnDropoutGetStatesSize(h, statessize_p)
    statessize = statessize_p[1]
    states = CuArray(Int8, Int(statessize))

    xdesc = tensor_desc(x)
    ydesc = tensor_desc(y)
    reservesize_p = Cint[0]
    cudnnDropoutGetReserveSpaceSize(xdesc, reservesize_p)
    reservesize = reservesize_p[1]
    reservespace = CuArray(Int8, Int(reservesize))

    states_p = Ptr{Void}[0]
    cudnnSetDropoutDescriptor(dropoutdesc, h, Cfloat(droprate), states, statessize, 0)

    cudnnDropoutForward(h, dropoutdesc, xdesc, x, ydesc, y, reservespace, reservesize)

    cudnnDestroyDropoutDescriptor(dropoutdesc)
    cudnnDestroyTensorDescriptor(xdesc)
    cudnnDestroyTensorDescriptor(ydesc)
    y, states, statessize, reservespace, reservesize
end

function ∇dropout!(x::CuArray, dy::CuArray, droprate, states, statessize, reservespace, reservesize, dx::CuArray)
    h = gethandle(device(x))
    p = Ptr{Void}[0]
    cudnnCreateDropoutDescriptor(p)
    dropoutdesc = p[1]

    dydesc = tensor_desc(dy)
    dxdesc = tensor_desc(dx)
    cudnnSetDropoutDescriptor(dropoutdesc, h, Cfloat(droprate), states, statessize, 0)

    cudnnDropoutForward(h, dropoutdesc, dydesc, dy, dxdesc, dx, reservespace, reservesize)

    cudnnDestroyDropoutDescriptor(dropoutdesc)
    cudnnDestroyTensorDescriptor(dydesc)
    cudnnDestroyTensorDescriptor(dxdesc)
end
