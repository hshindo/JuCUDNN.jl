export
    rnn_training, rnn_training!, rnn_inference, rnn_inference!,
    ∇rnn_data!, ∇rnn_weight!
export
    CUDNN_RNN_RELU, CUDNN_RNN_TANH, CUDNN_LSTM, CUDNN_GRU, # network
    CUDNN_UNIDIRECTIONAL, CUDNN_BIDIRECTIONAL, # direction
    CUDNN_LINEAR_INPUT, CUDNN_SKIP_INPUT # input

function rnn_desc{T}(x::CuArray{T}, hiddensize, numlayers, input_t, dir_t, net_t,
    dropdesc)

    p = Ptr{Void}[0]
    cudnnCreateRNNDescriptor(p)
    cudnnSetRNNDescriptor(p[1], Cint(hiddensize), Cint(numlayers), dropdesc,
        input_t, dir_t, net_t, datatype(T))
    p[1]
end

function rnn_desc{T}(x::CuArray{T}, hiddensize, numlayers, input_t, dir_t, net_t,
    droprate, seed)

    h = gethandle(device(x))
    statessize_p = Cint[0]
    cudnnDropoutGetStatesSize(h, statessize_p)
    statessize = statessize_p[1]
    states = CuArray(Int8, Int(statessize))

    p = Ptr{Void}[0]
    cudnnCreateDropoutDescriptor(p)
    cudnnSetDropoutDescriptor(p[1], h, Cfloat(droprate), states, statessize, seed)

    rnndesc = rnn_desc(x, hiddensize, numlayers, input_t, dir_t, net_t, p[1])
    rnndesc, p[1], states
end

function rnn_training!{T}(xdims, x::CuArray{T}, hx::CuArray{T}, cx::CuArray{T},
    droprate, input_t, dir_t, net_t; seed=0)

    xdesc = tensor_desc(CuArray(T, xdims[1]...))
    xdescs = fill(xdesc, length(xdims))
    for i=1:length(xdims) xdescs[i] = tensor_desc(CuArray(T, xdims[i])) end
    hxdesc = tensor_desc(hx)
    cxdesc = tensor_desc(cx)

    y = similar(x)
    hy = similar(hx)
    cy = similar(cx)
    ydescs = similar(xdescs)
    for i=1:length(xdims) ydescs[i] = tensor_desc(CuArray(T, xdims[i])) end
    hydesc = tensor_desc(hy)
    cydesc = tensor_desc(cy)

    h = gethandle(device(x))
    rnndesc, dropdesc, dropstate = rnn_desc(x, size(hx,2), size(hx,4), input_t,
        dir_t, net_t, droprate, seed)
    wsize_p = Cint[0]
    cudnnGetRNNParamsSize(h, rnndesc, xdesc, wsize_p, datatype(T))
    wsize = wsize_p[1]
    w = curand(T, 1, 1, 1, Int(wsize/(T.size)))
    wdesc = filter_desc(w)

    worksize_p = Cint[0]
    cudnnGetRNNWorkspaceSize(h, rnndesc, Cint(length(xdescs)), xdescs, worksize_p)
    worksize = worksize_p[1]
    workspace = CuArray(Int8, Int(worksize))

    trainsize_p = Cint[0]
    cudnnGetRNNTrainingReserveSize(h, rnndesc, Cint(length(xdescs)), xdescs, trainsize_p)
    trainsize = trainsize_p[1]
    trainspace = CuArray(Int8, Int(trainsize))

    mdesc_p = Ptr{Void}[0]
    cudnnCreateFilterDescriptor(mdesc_p)
    mdesc = mdesc_p[1]
    m_p = Ptr{Void}[0]
    cudnnGetRNNLinLayerMatrixParams(h, rnndesc, Cint(0), xdesc, wdesc, w,
        Cint(0), mdesc, m_p)
    m = m_p[1]

    bdesc_p = Ptr{Void}[0]
    cudnnCreateFilterDescriptor(bdesc_p)
    bdesc = bdesc_p[1]
    b_p = Ptr{Void}[0]
    cudnnGetRNNLinLayerBiasParams(h, rnndesc, Cint(0), xdesc, wdesc, w,
        Cint(0), bdesc, b_p)
    b = b_p[1]

    cudnnRNNForwardTraining(h, rnndesc, Cint(length(xdescs)), xdescs, x, hxdesc,
        hx, cxdesc, cx, wdesc, w, ydescs, y, hydesc, hy, cydesc, cy, workspace,
        worksize, trainspace, trainsize)

    cudnnDestroyFilterDescriptor(bdesc)
    cudnnDestroyFilterDescriptor(mdesc)
    cudnnDestroyFilterDescriptor(wdesc)
    cudnnDestroyRNNDescriptor(rnndesc)
    cudnnDestroyTensorDescriptor(cydesc)
    cudnnDestroyTensorDescriptor(hydesc)
    for desc in ydescs cudnnDestroyTensorDescriptor(desc) end
    cudnnDestroyTensorDescriptor(cxdesc)
    cudnnDestroyTensorDescriptor(hxdesc)
    for desc in xdescs cudnnDestroyTensorDescriptor(desc) end
    cudnnDestroyTensorDescriptor(xdesc)
    w, y, hy, cy, dropdesc, dropstate
end

function rnn_training(x::CuArray, hx::CuArray, cx::CuArray, droprate, input_t,
    dir_t, net_t)

    xdims = fill((1, size(x,1), size(x,2), size(x,3)), size(x,4))
    rnn_training!(xdims, x, hx, cx, droprate, input_t, dir_t, net_t)
end

function rnn_inference!{T}(xdims, x::CuArray{T}, hx::CuArray{T}, cx::CuArray{T},
    w::CuArray{T}, dropdesc, input_t, dir_t, net_t)

    xdesc = tensor_desc(CuArray(T, xdims[1]...))
    xdescs = fill(xdesc, length(xdims))
    for i=1:length(xdims) xdescs[i] = tensor_desc(CuArray(T, xdims[i])) end
    hxdesc = tensor_desc(hx)
    cxdesc = tensor_desc(cx)

    y = similar(x)
    hy = similar(hx)
    cy = similar(cx)
    ydescs = similar(xdescs)
    for i=1:length(xdims) ydescs[i] = tensor_desc(CuArray(T, xdims[i])) end
    hydesc = tensor_desc(hy)
    cydesc = tensor_desc(cy)

    rnndesc = rnn_desc(x, size(hx,2), size(hx,4), input_t, dir_t, net_t, dropdesc)
    wdesc = filter_desc(w)

    h = gethandle(device(x))
    worksize_p = Cint[0]
    cudnnGetRNNWorkspaceSize(h, rnndesc, Cint(length(xdescs)), xdescs, worksize_p)
    worksize = worksize_p[1]
    workspace = CuArray(Int8, Int(worksize))

    mdesc_p = Ptr{Void}[0]
    cudnnCreateFilterDescriptor(mdesc_p)
    mdesc = mdesc_p[1]
    m_p = Ptr{Void}[0]
    cudnnGetRNNLinLayerMatrixParams(h, rnndesc, Cint(0), xdesc, wdesc, w,
        Cint(0), mdesc, m_p)
    m = m_p[1]

    bdesc_p = Ptr{Void}[0]
    cudnnCreateFilterDescriptor(bdesc_p)
    bdesc = bdesc_p[1]
    b_p = Ptr{Void}[0]
    cudnnGetRNNLinLayerBiasParams(h, rnndesc, Cint(0), xdesc, wdesc, w,
        Cint(0), bdesc, b_p)
    b = b_p[1]

    cudnnRNNForwardInference(h, rnndesc, Cint(length(xdescs)), xdescs, x, hxdesc,
        hx, cxdesc, cx, wdesc, w, ydescs, y, hydesc, hy, cydesc, cy, workspace,
        worksize)

    cudnnDestroyFilterDescriptor(bdesc)
    cudnnDestroyFilterDescriptor(mdesc)
    cudnnDestroyFilterDescriptor(wdesc)
    cudnnDestroyRNNDescriptor(rnndesc)
    cudnnDestroyTensorDescriptor(cydesc)
    cudnnDestroyTensorDescriptor(hydesc)
    for desc in ydescs cudnnDestroyTensorDescriptor(desc) end
    cudnnDestroyTensorDescriptor(cxdesc)
    cudnnDestroyTensorDescriptor(hxdesc)
    for desc in xdescs cudnnDestroyTensorDescriptor(desc) end
    cudnnDestroyTensorDescriptor(xdesc)
    y, hy, cy
end

function rnn_inference(x::CuArray, hx::CuArray, cx::CuArray, w::CuArray,
    dropdesc, input_t, dir_t, net_t)

    xdims = fill((1, size(x,1), size(x,2), size(x,3)), size(x,4))
    rnn_inference!(xdims, x, hx, cx, w, dropdesc, input_t, dir_t, net_t)
end

function ∇rnn_data!{T}(x::CuArray{T}, hx::CuArray{T}, cx::CuArray{T},
    w::CuArray{T}, y::CuArray{T}, dy::CuArray{T}, dhy::CuArray{T},
    dcy::CuArray{T}, dropdesc, input_t, dir_t, net_t,
    dx::CuArray{T}, dhx::CuArray{T}, dcx::CuArray{T})

    xdims = fill((1, size(x,1), size(x,2), size(x,3)), size(x,4))
    xdesc = tensor_desc(CuArray(T, xdims[1]...))
    xdescs = fill(xdesc, length(xdims))
    for i=1:length(xdims) xdescs[i] = tensor_desc(CuArray(T, xdims[i])) end
    hxdesc = tensor_desc(hx)
    cxdesc = tensor_desc(cx)
    ydescs = similar(xdescs)
    for i=1:length(xdims) ydescs[i] = tensor_desc(CuArray(T, xdims[i])) end
    dydescs = similar(xdescs)
    for i=1:length(xdims) dydescs[i] = tensor_desc(CuArray(T, xdims[i])) end
    dhydesc = tensor_desc(dhy)
    dcydesc = tensor_desc(dcy)
    dxdescs = similar(xdescs)
    for i=1:length(xdims) dxdescs[i] = tensor_desc(CuArray(T, xdims[i])) end
    dhxdesc = tensor_desc(dhx)
    dcxdesc = tensor_desc(dcx)

    rnndesc = rnn_desc(x, size(hx,2), size(hx,4), input_t, dir_t, net_t, dropdesc)
    wdesc = filter_desc(w)

    h = gethandle(device(x))
    worksize_p = Cint[0]
    cudnnGetRNNWorkspaceSize(h, rnndesc, Cint(length(xdescs)), xdescs, worksize_p)
    worksize = worksize_p[1]
    workspace = CuArray(Int8, Int(worksize))

    trainsize_p = Cint[0]
    cudnnGetRNNTrainingReserveSize(h, rnndesc, Cint(length(xdescs)), xdescs, trainsize_p)
    trainsize = trainsize_p[1]
    trainspace = CuArray(Int8, Int(trainsize))

    mdesc_p = Ptr{Void}[0]
    cudnnCreateFilterDescriptor(mdesc_p)
    mdesc = mdesc_p[1]
    m_p = Ptr{Void}[0]
    cudnnGetRNNLinLayerMatrixParams(h, rnndesc, Cint(0), xdesc, wdesc, w,
        Cint(0), mdesc, m_p)
    m = m_p[1]

    bdesc_p = Ptr{Void}[0]
    cudnnCreateFilterDescriptor(bdesc_p)
    bdesc = bdesc_p[1]
    b_p = Ptr{Void}[0]
    cudnnGetRNNLinLayerBiasParams(h, rnndesc, Cint(0), xdesc, wdesc, w,
        Cint(0), bdesc, b_p)
    b = b_p[1]

    cudnnRNNBackwardData(h, rnndesc, Cint(length(xdescs)), ydescs, y, dydescs, dy,
        dhydesc, dhy, dcydesc, dcy, wdesc, w, hxdesc, hx, cxdesc, cx, dxdescs, dx,
        dhxdesc, dhx, dcxdesc, dcx, workspace, worksize, trainspace, trainsize)

    cudnnDestroyFilterDescriptor(bdesc)
    cudnnDestroyFilterDescriptor(mdesc)
    cudnnDestroyFilterDescriptor(wdesc)
    cudnnDestroyRNNDescriptor(rnndesc)
    cudnnDestroyTensorDescriptor(dcxdesc)
    cudnnDestroyTensorDescriptor(dhxdesc)
    for desc in dxdescs cudnnDestroyTensorDescriptor(desc) end
    cudnnDestroyTensorDescriptor(dcydesc)
    cudnnDestroyTensorDescriptor(dhydesc)
    for desc in dydescs cudnnDestroyTensorDescriptor(desc) end
    for desc in ydescs cudnnDestroyTensorDescriptor(desc) end
    cudnnDestroyTensorDescriptor(cxdesc)
    cudnnDestroyTensorDescriptor(hxdesc)
    for desc in xdescs cudnnDestroyTensorDescriptor(desc) end
    cudnnDestroyTensorDescriptor(xdesc)
end

function ∇rnn_weight!{T}(x::CuArray{T}, hx::CuArray{T}, y::CuArray{T},
    w::CuArray{T}, dropdesc, input_t, dir_t, net_t, dw::CuArray{T})

    xdims = fill((1, size(x,1), size(x,2), size(x,3)), size(x,4))
    xdesc = tensor_desc(CuArray(T, xdims[1]...))
    xdescs = fill(xdesc, length(xdims))
    for i=1:length(xdims) xdescs[i] = tensor_desc(CuArray(T, xdims[i])) end
    hxdesc = tensor_desc(hx)
    ydescs = similar(xdescs)
    for i=1:length(xdims) ydescs[i] = tensor_desc(CuArray(T, xdims[i])) end

    rnndesc = rnn_desc(x, size(hx,2), size(hx,4), input_t, dir_t, net_t, dropdesc)
    wdesc = filter_desc(w)
    dwdesc = filter_desc(dw)

    h = gethandle(device(x))
    worksize_p = Cint[0]
    cudnnGetRNNWorkspaceSize(h, rnndesc, Cint(length(xdescs)), xdescs, worksize_p)
    worksize = worksize_p[1]
    workspace = CuArray(Int8, Int(worksize))

    trainsize_p = Cint[0]
    cudnnGetRNNTrainingReserveSize(h, rnndesc, Cint(length(xdescs)), xdescs, trainsize_p)
    trainsize = trainsize_p[1]
    trainspace = CuArray(Int8, Int(trainsize))

    mdesc_p = Ptr{Void}[0]
    cudnnCreateFilterDescriptor(mdesc_p)
    mdesc = mdesc_p[1]
    m_p = Ptr{Void}[0]
    cudnnGetRNNLinLayerMatrixParams(h, rnndesc, Cint(0), xdesc, wdesc, w,
        Cint(0), mdesc, m_p)
    m = m_p[1]

    bdesc_p = Ptr{Void}[0]
    cudnnCreateFilterDescriptor(bdesc_p)
    bdesc = bdesc_p[1]
    b_p = Ptr{Void}[0]
    cudnnGetRNNLinLayerBiasParams(h, rnndesc, Cint(0), xdesc, wdesc, w,
        Cint(0), bdesc, b_p)
    b = b_p[1]

    cudnnRNNBackwardWeights(h, rnndesc, Cint(length(xdescs)), xdescs, x, hxdesc,
        hx, ydescs, y, workspace, worksize, dwdesc, dw, trainspace,
        trainsize)

    cudnnDestroyFilterDescriptor(bdesc)
    cudnnDestroyFilterDescriptor(mdesc)
    cudnnDestroyFilterDescriptor(dwdesc)
    cudnnDestroyFilterDescriptor(wdesc)
    cudnnDestroyRNNDescriptor(rnndesc)
    for desc in ydescs cudnnDestroyTensorDescriptor(desc) end
    cudnnDestroyTensorDescriptor(hxdesc)
    for desc in xdescs cudnnDestroyTensorDescriptor(desc) end
    cudnnDestroyTensorDescriptor(xdesc)
end
