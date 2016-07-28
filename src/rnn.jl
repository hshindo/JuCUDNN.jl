export
    rnn_desc
export
    CUDNN_RNN_RELU, CUDNN_RNN_TANH, CUDNN_LSTM, CUDNN_GRU, # mode
    CUDNN_UNIDIRECTIONAL, CUDNN_BIDIRECTIONAL, # direction
    CUDNN_LINEAR_INPUT, CUDNN_SKIP_INPUT # inputmode

function rnn_desc{T}(x::CuArray{T}, hiddensize, numlayers, droprate, inputmode,
    direction, mode)

    h = gethandle(device(x))
    drop_p = Ptr{Void}[0]
    cudnnCreateDropoutDescriptor(drop_p)
    dropoutdesc = drop_p[1]
    statessize_p = Cint[0]
    cudnnDropoutGetStatesSize(h, statessize_p)
    statessize = statessize_p[1]
    states = CuArray(Int8, Int(statessize))
    cudnnSetDropoutDescriptor(dropoutdesc, h, Cfloat(droprate), states, statessize, 0)

    p = Ptr{Void}[0]
    cudnnCreateRNNDescriptor(p)
    cudnnSetRNNDescriptor(p[1], Cint(hiddensize), Cint(numlayers), dropoutdesc,
        inputmode, direction, mode, datatype(T))
    p[1], states, statessize
end
