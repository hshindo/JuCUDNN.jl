module JuCUDNN

using JuCUDA

include("../libcudnn-5/libcudnn.jl")
include("../libcudnn-5/libcudnn_types.jl")

@windows? (
  begin
    const libcudnn = Libdl.find_library(["cudnn64_5"])
  end : begin
    const libcudnn = Libdl.find_library(["libcudnn"])
  end)
isempty(libcudnn) && throw("CUDNN library cannot be found.")
info("CUDNN version: $(cudnnGetVersion())")

function checkstatus(status)
  if status != CUDNN_STATUS_SUCCESS
    Base.show_backtrace(STDOUT, backtrace())
    throw(bytestring(cudnnGetErrorString(status)))
  end
end

datatype(::Type{Float32}) = CUDNN_DATA_FLOAT
datatype(::Type{Float64}) = CUDNN_DATA_DOUBLE
datatype(::Type{Float16}) = CUDNN_DATA_HALF

include("handle.jl")
include("tensor.jl")
include("activation.jl")
include("convolution.jl")
include("dropout.jl")
include("filter.jl")
include("pooling.jl")
include("softmax.jl")
include("batchnorm.jl")
include("rnn.jl")

end
