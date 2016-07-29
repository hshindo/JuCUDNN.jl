workspace()
using JuCUDA
using JuCUDNN

ndevices()

x = curand(Float32,5,4,3,2)
hiddensize = 1
numlayers = 1
droprate = 0.5
desc = rnn_desc(x, hiddensize, numlayers, droprate, CUDNN_LINEAR_INPUT, CUDNN_UNIDIRECTIONAL, CUDNN_RNN_RELU)


JuCUDNN.cudnnGetVersion()

x = curand(Float32,5,4,3,2)
x = curand(Float32,1000)
y = dropout(x, 0.5)
Array(x)
Array(y)

x = curand(Float32,5,4,3,2)
Array(x)
y, s, ssize, rspace, rsize = dropout(x, 0.5)
dy=y
dx=zeros(x)
∇dropout!(x, dy, 0.5, s, ssize, rspace, rsize, dx)
Array(dx)
