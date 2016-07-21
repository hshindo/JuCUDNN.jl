workspace()
using JuCUDA
using JuCUDNN

ndevices()

x = curand(Float32,5,4,3,2)
x = curand(Float32,1000)
y = dropout(x, 0.5)
Array(x)
Array(y)
