require 'nn'

local net = nn.Sequential()
nout = 10 --same for both CIFAR and MNIST
nin = 3072
convLayers={3,16,128}
for i = 1, #convLayers-1 do
    net:add(nn.SpatialConvolution(convLayers[i],convLayers[i+1],5,5,1,1,2,2))
    net:add(nn.ReLU())
    net:add(nn.SpatialMaxPooling(2,2,2,2))
end
net:add(nn.View(-1,8192))
hidden_layer_sizes={8192,64}
net:add(nn.Linear(hidden_layer_sizes[1],hidden_layer_sizes[2]))
net:add(nn.ReLU())
net:add(nn.Linear(hidden_layer_sizes[2],nout))

return net