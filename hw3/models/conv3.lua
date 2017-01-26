local nn = require 'nn'


local Convolution = nn.SpatialConvolution
local ReLU = nn.ReLU
local Max = nn.SpatialMaxPooling
local View = nn.View
local Linear = nn.Linear

local model  = nn.Sequential()

model:add(Convolution(3, 64, 5, 5))
model:add(ReLU())
model:add(Max(2,2,2,2))
model:add(Convolution(64, 100, 5, 5))
model:add(ReLU())
model:add(Max(2,2,2,2))
model:add(View(2500))
model:add(Linear(2500, 500))
model:add(ReLU())
model:add(Linear(500, 43))

return model
