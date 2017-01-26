local nn = require 'nn'


local Convolution = nn.SpatialConvolution
local ReLU = nn.ReLU
local Max = nn.SpatialMaxPooling
local View = nn.View
local Linear = nn.Linear

local model  = nn.Sequential()

model:add(Convolution(3, 12, 5, 5))
model:add(ReLU())
model:add(Max(2,2,2,2))
model:add(Convolution(12, 48, 5, 5))
model:add(ReLU())
model:add(Max(2,2,2,2))
model:add(View(1200))
model:add(Linear(1200, 240))
model:add(ReLU())
model:add(Linear(240, 43))

return model
