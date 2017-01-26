local nn = require 'nn'


local Convolution = nn.SpatialConvolution
local ReLU = nn.ReLU
local Max = nn.SpatialMaxPooling
local View = nn.View
local Linear = nn.Linear

local model  = nn.Sequential()

model:add(Convolution(3, 32, 5, 5))
model:add(ReLU())
model:add(Max(2,2,2,2))
model:add(Convolution(32, 256, 5, 5))
model:add(ReLU())
model:add(Max(2,2,2,2))
model:add(View(6400))
model:add(Linear(6400, 1200))
model:add(ReLU())
model:add(Linear(1200, 300))
model:add(ReLU())
model:add(Linear(300, 43))

return model
