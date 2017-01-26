local nn = require 'nn'


local Convolution = nn.SpatialConvolution
local Tanh = nn.Tanh
local Max = nn.SpatialMaxPooling
local View = nn.View
local Linear = nn.Linear

local model  = nn.Sequential()

model:add(Convolution(3, 16, 5, 5))
model:add(Tanh())
model:add(Max(2,2,2,2))
model:add(Convolution(16, 128, 5, 5))
model:add(Tanh())
model:add(Max(2,2,2,2))
model:add(View(3200))
model:add(Linear(3200, 64))
model:add(Tanh())
model:add(Linear(64, 43))

return model
