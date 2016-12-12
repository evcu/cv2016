local nn = require 'nn'


local Convolution = nn.SpatialConvolution
local Tanh = nn.Tanh
local Max = nn.SpatialMaxPooling
local View = nn.View
local Linear = nn.Linear

local model  = nn.Sequential()

model:add(Convolution(1, 6, 5, 5, 1, 1, 2, 2))
model:add(Tanh())
model:add(Max(2,2,2,2))
model:add(Convolution(6, 96, 5, 5))
model:add(Tanh())
model:add(Max(2,2,2,2))
model:add(View(25*96))
model:add(Linear(25*96, 120))
model:add(Tanh())
model:add(Linear(120, 84))
model:add(Tanh())
model:add(Linear(84, 10))

return model
