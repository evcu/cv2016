nn = require 'nn'

local features = nn.Sequential()
features:add(nn.SpatialConvolution(3,64,11,11,2,2,2,2))       -- 64 -> 29
features:add(nn.ReLU(true))
features:add(nn.SpatialMaxPooling(3,3,2,2))                   -- 29 ->  14
features:add(nn.SpatialConvolution(64,192,5,5))       --  14 -> 10
features:add(nn.ReLU(true))
features:add(nn.SpatialMaxPooling(2,2,2,2))                   --  10 ->  5
features:add(nn.View(192*5*5))
features:add(nn.Dropout(0.5))
features:add(nn.Linear(192*5*5, 1000))
features:add(nn.ReLU())
features:add(nn.Dropout(0.5))
features:add(nn.Linear(1000, 43))

return features