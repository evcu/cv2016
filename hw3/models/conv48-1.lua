nn = require 'nn'

local features = nn.Sequential()
features:add(nn.SpatialConvolution(3,27,8,8))       -- 48 -> 41
features:add(nn.ReLU(true))
features:add(nn.SpatialMaxPooling(3,3,2,2))                   -- 41 ->  20
features:add(nn.SpatialConvolution(27,128,5,5))       --  20 -> 16
features:add(nn.ReLU(true))
features:add(nn.SpatialMaxPooling(3,3,2,2))                   --  16 ->  8
features:add(nn.View(128*7*7))
features:add(nn.Dropout(0.5))
features:add(nn.Linear(128*7*7, 320))
features:add(nn.ReLU())
features:add(nn.Dropout(0.5))
features:add(nn.Linear(320, 43))

return features