nn = require 'nn'
image = require 'image'
-- Define the normalization neighborhood:
neighborhood = image.gaussian1D(13)

-- Define our local normalization operator (It is an actual nn module, 
-- which could be inserted into a trainable model):

local features = nn.Sequential()
features:add(nn.SpatialContrastiveNormalization(3, neighborhood, 1):float())
features:add(nn.SpatialConvolution(3,27,11,11,2,2,2,2))       -- 64 -> 29
features:add(nn.ReLU(true))
features:add(nn.SpatialMaxPooling(3,3,2,2))                   -- 29 ->  14
features:add(nn.SpatialConvolution(27,128,5,5))       --  14 -> 10
features:add(nn.ReLU(true))
features:add(nn.SpatialMaxPooling(2,2,2,2))                   --  10 ->  5
features:add(nn.View(128*5*5))
features:add(nn.Dropout(0.5))
features:add(nn.Linear(128*5*5, 320))
features:add(nn.ReLU())
features:add(nn.Dropout(0.5))
features:add(nn.Linear(320, 43))

return features