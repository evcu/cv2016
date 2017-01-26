require 'nn'

local vgg = nn.Sequential()
-- building block
local function ConvBNReLU(nInputPlane, nOutputPlane)
  vgg:add(nn.SpatialConvolution(nInputPlane, nOutputPlane, 3,3, 1,1, 1,1))
  vgg:add(nn.SpatialBatchNormalization(nOutputPlane,1e-3))
  vgg:add(nn.ReLU(true))
  return vgg
end
-- Will use "ceil" MaxPooling because we want to save as much feature space as we can
local MaxPooling = nn.SpatialMaxPooling

ConvBNReLU(3,32):add(nn.Dropout(0.3))
ConvBNReLU(32,32)
vgg:add(MaxPooling(2,2,2,2):ceil())
ConvBNReLU(32,64):add(nn.Dropout(0.4))
ConvBNReLU(64,64)
vgg:add(MaxPooling(2,2,2,2):ceil())
ConvBNReLU(64,128):add(nn.Dropout(0.4))
ConvBNReLU(128,128)
vgg:add(MaxPooling(2,2,2,2):ceil())
vgg:add(nn.View(6*6*128))
vgg:add(nn.Dropout(0.5))
vgg:add(nn.Linear(6*6*128,512))
vgg:add(nn.BatchNormalization(512))
vgg:add(nn.ReLU(true))
vgg:add(nn.Dropout(0.5))
vgg:add(nn.Linear(512,43))

return vgg