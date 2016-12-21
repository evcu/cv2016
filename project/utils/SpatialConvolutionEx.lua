-- local dbg = require 'debugger'
-- dbg()

local THNN = require 'nn.THNN'
local SpatialConvolution = nn.SpatialConvolution

function SpatialConvolution:setMask(mask)
   if self.weight:isSameSizeAs(mask) then
      self.mask = mask
      self.weight:cmul(self.mask)
   else
      error('Size of Mask must be same as the weight tensor')
   end
end

function SpatialConvolution:removeMask(mask)
      self.mask = nil
end

local old = SpatialConvolution.accGradParameters
function SpatialConvolution:accGradParameters(input, gradOutput, scale)
   old(self,input, gradOutput, scale)
   if self.mask ~= nil then
      self.gradWeight:cmul(self.mask)
   end
end
