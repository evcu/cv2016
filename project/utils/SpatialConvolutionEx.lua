-- local dbg = require 'debugger'
-- dbg()
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

function SpatialConvolution:accGradParameters(input, gradOutput, scale)
   assert(input.THNN, torch.type(input)..'.THNN backend not imported')
   scale = scale or 1
   backCompatibility(self)
   input.THNN.SpatialConvolutionMM_accGradParameters(
      input:cdata(),
      gradOutput:cdata(),
      self.gradWeight:cdata(),
      THNN.optionalTensor(self.gradBias),
      self.finput:cdata(),
      self.fgradInput:cdata(),
      self.kW, self.kH,
      self.dW, self.dH,
      self.padW, self.padH,
      scale
   )
   if self.mask ~= nil then
      self.gradWeight:cmul(self.mask)
   end
end
