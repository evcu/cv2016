local Linear = nn.Linear
function Linear:setMask(mask)
   if self.weight:isSameSizeAs(mask) then
      self.mask = mask
      self.weight:cmul(self.mask)
   else
      error('Size of Mask must be same as the weight tensor')
   end
end

function Linear:removeMask(mask)
      self.mask = nil
end

local old = Linear.accGradParameters
function Linear:accGradParameters(input, gradOutput, scale)
   old(self,input, gradOutput, scale)
   if self.mask ~= nil then
      self.gradWeight:cmul(self.mask)
   end
end


