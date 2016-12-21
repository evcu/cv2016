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

local function updateAddBuffer(self, input)
   local nframe = input:size(1)
   self.addBuffer = self.addBuffer or input.new()
   if self.addBuffer:nElement() ~= nframe then
      self.addBuffer:resize(nframe):fill(1)
   end
end

function Linear:accGradParameters(input, gradOutput, scale)
   scale = scale or 1
   if input:dim() == 1 then
      self.gradWeight:addr(scale, gradOutput, input)
      if self.bias then self.gradBias:add(scale, gradOutput) end
   elseif input:dim() == 2 then
      self.gradWeight:addmm(scale, gradOutput:t(), input)
      if self.bias then
         -- update the size of addBuffer if the input is not the same size as the one we had in last updateGradInput
         updateAddBuffer(self, input)
         self.gradBias:addmv(scale, gradOutput:t(), self.addBuffer)
      end
   end
   if self.mask ~= nil then
      self.gradWeight:cmul(self.mask)
   end
end


