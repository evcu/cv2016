local PruningLayer, Parent = torch.class('nn.PruningLayer', 'nn.Module')

function PruningLayer:__init(mask,inplace)
   Parent.__init(self)
   self.train = true
   self.inplace = inplace
   self.mask = mask or torch.Tensor()
end

function PruningLayer:updateOutput(input)
   if self.inplace then
      self.output:set(input)
   else
      self.output:resizeAs(input):copy(input)
   end
   self.output:cmul(self.mask)
   return self.output
end

function PruningLayer:updateGradInput(input, gradOutput)
   if self.inplace then
      self.gradInput:set(gradOutput)
   else
      self.gradInput:resizeAs(gradOutput):copy(gradOutput)
   end
      self.gradInput:cmul(self.mask) -- simply mask the gradients with the noise vector
   return self.gradInput
end

function PruningLayer:setMask(new_mask)
   self.mask:set()
   self.mask = new_mask
end

function PruningLayer:__tostring__()
   return string.format('%s(%f)', torch.type(self), self.p)
end


function PruningLayer:clearState()
   if self.mask then
      self.mask:set()
   end
   return Parent.clearState(self)
end