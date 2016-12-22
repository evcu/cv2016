local pruner = {}
local verbose = 1
-- function pruner:pruneOld(model,func,module_nos,param)
-- 	local perc = torch.Tensor(#module_nos)
-- 	self.masks = {}
-- 	for i=1,#module_nos do
-- 		local ws = model:get(module_nos[i]).weight
-- 		self.masks[module_nos[i]] = func(ws,param[i])
-- 		local retained = torch.sum(self.masks[module_nos[i]])/torch.numel(ws)
-- 	    if verbose then
-- 	    	print('Module'..i ..': '.. retained*100 ..'% retained')
-- 	    end
-- 	    perc[i] = retained
-- 	end
-- 	return perc
-- end

function pruner:setVariables(model,f_pruner,f_train,f_test,eng)
	self.f_train = f_train
	self.f_pruner = f_pruner
	self.f_test = f_test
	self.engine = eng
	self.model = model
	self.IMPORTANCE_INIT = 1
end

function pruner:prune(layer_nos,param)
	local perc = torch.Tensor(#layer_nos)
	for i=1,#layer_nos do
		local mask = self:f_pruner(layer_nos[i],param[i])
		self.model:get(layer_nos[i]):setMask(mask)
		local retained = torch.sum(mask)/torch.numel(mask)
	    if verbose then
	    	print('Module'..layer_nos[i] ..': '.. retained*100 ..'% retained')
	    end
	    perc[i] = retained
	end
	local res = self.f_test(self.model)
	return perc , res
end

function pruner:reTrainAndTest(nEpochs)
	local dbg = require 'debugger'
	dbg()
	return self.f_test(self.f_train(self.model,nEpochs))
end

function pruner:getConnectionDiv(c_w,i)
	bfun = function(state)
			state.network:get(i).gradWeight:cdiv(c_w)
			state.network:get(i).weight:cdiv(c_w)
			if verbose then
				--print('Layer'..k..': Grad/Weight divided\n')
	        end
		end
	return bfun
end
function pruner:getConnectionMult(c_w,i)
	bfun = function(state)
			if verbose then
				print('getConnectionMult')
				print(state.network:get(i).weight[1])--TODO fix this!
	        end
	        state.network:get(i).weight:cmul(c_w)
		end
	return bfun
end

function pruner:calculateCompression()
	czero = 0 
	ctotal = 0
	for i=1,self.model:size() do
		if self.model:get(i).weight ~= nil then
			w = self.model:get(i).weight
			--print(i,w:numel())
			ctotal = ctotal + w:numel()
			czero = czero + w:eq(0):sum()
		end
	end

-- 	print(ctotal,czero)
	return czero/ctotal
end

--Mask functions:
function pruner:maskThreshold(l_i,thres)
	local ws = self.model:get(l_i).weight
	local mask = isCuda(torch.abs(ws):gt(thres):double())
	return mask
end

function pruner:maskPercentage(l_i,del_p)
	local layer = self.model:get(l_i)
	if del_p == 0 then
		 thres = -math.huge 
	else
	    local sorted = torch.sort(torch.abs(layer.weight:view(-1)))
	     thres = sorted[sorted:size(1)*del_p]
	end
	return self:maskThreshold(l_i,thres)
end

function pruner:maskL2(l_i,del_p)
	initial_weights = self.model:get(l_i).weight:clone()
	self.engine.hooks.onSample = self:getConnectionMult(initial_weights,l_i)
	self.engine.hooks.onBackward = self:getConnectionDiv(initial_weights,l_i)
	self.model:get(l_i).weight:fill(self.IMPORTANCE_INIT)
	res = self.f_train(self.model,1) 
	self.engine.hooks.onSample = function() end
	self.engine.hooks.onBackward = function() end
	mask = self:maskPercentage(l_i,del_p)
	self.model:get(l_i).weight = initial_weights
		local dbg = require 'debugger'
	dbg()
	return mask
end

function pruner:maskL1(l_i,del_p)
	initial_weights = self.model:get(l_i).weight:clone()
	old_onSample = self.engine.hooks.onSample 
	old_onBackward = self.engine.hooks.onBackward 
	self.engine.hooks.onSample = self:getConnectionMult(initial_weights,l_i)
	self.engine.hooks.onBackward = self:getConnectionDiv(initial_weights,l_i)
	self.model:get(l_i).weight:fill(self.IMPORTANCE_INIT)
	self.model:insert(nn.L1Penalty(1,true),l_i+2)
	res = self.f_train(self.model,1) 
	self.model:remove(l_i+2)
	self.engine.hooks.onSample = old_onSample
	self.engine.hooks.onBackward = old_onBackward
	mask = self:maskPercentage(l_i,del_p)
	self.model:get(l_i).weight = initial_weights
end


return pruner
