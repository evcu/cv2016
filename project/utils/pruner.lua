local pruner = {}
local verbose = 1
function pruner:pruneOld(model,func,module_nos,param)
	local perc = torch.Tensor(#module_nos)
	self.masks = {}
	for i=1,#module_nos do
		local ws = model:get(module_nos[i]).weight
		self.masks[module_nos[i]] = func(ws,param[i])
		local retained = torch.sum(self.masks[module_nos[i]])/torch.numel(ws)
	    if verbose then
	    	print('Module'..i ..': '.. retained*100 ..'% retained')
	    end
	    perc[i] = retained
	end
	return perc
end

function pruner:setVariables(model,f_pruner,f_train,f_test,eng)
	self.f_train = f_train
	self.f_pruner = f_pruner
	self.f_test = f_test
	self.engine = eng
	self.model = model
end

function pruner:prune(module_nos,param)
	local perc = torch.Tensor(#module_nos)
	self.masks = {}
	for i=1,#module_nos do
		local ws = self.model:get(module_nos[i]).weight
		self.masks[module_nos[i]] = self.f_pruner(ws,param[i])
		local retained = torch.sum(self.masks[module_nos[i]])/torch.numel(ws)
	    if verbose then
	    	print('Module'..i ..': '.. retained*100 ..'% retained')
	    end
	    perc[i] = retained
	end
	local res = self.f_test(self.model)
	return perc , res
end

function pruner:reTrain(nEpochs)
	self.engine.hooks.onBackward = self:getOnBackwardPruner() -- get recent mask
	res = self.f_train(self.model,nEpochs)
	return res
end

function pruner.maskThreshold(weights,thres)
	local mask = torch.abs(weights):gt(thres):double()
    weights:cmul(mask)
	return mask
end

function pruner.maskPercentage(weights,del_p)
	if del_p == 0 then
		 thres = -math.huge 
	else
	    local sorted = torch.sort(torch.abs(weights:view(-1)))
	     thres = sorted[sorted:size(1)*del_p]
	end
	return pruner.maskThreshold(weights,thres)
end

function pruner:getOnBackwardPruner()
	bfun = function(state)
		    for k,v in pairs(self.masks) do
				state.network:get(k).gradWeight:cmul(v)
				if verbose then
					--print('Layer'..k..': Gradprunned\n')
		        end
			end
		end
	return bfun
end



return pruner
