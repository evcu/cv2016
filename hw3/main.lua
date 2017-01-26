require 'torch'
require 'optim'
require 'os'
require 'optim'
require 'xlua'
-- require 'cunn'
-- require 'cudnn' -- faster convolutions

--[[
--  Hint:  Plot as much as you can.  
--  Look into torch wiki for packages that can help you plot.
--]]

local tnt = require 'torchnet'
local image = require 'image'
local optParser = require 'opts'
--local dbg   = require 'debugger'
local opt = optParser.parse(arg)
local DATA_PATH = (opt.data ~= '' and opt.data or './data/')

torch.setdefaulttensortype('torch.FloatTensor')
torch.save(opt.logDir ..'/'.. opt.jobID..'.opts', opt)
torch.setnumthreads(opt.nThr)
torch.manualSeed(opt.manualSeed)
-- cutorch.manualSeedAll(opt.manualSeed)


function getIterator(dataset)
    --[[
    -- Hint:  Use ParallelIterator for using multiple CPU cores
    --]]
    if opt.nThrLoad == 1 then
        return tnt.DatasetIterator{
            dataset = tnt.BatchDataset{
                batchsize = opt.batchsize,
                dataset = dataset
            }
        }
    else
        return getIteratorP(dataset)
    end

end

function getIteratorP(dataset)
  return tnt.ParallelDatasetIterator{
        nthread = opt.nThrLoad,
        init    = function() 
                    require 'torchnet'
                    pred_data = require("prepro/".. opt.pre)
                end,
        closure = function()
          return tnt.BatchDataset{
             batchsize = opt.batchsize,
             dataset = dataset
             }
         end
     }
end


pred_data = require("prepro/".. opt.pre)


local trainDataset = tnt.SplitDataset{
    partitions = {train=1-opt.val, val=opt.val},
    initialpartition = 'train',
    --[[
    --  Hint:  Use a resampling strategy that keeps the 
    --  class distribution even during initial training epochs 
    --  and then slowly converges to the actual distribution 
    --  in later stages of training.
    --]]
    dataset = tnt.ShuffleDataset{
        dataset = tnt.ListDataset{
            list = torch.range(1, pred_data.train.size):long(),
            load = function(idx)
                return {
                    input =  pred_data.train.data[idx],
                    target = pred_data.train.label[idx]
                }
            end
        }
    }
}

local testDataset = tnt.ListDataset{
    list = torch.range(1, pred_data.test.size):long(),
    load = function(idx)
        return {
            input = pred_data.test.data[idx],
            sampleId = pred_data.test.samplenames[idx],
            target = pred_data.test.label[idx]
        }
    end
}

--[[
-- Hint:  Use :cuda to convert your model to use GPUs
--]]
local model = require("models/".. opt.model)
local engine = tnt.OptimEngine()
local meter = tnt.AverageValueMeter()
local criterion = nn.CrossEntropyCriterion()
local clerr = tnt.ClassErrorMeter{topk = {1}}
local timer = tnt.TimeMeter()
local batch = 1
local b_size_Tr = math.floor((pred_data.train.size*(1-opt.val))/opt.batchsize)
local b_size_Va = math.floor((pred_data.train.size*opt.val)/opt.batchsize)
local b_size_Te = math.floor(pred_data.test.size/opt.batchsize)

engine.hooks.onStart = function(state)
    meter:reset()
    clerr:reset()
    timer:reset()
    batch = 1
    if state.training then
        mode = 'Train'
        b_size = b_size_Tr
    else
        mode = 'Val'
        b_size = b_size_Va
    end
end


local convergence_file = assert(io.open(opt.logDir ..'/'.. opt.jobID..".log", "w"))
convergence_file:write("Epoch,TrainError,ValidError,TestError\n")

engine.hooks.onForwardCriterion = function(state)
    meter:add(state.criterion.output)
    clerr:add(state.network.output, state.sample.target)
    if opt.verbose == true then
        print(string.format("%s Batch: %d/%d; avg. loss: %2.4f; avg. error: %2.4f",
                mode, batch, b_size, meter:value(), clerr:value{k = 1}))
    else
        xlua.progress(batch, b_size)
    end
    batch = batch + 1 -- batch increment has to happen here to work for train, val and test.
    timer:incUnit()
end

engine.hooks.onEnd = function(state)
    print(string.format("%s: avg. loss: %2.4f; avg. error: %2.4f, time: %2.4f",
    mode, meter:value(), clerr:value{k = 1}, timer:value()))
    convergence_file:write(string.format("%2.4f",clerr:value{k = 1}))
end

local epoch = 1


while epoch <= opt.nEpochs do
    convergence_file:write(tostring(epoch)..",")
    trainDataset:select('train')
    engine:train{
        network = model,
        criterion = criterion,
        iterator = getIterator(trainDataset),
        optimMethod = optim.sgd,
        maxepoch = 1,
        config = {
            learningRate = opt.LR,
            momentum = opt.momentum,
            learningRateDecay = opt.LRD
        }
    }
    convergence_file:write(tostring(epoch)..",")
    trainDataset:select('val')
    engine:test{
        network = model,
        criterion = criterion,
        iterator = getIterator(trainDataset)
    }
    convergence_file:write(tostring(epoch)..",")
    b_size = b_size_Te
    engine:test{
        network = model,
        criterion = criterion,
        iterator = getIterator(testDataset)
    }
    print('Done with Epoch '..tostring(epoch))
    convergence_file:write("\n")
    epoch = epoch + 1
end

local submission = assert(io.open(opt.logDir ..'/'.. opt.jobID..".csv", "w"))
submission:write("Filename,ClassId\n")
batch = 1

--[[
--  This piece of code creates the submission
--  file that has to be uploaded in kaggle.
--]]

engine.hooks.onForward = function(state)
    local fileNames  = state.sample.sampleId
    local _, pred = state.network.output:max(2)
    pred = pred - 1
    for i = 1, pred:size(1) do
        submission:write(string.format("%05d,%d\n", fileNames[i][1], pred[i][1]))
    end
    xlua.progress(batch, state.iterator.dataset:size())
    batch = batch + 1
end
engine.hooks.onEnd = function(state)
    submission:close()
    convergence_file:close()
end
engine:test{
    network = model,
    iterator = getIterator(testDataset)
}
model:clearState()
torch.save(opt.logDir ..'/'.. opt.jobID..'.t7', model)


print("The End!")
