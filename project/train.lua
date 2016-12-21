require 'torch'
require 'optim'
require 'os'
require 'xlua'

local tnt = require 'torchnet'
local image = require 'image'
local optParser = require 'utils.opts_train'
--local dbg   = require 'debugger'
local opt = optParser.parse(arg)
local mnist = require('data.'..opt.data)
torch.save(opt.logDir ..'/'.. opt.jobID..'.opts_train', opt)
torch.manualSeed(opt.manualSeed)
if opt.cuda then
    require 'cunn'
end

function isCuda(_item)
    if opt.cuda then
        _item:cuda()
    end
    return _item
end

function getIterator(dataset)
        return tnt.DatasetIterator{
            dataset = tnt.BatchDataset{
                batchsize = opt.batchsize,
                dataset = dataset
            }
        }
end

local trainset = mnist.traindataset()
local testset = mnist.testdataset()
local trainDataset = tnt.ShuffleDataset{
        dataset = tnt.ListDataset{
        list = torch.range(1, trainset.data:size(1)):long(),
        load = function(idx)
            return {
                input =  isCuda(trainset.data[{{idx},{},{}}]:double():div(256)),
                target = isCuda(torch.LongTensor({trainset.label[idx]+1}))
            }
        end
        }
    }

local testDataset = tnt.ListDataset{
    list = torch.range(1, testset.data:size(1)):long(),
    load = function(idx)
        return {
            input =  isCuda(testset.data[{{idx},{},{}}]:double():div(256)),
            target = isCuda(torch.LongTensor({testset.label[idx]+1}))
        }
    end
}


local model = isCuda(require("models/".. opt.model))
local engine = tnt.OptimEngine()
local meter = tnt.AverageValueMeter()
local criterion = isCuda(nn.CrossEntropyCriterion())
local clerr = tnt.ClassErrorMeter{topk = {1}}
local timer = tnt.TimeMeter()
local batch = 1

engine.hooks.onStart = function(state)
    meter:reset()
    clerr:reset()
    timer:reset()
    batch = 1
end

engine.hooks.onForwardCriterion = function(state)
    meter:add(state.criterion.output)
    clerr:add(state.network.output, state.sample.target)
    b_size = state.iterator.dataset:size()
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
convergence_file = assert(io.open(opt.logDir ..'/'.. opt.jobID..".log", "w"))
convergence_file:write("Epoch,TrainError,TestError\n")

local epoch = 1
while epoch <= opt.nEpochs do
    convergence_file:write(tostring(epoch)..",")
    engine:train{
        network = model,
        criterion = criterion,
        iterator = getIterator(trainDataset),
        optimMethod = optim.sgd,
        maxepoch = 1,
        config = {
            learningRate = opt.LR,
            momentum = opt.momentum,
        }
    }
    convergence_file:write(",")
    engine:test{
        network = model,
        criterion = criterion,
        iterator = getIterator(testDataset)
    }
    print('Done with Epoch '..tostring(epoch))
    convergence_file:write("\n")
    epoch = epoch + 1
end

convergence_file:close()
model:clearState()
torch.save(opt.outDir ..'/'.. opt.jobID..'.t7', model)
print("Training Ended!")
