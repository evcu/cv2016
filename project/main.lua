require 'torch'
require 'optim'
require 'os'
require 'xlua'

local tnt = require 'torchnet'
local image = require 'image'
local optParser = require 'utils.opts_main'
--local dbg   = require 'debugger'
local opt = optParser.parse(arg)
local mnist = require('data.'..opt.data)
torch.save(opt.logDir ..'/'.. opt.jobID..'.opts', opt)
torch.manualSeed(opt.manualSeed)

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
                input =  trainset.data[{{idx},{},{}}]:double():div(256),
                target = torch.LongTensor({trainset.label[idx]+1})
            }
        end
        }
    }

local testDataset = tnt.ListDataset{
    list = torch.range(1, testset.data:size(1)):long(),
    load = function(idx)
        return {
            input =  testset.data[{{idx},{},{}}]:double():div(256),
            target = torch.LongTensor({testset.label[idx]+1})
        }
    end
}


local model = torch.load('inp/'..opt.model..'.t7')
local engine = tnt.OptimEngine()
local meter = tnt.AverageValueMeter()
local criterion = nn.CrossEntropyCriterion()
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
end

function TestModel(given_model)
    engine:test{
        network = given_model,
        criterion = criterion,
        iterator = getIterator(testDataset)
    }
    return clerr:value{k = 1}
end

function TrainModel(given_model,n_epoch)
    local epoch = 1
    while epoch <= n_epoch do
        engine:train{
                network = given_model,
                criterion = criterion,
                iterator = getIterator(trainDataset),
                optimMethod = optim.sgd,
                maxepoch = 1,
                config = {
                    learningRate = opt.LR,
                    momentum = opt.momentum,
                }
            }

        engine:test{
            network = model,
            criterion = criterion,
            iterator = getIterator(testDataset)
        }
        print('Done with Epoch '..tostring(epoch))
        epoch = epoch + 1
    end
    return clerr:value{k = 1}
end


local pruner = require('utils.pruner')
pruner:setVariables(model,pruner.maskPercentage,TrainModel,TestModel,engine)
if opt.LSP ~= 0 then
    plot_file = assert(io.open(opt.logDir ..'/'..opt.jobID.."-".. opt.l[1]..".plotlog", "w"))
    plot_file:write("Retained%,TestError\n")
    for i=0,1,1/opt.LSP do
        retained,acc = pruner:prune(opt.l,{i})
        if opt.reTrain then
            acc = pruner:reTrain(opt.nEpochs)
        end
        plot_file:write(retained[1] .. ",".. acc .."\n")
    end
    plot_file:close()
else
    pruner:prune(opt.l,opt.p)
    if opt.reTrain then
        pruner:reTrain(opt.nEpochs)
    end 
end

model:clearState()
torch.save(opt.logDir ..'/'.. opt.model..'.t7pruned', model)






