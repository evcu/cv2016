require 'torch'
require 'optim'
require 'os'
require 'xlua'

local tnt = require 'torchnet'
local image = require 'image'
local optParser = require 'utils.opts_main'
require 'utils.LinearEx'
require 'utils.SpatialConvolutionEx'
--local dbg   = require 'debugger'
local opt = optParser.parse(arg)
local mnist = require('data.'..opt.data)
torch.save(opt.logDir ..'/'.. opt.jobID..'.opts', opt)
torch.manualSeed(opt.manualSeed)
if opt.cuda then
    require 'cunn'
end
function isCuda(_item)
    if opt.cuda then
        _item = _item:cuda()
    else
        _item = _item:double()
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

if opt.test then
    local model = isCuda(torch.load('logs/'..opt.model..'.t7pruned'))
else
    local model = isCuda(torch.load('inp/'..opt.model..'.t7'))
end
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
end

function TestModel(given_model)
    engine:test{
        network = given_model,
        criterion = criterion,
        iterator = getIterator(testDataset)
    }
    return clerr:value{k = 1},meter:value()
end

function CalculateHessianValues(given_model)
    nn.hessian.enable()
    engine:train{
        network = given_model,
        criterion = criterion,
        iterator = getIterator(testDataset),
        optimMethod = optim.sgd,
            maxepoch = 1,
            config = {
                learningRate = 0, --no learning just hessian calculation
            }
    }
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

        print('Done with Epoch '..tostring(epoch))
        epoch = epoch + 1
    end
    return given_model
end

local init_acc = TestModel(model)
local pruner = require('utils.pruner')
if opt.test then
    isCuda(torch.load('inp/'..opt.model..'.t7'))
    print('Total Compression: '..pruner:calculateCompression())
    print('Accuracy: '..init_acc)
    os.exit()
end

local prunerFunc = ((opt.pruner =='emp') and pruner.maskEmprical) or ((opt.pruner =='taylor1') and pruner.maskTaylor1) or ((opt.pruner =='taylor2') and pruner.maskTaylor2) or ((opt.pruner =='l1') and pruner.maskL1) or ((opt.pruner =='l2') and pruner.maskL2) or ((opt.pruner =='mag') and pruner.maskPercentage) or nil
assert(prunerFunc ~= nil, 'Pruner function can\'t set, fix the code')
pruner:setVariables(model,prunerFunc,TrainModel,TestModel,CalculateHessianValues)

plot_file = assert(io.open(opt.logDir ..'/'..opt.jobID..".plotlog", "w"))
plot_file:write("LayerNo,Retained%,TestError\n")

i=1
while i <= opt.iPruning do
    local current_p
    for l,p in pairs(opt.l) do
        current_p = (p/opt.iPruning)*i
        retained,acc,oldmask = pruner:pruneLayer(l,current_p)
        if opt.reTrain then
            acc = pruner:reTrainAndTest(opt.nEpochs)
            if opt.reLoad then
                pruner.model = isCuda(torch.load('inp/'..opt.model..'.t7'))
            end
        end
        if acc<(init_acc-opt.acctradeoff) then --We can't tolerate that
            pruner:revertMask(l,oldmask)
            if opt.verbose then
                print('@ Layer '..l..': stopped pruning since drop in accuracy exceeded threshold provided' )
            end
            opt.l[l] = nil -- So we are done with pruning this layer.
            plot_file:write(l.. ",stop\n")
        else
            plot_file:write(l.. ","..retained .. ",".. acc .."\n")
        end
    end
    i = i + 1
end
print('Total Compression: '..pruner:calculateCompression())
plot_file:close()


--print(pruner:calculateCompression())
model:clearState()
torch.save(opt.logDir ..'/'.. opt.model:float()..'.t7pruned.f', model)
torch.save(opt.logDir ..'/'.. opt.model:double()..'.t7pruned.d', model)







