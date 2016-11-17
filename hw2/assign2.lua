require 'nn'
require 'image'
require 'torch'
require 'env'
require 'trepl'

--[[

This file shows the modified example from the paper "Torchnet: An Open-Source Platform
for (Deep) Learning Research".

Revisions by Rob Fergus (fergus@cs.nyu.edu) and Christian Puhrsch (cpuhrsch@fb.com)
Version 1.0 (10/14/16)

Updated by Utku Evci for the homework of the Computer Vision class thougt by Rob Fergus.

--]]

local cmd = torch.CmdLine()
cmd:option('-lr', 0.1, 'learning rate')
cmd:option('-batchsize', 100, 'batchsize')
cmd:option('-mnist', false, 'use mnist')
cmd:option('-cifar', false, 'use cifar')
cmd:option('-epochs', 10 , 'epochs')
cmd:option('-showSamples', false, 'show sample data')
cmd:option('-showFL', false , 'show First Layer weights')
cmd:option('-q4', false , 'Train the model in q4')
cmd:option('-load', '' , 'Load the saved model from the given file_name and evaluate')
cmd:option('-save', '' , 'Save the model after training as file_name given')
cmd:option('-loadEval', 'n' , 'Save the resulting model')

local config = cmd:parse(arg)
local tnt   = require 'torchnet'
--local dbg   = require 'debugger'
-- to set breakpoint put just put: dbg() at desired line

local base_data_path = 'data/'

-- Dataprep for MNIST

if config.mnist == true then
    if not paths.filep(base_data_path .. 'train_small_28x28.t7') then
        local train = torch.load(base_data_path .. 'train_28x28.t7', 'ascii')
        local train_small = {}
        train_small.data   = train.data[{{1, 50000}, {}, {}, {}}]
        train_small.labels = train.labels[{{1, 50000}}]
        torch.save(base_data_path .. 'train_small_28x28.t7', train_small, 'ascii')
    end

    if not paths.filep(base_data_path .. 'valid.t7') then
        local train = torch.load(base_data_path .. 'train_28x28.t7', 'ascii')
        local valid = {}
        valid.data   = train.data[{{50001, 60000}, {}, {}, {}}]
        valid.labels = train.labels[{{50001, 60000}}]
        torch.save(base_data_path .. 'valid_28x28.t7', valid, 'ascii')
    end
end

------------------------------------------------------------------------
-- Build the dataloader

-- getDatasets returns a dataset that performs some minor transformation on
-- the input and the target (TransformDataset), shuffles the order of the
-- samples without replacement (ShuffleDataset) and merges them into
-- batches (BatchDataset).
local function getMnistIterator(datasets)
    local listdatasets = {}
    for _, dataset in pairs(datasets) do
      local list = torch.range(1, dataset.data:size(1)):totable()
    --  local list = torch.range(1, 50):totable()
        table.insert(listdatasets,
                    tnt.ListDataset{
                        list = list,
                        load = function(idx)
                            return {
                                input  = dataset.data[idx],
                                target = dataset.labels[idx]
                            } -- sample contains input and target
                        end
                    })
    end
    return tnt.DatasetIterator{
        dataset = tnt.BatchDataset{
            batchsize = config.batchsize,
            dataset = tnt.ShuffleDataset{
               dataset = tnt.TransformDataset{
                    transform = function(x)
               return {
              input  = x.input:view(-1):double(),
              target = torch.LongTensor{x.target + 1}
                        }
                    end,
                    dataset = tnt.ConcatDataset{
                        datasets = listdatasets
                    }
                },
            }
        }
    }
end

local function getCifarIterator(datasets)
    local listdatasets = {}
    for _, dataset in pairs(datasets) do
        local list = torch.range(1, dataset.data:size(1)):totable()
 --     local list = torch.range(1, 88):totable()
        table.insert(listdatasets,
                    tnt.ListDataset{
                        list = list,
                        load = function(idx)
                            return {
                                input  = dataset.data[{{}, idx}],
                                target = dataset.labels[{{}, idx}]
                            } -- sample contains input and target
                        end
                    })
    end
    return tnt.DatasetIterator{
        dataset = tnt.BatchDataset{
            batchsize = config.batchsize,
            dataset = tnt.ShuffleDataset{
               dataset = tnt.TransformDataset{
                    transform = function(x)
               return {
              input  = x.input:double():reshape(3,32,32),
              target = x.target:long():add(1),
               }
                    end,
                    dataset = tnt.ConcatDataset{
                        datasets = listdatasets
                    }
                },
            }
        }
    }
end


local function genLabeldedIm(im,label,f_size,bias)
    bias = bias or 0
	--Gets a 1*m*n or 3*m*n image. 
	local m=im:size(2)
	local n=im:size(3)
	local d=im:size(1)
	temp=im:cat(torch.zeros(d,f_size,n):byte(),2)
	temp=temp:cat(torch.zeros(d,m+f_size,f_size):byte(),3)
	if d == 1 then
		im_2d=torch.cat(temp,temp,1)
		temp=torch.cat(im_2d,temp,1)
	end
	temp=image.drawText(temp,label .. '',1,m+1)
	return image.drawRect(temp,1,1,n,m)
end

if config.showSamples == true then
	local totalSample=100
	local label_size=10
	--MNIST
    local m_images = torch.load(base_data_path .. 'train_28x28.t7', 'ascii')
    local m_sample_indices = torch.randperm(m_images.labels:size(1))[{{1,totalSample}}]:long()
    local my_l = m_images.labels:index(1,m_sample_indices)
	local my_im = m_images.data:index(1,m_sample_indices)
	im_with_label = torch.zeros(totalSample,3,my_im:size(3)+label_size,my_im:size(4)+label_size)
    for i=1,totalSample do
    	im_with_label[i]=genLabeldedIm(my_im[i],my_l[i],10)
    end
	image.display{image=im_with_label,nrow=10}    
	--CIFAR
	local cifardata = {torch.load(base_data_path .. 'cifar-10-torch/data_batch_1.t7', 'ascii'),
        torch.load(base_data_path .. 'cifar-10-torch/data_batch_2.t7', 'ascii'),
        torch.load(base_data_path .. 'cifar-10-torch/data_batch_3.t7', 'ascii'),
        torch.load(base_data_path .. 'cifar-10-torch/data_batch_4.t7', 'ascii')}
    local m_images={}
    m_images.data = torch.zeros(40000,3,32,32):byte()
    m_images.labels = torch.zeros(40000)
    local i = 1
	for _, cbatch in pairs(cifardata) do
		for j=1,cbatch.labels:size(2) do
			m_images.data[i] = cbatch.data[{{},j}]:reshape(3,32,32):byte()
        	m_images.labels[i] = cbatch.labels[{1,j}]
        	i = i + 1 
		end
    end
    local m_sample_indices = torch.randperm(m_images.labels:size(1))[{{1,totalSample}}]:long()
    local my_l = m_images.labels:index(1,m_sample_indices)
	local my_im = m_images.data:index(1,m_sample_indices)
	im_with_label = torch.zeros(totalSample,3,my_im:size(3)+label_size,my_im:size(4)+label_size)
    for i=1,totalSample do
    	im_with_label[i]=genLabeldedIm(my_im[i],my_l[i],10)
    end
	image.display{image=im_with_label,nrow=10}  
end



if config.cifar or config.mnist then
    ------------------------------------------------------------------------
    -- Make the model and the criterion

    local nout = 10 --same for both CIFAR and MNIST
    local nin = 1
    if config.mnist == true then nin = 784 end
    if config.cifar == true then nin = 3072 end

    local network = nn.Sequential()
    local criterion = nn.CrossEntropyCriterion()

    if config.q4 == true then --MLP for q4
        hidden_layer_sizes={1000}
        network:add(nn.Linear(nin,hidden_layer_sizes[1]))
        network:add(nn.Tanh())
        network:add(nn.Linear(hidden_layer_sizes[1],nout))
    elseif config.cifar == true then --CONV-NET for CIFAR
        convLayers={3,16,128}
        for i = 1, #convLayers-1 do
            network:add(nn.SpatialConvolution(convLayers[i],convLayers[i+1],5,5,1,1,2,2))
            network:add(nn.Tanh())
            network:add(nn.SpatialMaxPooling(2,2,2,2))
        end
        network:add(nn.View(-1,8192))
        hidden_layer_sizes={8192,64}
        network:add(nn.Linear(hidden_layer_sizes[1],hidden_layer_sizes[2]))
        network:add(nn.Tanh())
        network:add(nn.Linear(hidden_layer_sizes[2],nout))
    else--Default behaviour give to us -q3
        network:add(nn.Linear(nin, nout))
    end

    ------------------------------------------------------------------------
    -- Prepare torchnet environment for training and testing

    local trainiterator
    local validiterator
    local testiterator
    local datasets
    if config.mnist == true then
        datasets = {torch.load(base_data_path .. 'train_small_28x28.t7', 'ascii')}
        trainiterator = getMnistIterator(datasets)
        datasets = {torch.load(base_data_path .. 'valid_28x28.t7', 'ascii')}
        validiterator = getMnistIterator(datasets)
        datasets = {torch.load(base_data_path .. 'test_28x28.t7', 'ascii')}
        testiterator  = getMnistIterator(datasets)
    end
    if config.cifar == true then
        datasets = {torch.load(base_data_path .. 'cifar-10-torch/data_batch_1.t7', 'ascii'),
            torch.load(base_data_path .. 'cifar-10-torch/data_batch_2.t7', 'ascii'),
            torch.load(base_data_path .. 'cifar-10-torch/data_batch_3.t7', 'ascii'),
            torch.load(base_data_path .. 'cifar-10-torch/data_batch_4.t7', 'ascii')}
        trainiterator = getCifarIterator(datasets)
        datasets = {torch.load(base_data_path .. 'cifar-10-torch/data_batch_5.t7', 'ascii')}
        validiterator = getCifarIterator(datasets)
        datasets = {torch.load(base_data_path .. 'cifar-10-torch/test_batch.t7', 'ascii')}
        testiterator  = getCifarIterator(datasets)
    end

    -- If load is needed load and evaluate 
    if config.load ~= '' then
        network = torch.load(config.load)
    else --Train the model 
        local lr = config.lr
        local epochs = config.epochs

        print("Started training!")

        for epoch = 1, epochs do
            local timer = torch.Timer()
            local loss = 0
            local errors = 0
            local count = 0
            for d in trainiterator() do
                network:forward(d.input)
                criterion:forward(network.output, d.target)
                network:zeroGradParameters()
                criterion:backward(network.output, d.target)
                network:backward(d.input, criterion.gradInput)
                network:updateParameters(lr)

                loss = loss + criterion.output --criterion already averages over minibatch
                count = count + 1
                local _, pred = network.output:max(2)
                errors = errors + (pred:size(1) - pred:eq(d.target):sum())
            end
            loss = loss / count

            
            local validloss = 0
            local validerrors = 0
            count = 0
            for d in validiterator() do
                network:forward(d.input)
                criterion:forward(network.output, d.target)

                validloss = validloss + criterion.output --criterion already averages over minibatch
                count = count + 1
                local _, pred = network.output:max(2)
                validerrors = validerrors + (pred:size(1) - pred:eq(d.target):sum())
            end
            validloss = validloss / count

            print(string.format(
            'train | epoch = %d | lr = %1.4f | loss: %2.4f | error: %2.4f - valid | validloss: %2.4f | validerror: %2.4f | s/iter: %2.4f',
            epoch, lr, loss, errors, validloss, validerrors, timer:time().real
            ))
        end
        print("Finished training!")
    end

    --Test the model
	local testerrors = 0
	for d in testiterator() do
	    network:forward(d.input)
	    criterion:forward(network.output, d.target)
	    local _, pred = network.output:max(2)
	    testerrors = testerrors + (pred:size(1) - pred:eq(d.target):sum())
	end

	print(string.format('| test | error: %2.4f', testerrors))

    --Print the layers if needed
	if config.showFL == true then
        local pad_size = 10
        local function vizWeights(im,label,f_size,min_pix_val)
            min_pix_val = min_pix_val or 0
            --Gets a 1*m*n or 3*m*n image. 
            local m=im:size(2)
            local n=im:size(3)
            local d=im:size(1)
            --Padding zeros
            temp=im:cat(torch.zeros(d,f_size,n):double()+min_pix_val,2)
            temp=temp:cat(torch.zeros(d,m+f_size,f_size):double()+min_pix_val,3)
            --Increasing dim, since drawText and drawRect requires 3channels.
            if d == 1 then
                im_2d=torch.cat(temp,temp,1)
                temp=torch.cat(im_2d,temp,1)
            end
            temp=image.drawText(temp,label .. '',1,m+1)
            return image.drawRect(temp,1,1,n,m)
        end
        local l1_w = network:get(1).weight
        if config.mnist == true then
            l1_w = l1_w:reshape(l1_w:size(1),1,28,28)
            local net_min=torch.min(l1_w)
            local im_with_label = torch.zeros(l1_w:size(1),3,l1_w:size(3)+pad_size,l1_w:size(4)+pad_size)
            for i=1,l1_w:size(1) do
                im_with_label[i]=vizWeights(l1_w[i],i-1,pad_size,net_min) --maybe view is faster
            end
            image.display{image=im_with_label,nrow=math.floor(l1_w:size(1)/2),zoom=3}  
        elseif config.cifar == true then
            image.display{image=l1_w,nrow=4,zoom=20, padding=2}  
        end
	end

    --Save the model if neeed
    if config.save ~= '' then
        network:clearState()
        torch.save(config.save,network)
    end

end