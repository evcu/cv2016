local WIDTH, HEIGHT = 48, 48
local DATA_PATH = './data/'
local tnt = require 'torchnet'
local filename = 'data/prepro/cut48contra.t7'
--local dbg   = require 'debugger'
local RESAMPLE_FLAG = false
local nn = require 'nn'

--DON't submit this part

function string:split(delimiter)
  local result = { }
  local from  = 1
  local delim_from, delim_to = string.find( self, delimiter, from  )
  while delim_from do
    table.insert( result, string.sub( self, from , delim_from-1 ) )
    from  = delim_to + 1
    delim_from, delim_to = string.find( self, delimiter, from  )
  end
  table.insert( result, string.sub( self, from  ) )
  return result
end

function transformInput(im,r)
    return image.scale(image.crop(im,r[5],r[6],r[7],r[8]), WIDTH,HEIGHT)
end

function transformTestInput(im,r)
    return image.scale(image.crop(im,r[4],r[5],r[6],r[7]), WIDTH,HEIGHT)
end

function getTrainSample(dataset, idx)
    r = dataset[idx]
    classId, track, file = r[9], r[1], r[2]
    file = string.format("%05d/%05d_%05d.ppm", classId, track, file)
    return transformInput(image.load(DATA_PATH .. '/train_images/'..file),r)
end

function getTrainLabel(dataset, idx)
    return torch.Tensor{dataset[idx][9] + 1}
end

function getTestSample(dataset, idx)
    r = dataset[idx]
    file = DATA_PATH .. "/test_images/" .. string.format("%05d.ppm", r[1])
    return transformTestInput(image.load(file),r)
end


local f=io.open(filename,"r")
if f~=nil then 
    io.close(f)
    local dataset = torch.load(filename)
    return dataset 
else 
    local trainData = torch.load(DATA_PATH..'train.t7')
    local testData = torch.load(DATA_PATH..'test.t7')
    local dataset = {}
    local testlabel = torch.load(DATA_PATH..'testlabels.t7')

    dataset.train = {}
    dataset.train.size = trainData:size(1)
    dataset.train.data = torch.FloatTensor(dataset.train.size,3,WIDTH,HEIGHT)
    dataset.train.label = torch.FloatTensor(dataset.train.size,1)
    for i=1,dataset.train.size do
        dataset.train.data[i] =  getTrainSample(trainData, i)
        dataset.train.label[i][1] = getTrainLabel(trainData, i)         
    end

    dataset.test = {}
    dataset.test.size = testData:size(1)
    dataset.test.data = torch.FloatTensor(dataset.test.size,3,WIDTH,HEIGHT)
    dataset.test.samplenames = torch.LongTensor(dataset.test.size,1)
    dataset.test.label = torch.FloatTensor(dataset.test.size,1)
    for i=1,dataset.test.size do
        dataset.test.data[i] =  getTestSample(testData, i)
        dataset.test.samplenames[i][1]  = testData[i][1]
        dataset.test.label[i][1] = testlabel[i]
    end

    --PREPROCCESSING below is adapted from clementfarabet's tutorial 
    print '==> preprocessing data: colorspace RGB -> YUV'
    for i = 1,dataset.train.size do
       dataset.train.data[i] = image.rgb2yuv(dataset.train.data[i])
    end
    for i = 1,dataset.test.size do
       dataset.test.data[i] = image.rgb2yuv(dataset.test.data[i])
    end
    -- Name channels for convenience
    channels = {'y','u','v'}
    -- Normalize each channel, and store mean/std
    -- per channel. These values are important, as they are part of
    -- the trainable parameters. At test time, test data will be normalized
    -- using these values.
    print '==> preprocessing data: normalize each feature (channel) globally'
    mean = {}
    std = {}
    for i,channel in ipairs(channels) do
       -- normalize each channel globally:
       mean[i] = dataset.train.data[{ {},i,{},{} }]:mean()
       std[i] = dataset.train.data[{ {},i,{},{} }]:std()
       dataset.train.data[{ {},i,{},{} }]:add(-mean[i])
       dataset.train.data[{ {},i,{},{} }]:div(std[i])
    end
    -- Normalize test data, using the training means/stds
    for i,channel in ipairs(channels) do
       -- normalize each channel globally:
       dataset.test.data[{ {},i,{},{} }]:add(-mean[i])
       dataset.test.data[{ {},i,{},{} }]:div(std[i])
    end

    neighborhood = image.gaussian1D(13)

    -- Define our local normalization operator (It is an actual nn module, 
    -- which could be inserted into a trainable model):
    normalization = nn.SpatialContrastiveNormalization(3, neighborhood, 1)

    -- Normalize all channels locally:
    for i = 1,dataset.train.size do
      dataset.train.data[{ i,{c},{},{} }] = normalization:forward(dataset.train.data[{ i,{c},{},{} }])
    end
    for i = 1,dataset.test.size do
      dataset.test.data[{ i,{c},{},{} }] = normalization:forward(dataset.test.data[{ i,{c},{},{} }])
    end

    if RESAMPLE_FLAG then
        local N_LABELS = 43
        local l = dataset.train.label
        local occur = torch.Tensor(N_LABELS)
        for i=1,N_LABELS do
            occur[i]=l[l:eq(i)]:size(1)
        end
        local n_max = occur:max()
        local d = dataset.train.data
        local newdata = torch.Tensor(n_max*N_LABELS,d:size(2),d:size(3),d:size(4))
        local newlabels = torch.Tensor(n_max*N_LABELS,1)
        local global_i = 1
        for i=1,N_LABELS do
            local c = 0
            local j = 1
            while c < n_max do
                if j > dataset.train.size then
                    j =1
                end
                if l[j][1] == i then
                    newdata[global_i] = d[j]
                    newlabels[global_i] = l[j]
                    global_i = global_i + 1
                    c = c + 1
                end
                j = j + 1
            end
        end
        dataset.train.data = newdata
        dataset.train.label = newlabels
        dataset.train.size = newdata:size(1)
    end
    torch.save(filename,dataset)
    return dataset

end