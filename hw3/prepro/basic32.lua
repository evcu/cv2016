local WIDTH, HEIGHT = 32, 32
local DATA_PATH = './data/'
local tnt = require 'torchnet'
local filename = 'data/prepro/basic32.t7'
--local dbg   = require 'debugger'
local RESAMPLE_FLAG = false

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


function resize(img)
    return image.scale(img, WIDTH,HEIGHT)
end

--[[
-- Hint:  Should we add some more transforms? shifting, scaling?
-- Should all images be of size 32x32?  Are we losing 
-- information by resizing bigger images to a smaller size?
--]]
function transformInput(inp)
    f = tnt.transform.compose{
        [1] = resize
    }
    return f(inp)
end

function getTrainSample(dataset, idx)
    r = dataset[idx]
    classId, track, file = r[9], r[1], r[2]
    file = string.format("%05d/%05d_%05d.ppm", classId, track, file)
    return transformInput(image.load(DATA_PATH .. '/train_images/'..file))
end

function getTrainLabel(dataset, idx)
    return torch.Tensor{dataset[idx][9] + 1}
end

function getTestSample(dataset, idx)
    r = dataset[idx]
    file = DATA_PATH .. "/test_images/" .. string.format("%05d.ppm", r[1])
    return transformInput(image.load(file))
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
    dataset.train.data = torch.FloatTensor(dataset.train.size,3,32,32)
    dataset.train.label = torch.FloatTensor(dataset.train.size,1)
    for i=1,dataset.train.size do
        dataset.train.data[i] =  getTrainSample(trainData, i)
        dataset.train.label[i][1] = getTrainLabel(trainData, i)         
    end

    dataset.test = {}
    dataset.test.size = testData:size(1)
    dataset.test.data = torch.FloatTensor(dataset.test.size,3,32,32)
    dataset.test.samplenames = torch.LongTensor(dataset.test.size,1)
    dataset.test.label = torch.FloatTensor(dataset.test.size,1)
    for i=1,dataset.test.size do
        dataset.test.data[i] =  getTestSample(testData, i)
        dataset.test.samplenames[i][1]  = testData[i][1]
        dataset.test.label[i][1] = testlabel[i]
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