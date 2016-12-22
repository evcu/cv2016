local M = {}


local function split(str, sep)
    sep = sep or ','
    fields={}
    local matchfunc = string.gmatch(str, "([^"..sep.."]+)")
    if not matchfunc then return {str} end
    for str in matchfunc do
        table.insert(fields, str)
    end
    return fields
end

function readConf(path, sep, tonum)
    tonum = tonum or true
    sep = sep or ','
    local csvFile = {}
    local file = assert(io.open(path, "r"))
    for line in file:lines() do
        fields = split(line, sep)
        if tonum then -- convert numeric fields to numbers
            for i=1,#fields do
                fields[i] = tonumber(fields[i]) or fields[i]
            end
        end
        table.insert(csvFile, fields)
    end
    file:close()
    return csvFile[1],csvFile[2]
end


function M.parse(arg)
    local cmd = torch.CmdLine();
    cmd:text()
    cmd:text('The German Traffic Sign Recognition Benchmark: A multi-class classification ')
    cmd:text()
    cmd:text('Options:')
    cmd:option('-nEpochs',          1,            'Retrain epochs')
    cmd:option('-batchsize',        128,            'Batch size for epochs')
    cmd:option('-manualSeed',       '0',            'Manual seed for RNG')
    cmd:option('-LR',               0.01,            'initial learning rate')
    cmd:option('-momentum',         0.9,            'momentum')
    cmd:option('-logDir',          'logs',         'log directory')
    cmd:option('-reTrain',          false,         'log directory')
    cmd:option('-LSP',              0,          'layer Sensitivity Precision(LSP). Creates sensitivity values for pruning percantages got 20 times between [1,20]')
    cmd:option('-pruner',           'mag',          '\'mag\',\'taylor2\',\'taylor12\',\'taylor1\', \'l2 \' and \' l1\'')
    cmd:option('-l',                0,            'layer-id to be pruned')
    cmd:option('-p',                0,            'percentage to be pruned')
    cmd:option('-model',           'lenet5',             'Model to use for training')
    cmd:option('-data',            'mnist',             'Dataset generation')
    cmd:option('-jobID',            'localMain',             'Dataset generation')
    cmd:option('-verbose',          false,        'Print stats for every batch')
    cmd:option('-cuda',          false,        'Print stats for every batch')


    local opt = cmd:parse(arg or {})
    -- if opt.model == 'lenet5' then
    --     opt.layer-ids

    if opt.l > 0 then
        opt.l = {opt.l}
        opt.p = {opt.p}
    else
        opt.l , opt.p = readConf('inp/'..opt.model..'.conf')
    end
    return opt
end

return M
