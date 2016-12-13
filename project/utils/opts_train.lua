local M = {}

function M.parse(arg)
    local cmd = torch.CmdLine();
    cmd:text()
    cmd:text('The German Traffic Sign Recognition Benchmark: A multi-class classification ')
    cmd:text()
    cmd:text('Options:')
    cmd:option('-nEpochs',          300,            'Maximum epochs')
    cmd:option('-batchsize',        128,            'Batch size for epochs')
    cmd:option('-manualSeed',       '0',            'Manual seed for RNG')
    cmd:option('-LR',               0.1,            'initial learning rate')
    cmd:option('-momentum',         0.9,            'momentum')
    cmd:option('-logDir',          'logs',         'log directory')
    cmd:option('-outDir',          'inp',         'log directory')
    cmd:option('-model',           'lenet5',             'Model to use for training')
    cmd:option('-data',            'mnist',             'Dataset generation')
    cmd:option('-jobID',            'localTrain',             'Dataset generation')
    cmd:option('-verbose',          'false',        'Print stats for every batch')
    local opt = cmd:parse(arg or {})
    return opt
end

return M
