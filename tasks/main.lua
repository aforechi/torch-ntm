--[[

    Training a NTM to memorize input.

    The current version seems to work, giving good output after 5000 iterations
    or so. Proper initialization of the read/write weights seems to be crucial
    here.

--]]

require('../')
require('./util')
require('../datagen/CopyDataGenerator')
require('../datagen/RecallDataGenerator')
require('optim')
require('sys')

local cmd = torch.CmdLine()
cmd:text()
cmd:text('Options:')
cmd:option('--seed', false, 'torch rng seed')
cmd:option('--print_interval', 100, 'print progress every n time steps')
cmd:option('--save_interval', 1000, 'print progress every n time steps')
cmd:option('--save_file','ntm','filename to autosave the checkpont to. Will be inside checkpoint_dir/')
cmd:option('--init_from', '', 'initialize network parameters from checkpoint at this path')
cmd:option('--minLen', 2, 'minimum initial input length')
cmd:option('--maxLen', 20, 'maximum initial input length')

local checkpoint_dir = 'checkpoints'
local opt = cmd:parse(arg or {})

if opt.seed then torch.manualSeed(opt.seed) end

-- NTM config
local config = {
    input_dim = 10,
    output_dim = 10,
    mem_rows = 128,
    mem_cols = 20,
    cont_dim = 100
}

local tasks = {'copy'}
-- local tasks = {'recall'}
local dataGenerators = {
    copy = CopyDataGenerator(config.input_dim - 2),
    recall = RecallDataGenerator(config.input_dim - 2, 1)
}

local function forward(model, ioExample, print_flag)
    if print_flag then print('write head max') end
    for i=1,ioExample.initialInput:size(1) do
        model:forward(ioExample.initialInput[i])
        if print_flag then print_write_max(model) end
    end

    -- get output
    local output = torch.Tensor(ioExample.target:size())
    local criteria = {}
    local loss = 0
    if print_flag then print('read head max') end
    for i=1, ioExample.queryInput:size(1) do
        criteria[i] = nn.BCECriterion()
        criteria[i].sizeAverage = false
        output[i] = model:forward(ioExample.queryInput[i])
        output[i]:cmul(ioExample.targetMask[i])
        loss = loss + criteria[i]:forward(output[i], ioExample.target[i])
        if print_flag then print_read_max(model) end
    end
    return output, criteria, loss
end

local function backward(model, ioExample, output, criteria)
    for i=ioExample.queryInput:size(1), 1, -1 do
        local gradOutputs = criteria[i]:backward(output[i], ioExample.target[i])
        model:backward(ioExample.queryInput[i], gradOutputs)
    end

    local zeros = torch.zeros(output:size(2))
    for i=ioExample.initialInput:size(1), 1, -1 do
        model:backward(ioExample.initialInput[i], zeros)
    end
end

local model, start_iter
if string.len(opt.init_from) > 0 then
    print('loading a model from checkpoint ' .. opt.init_from)
    local checkpoint = torch.load(opt.init_from)
    model = checkpoint.model
    start_iter = checkpoint.iter + 1
else
    model = ntm.NTM(config)
    start_iter = 1
end

local params, grads = model:getParameters()

local num_iters = 10000
local start = sys.clock()

print(string.rep('=', 80))
print("NTM copy task")
print('training up to ' .. num_iters .. ' iteration(s)')
print('min sequence length = ' .. opt.minLen)
print('max sequence length = ' .. opt.maxLen)
print(string.rep('=', 80))
print('num params: ' .. params:size(1))

local rmsprop_state = {
    learningRate = 1e-4,
    momentum = 0.9,
    decay = 0.95
}

-- train
local lossHistory = {}
local summedLossSave, summedLossPrint = 0, 0
for iter = start_iter, num_iters do
    local print_flag = (iter % opt.print_interval == 0)
    local save_flag = (iter % opt.save_interval == 0)
    
    if save_flag then
        local avgLoss = summedLossSave / opt.save_interval
        local savefile = string.format('%s/%s_iter%.2f_%.4f.t7', checkpoint_dir, opt.save_file, iter, avgLoss)
        print('saving model to ' .. savefile)
        torch.save(savefile, {
            model = model,
            iter = iter
        })
        summedLossSave = 0
    end

    local feval = function(x)
        if print_flag then
            print(string.rep('-', 80))
            print('iter = ' .. iter)
            print('learn rate = ' .. rmsprop_state.learningRate)
            print('momentum = ' .. rmsprop_state.momentum)
            print('decay = ' .. rmsprop_state.decay)
            printf('t = %.1fs\n', sys.clock() - start)
        end

        grads:zero()

        local len = math.random(opt.minLen, opt.maxLen)
        local taskName = tasks[math.random(#tasks)]
        local ioExample = dataGenerators[taskName]:getRandomDataPoint(len, 1)
        local output, criteria, loss = forward(model, ioExample, print_flag)
        local effectiveLen = ioExample.targetMask[{{}, 1}]:sum() -- how many rows are non-masked?
        local natsPerStep = loss / effectiveLen
        backward(model, ioExample, output, criteria)
 
        -- clip gradients
        grads:clamp(-10, 10)
        if print_flag then
            lossHistory[iter] = summedLossPrint / opt.print_interval
            summedLossPrint = 0
            print('max grad = ' .. grads:max())
            print('min grad = ' .. grads:min())
            print('nats per time step = ' .. natsPerStep)
            save_plots(taskName, ioExample, output, lossHistory, opt.save_file)
        end
        return natsPerStep, grads
    end

    local _params, loss = ntm.rmsprop(feval, params, rmsprop_state)
    summedLossSave = summedLossSave + loss[1]
    summedLossPrint = summedLossPrint + loss[1]
end
