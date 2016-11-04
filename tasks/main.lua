--[[

    Training a NTM to memorize input.

    The current version seems to work, giving good output after 5000 iterations
    or so. Proper initialization of the read/write weights seems to be crucial
    here.

--]]

require('../')
require('./util')
require('../datagen/CopyDataGenerator')
require('optim')
require('sys')

torch.manualSeed(0)

local cmd = torch.CmdLine()
cmd:text()
cmd:text('Options:')
cmd:option('--print_interval', 100, 'print progress every n time steps')
cmd:option('--save_interval', 1000, 'print progress every n time steps')
cmd:option('--savefile','ntm','filename to autosave the checkpont to. Will be inside checkpoint_dir/')
cmd:option('--init_from', '', 'initialize network parameters from checkpoint at this path')

local checkpoint_dir = 'checkpoints'
local opt = cmd:parse(arg or {})

-- NTM config
local config = {
    input_dim = 10,
    output_dim = 10,
    mem_rows = 128,
    mem_cols = 20,
    cont_dim = 100
}

local tasks = {'copy'}
local dataGenerators = {
    copy = CopyDataGenerator(config.input_dim - 2),
}

local function forward(model, ioExample)
    for i=1,ioExample.initialInput:size(1) do
        model:forward(ioExample.initialInput[i])
    end

    -- get output
    local output = torch.Tensor(ioExample.target:size())
    local criteria = {}
    local loss = 0
    for i=1, ioExample.queryInput:size(1) do
        criteria[i] = nn.BCECriterion()
        criteria[i].sizeAverage = false
        output[i] = model:forward(ioExample.queryInput[i])
        loss = loss + criteria[i]:forward(output[i], ioExample.queryInput[i])
    end
    return output:cmul(ioExample.targetMask), criteria, loss
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
local min_len = 1
local max_len = 20

print(string.rep('=', 80))
print("NTM copy task")
print('training up to ' .. num_iters .. ' iteration(s)')
print('min sequence length = ' .. min_len)
print('max sequence length = ' .. max_len)
print(string.rep('=', 80))
print('num params: ' .. params:size(1))

local rmsprop_state = {
    learningRate = 1e-4,
    momentum = 0.9,
    decay = 0.95
}

-- train
local summedLoss = 0
for iter = start_iter, num_iters do
    local print_flag = (iter % opt.print_interval == 0)
    local save_flag = (iter % opt.save_interval == 0)
    
    if save_flag then
        local avgLoss = summedLoss / opt.save_interval
        local savefile = string.format('%s/%s_iter%.2f_%.4f.t7', checkpoint_dir, opt.savefile, iter, avgLoss)
        print('saving model to ' .. savefile)
        torch.save(savefile, {
            model = model,
            iter = iter
        })
        summedLoss = 0
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

        local len = math.random(min_len, max_len)
        local taskName = tasks[math.random(#tasks)]
        local ioExample = dataGenerators[taskName]:getRandomDataPoint(len)
        local output, criteria, loss = forward(model, ioExample)

        backward(model, ioExample, output, criteria)
        if print_flag then
            print("target:")
            print(ioExample.target)
            print("output:")
            print(output)
            save_plots(taskName, ioExample, output)
        end

        -- clip gradients
        grads:clamp(-10, 10)
        if print_flag then
            print('max grad = ' .. grads:max())
            print('min grad = ' .. grads:min())
            print('nats per time step = ' .. loss / len)
        end
        return loss / len, grads
    end

    local _params, loss = ntm.rmsprop(feval, params, rmsprop_state)
    summedLoss = summedLoss + loss[1]
end
