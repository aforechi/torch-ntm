require('./DataGenerator.lua')
require('./IOExample.lua')

local CopyDataGenerator, parent = torch.class('CopyDataGenerator', 'AbstractDataGenerator')

local SPECIAL_BITS  = 2
local PADDING_STEPS = 2

function CopyDataGenerator:__init(bits)
    -- Generate sequences with specified input dimension
    self.bits           = bits or 6
    self.inputDim       = self.bits + SPECIAL_BITS
    self.outputDim      = self.inputDim -- makes everything conceptually cleaner

    self.startSymbol    = torch.zeros(self.inputDim)
    self.startSymbol[1] = 1
    self.endSymbol      = torch.zeros(self.inputDim)
    self.endSymbol[2]   = 1
end

function CopyDataGenerator:getRandomDataPoint(len)
    local seq = torch.zeros(len, self.inputDim)
    for i = 1, len do
        seq[{i, {SPECIAL_BITS + 1, self.inputDim}}] = torch.rand(self.inputDim - SPECIAL_BITS):round()
    end

    local initialInput           = torch.zeros(len + 2, self.inputDim)
    initialInput[1]              = self.startSymbol:clone()
    initialInput[{{2, len + 1}}] = seq:clone()
    initialInput[len + 2]        = self.endSymbol:clone()

    local targetMask             = torch.ones(seq:size())
    local queryTarget            = seq:clone():cmul(targetMask)
    local queryInput             = torch.zeros(seq:size())

    return IOExample(initialInput, queryInput, queryTarget, targetMask)
end

function CopyDataGenerator:__tostring()
    return 'CopyDataGenerator: # bits ' .. self.bits
end