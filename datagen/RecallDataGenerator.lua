-- require('./DataGenerator.lua')
-- require('./IOExample.lua')
-- TODO: Package this

local RecallDataGenerator, parent = torch.class('RecallDataGenerator', 'AbstractDataGenerator')

local SPECIAL_BITS  = 2

function RecallDataGenerator:__init(bits, keyLength)
   -- Generate sequences with specified input dimension
   self.bits           = bits or 6
   self.keyLength      = keyLength or 3
   self.inputDim       = self.bits + SPECIAL_BITS
   self.outputDim      = self.inputDim

   self.delimSymbol    = torch.zeros(self.inputDim)
   self.delimSymbol[1] = 1
   self.querySymbol    = torch.zeros(self.inputDim)
   self.querySymbol[2] = 1
end

function RecallDataGenerator:getRandomDataPoint(numKeys, numQueries)
   local initialInput = torch.zeros((self.keyLength + 1) * numKeys, self.inputDim)
   local idx = 1
   for i = 1, numKeys do
      initialInput[idx] = self.delimSymbol
      idx = idx + 1
      for j = 1, self.keyLength do
         initialInput[{idx, {SPECIAL_BITS + 1, self.inputDim}}] = 
            torch.rand(self.inputDim - SPECIAL_BITS):round() 
         idx = idx + 1
      end
   end

   local queryInput = torch.zeros(numQueries * 2 * (self.keyLength + 1), self.inputDim)
   local target     = torch.zeros(numQueries * 2 * (self.keyLength + 1), self.outputDim)
   local targetMask = torch.zeros(numQueries * 2 * (self.keyLength + 1), self.outputDim)

   local idex = 1
   for i = 1, numQueries do
      local keyIdx   = math.random(numKeys - 1)
      local keyStart = (self.keyLength + 1) * (keyIdx - 1) + 2 -- first slot is delim
      local keyEnd   = keyStart + self.keyLength - 1
      local valStart = keyEnd + 2 -- one extra again for delim
      local valEnd   = valStart + self.keyLength - 1
      -- First half of query
      queryInput[idex] = self.querySymbol
      idex             = idex + 1
      queryInput[{{idex, idex + self.keyLength - 1}}] = initialInput[{{keyStart, keyEnd}}]
      idex                                            = idex + self.keyLength
      -- Second half of query
      queryInput[idex] = self.querySymbol
      idex             = idex + 1
      target[{{idex, idex + self.keyLength - 1}}]     = initialInput[{{valStart, valEnd}}]
      targetMask[{{idex, idex + self.keyLength - 1}}] = torch.ones(self.keyLength, self.outputDim)
      idex                                            = idex + self.keyLength
   end
   return IOExample(initialInput, queryInput, target, targetMask)
end

function RecallDataGenerator:__tostring()
   return 'RecallDataGenerator: # bits ' .. self.bits .. ' key length ' .. self.keyLength
end
