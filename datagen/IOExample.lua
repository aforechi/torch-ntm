local IOExample = torch.class('IOExample')

function IOExample:__init(initialInput, queryInput, target, targetMask)
    self.initialInput = initialInput
    self.queryInput = queryInput
    self.target = target
    self.targetMask = targetMask
end
