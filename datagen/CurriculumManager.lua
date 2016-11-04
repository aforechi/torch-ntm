local AbstractCurriculumManager = torch.class('AbstractCurriculumManager')

function AbstractCurriculumManager:__init(lowLossThreshold,
                                          consecLowLossThreshold)
    self.lowLossThreshold = lowLossThreshold -- natsPerStep
    self.consecLowLossThreshold = consecLowLossThreshold -- time steps
    self.consecLowLosses = 0
end

function AbstractCurriculumManager:increaseDifficulty(loss)
    if loss < self.lowLossThreshold then
        self.consecLowLosses = self.consecLowLosses + 1
    else
        self.consecLowLosses = 0
    end

    if self.consecLowLosses >= self.consecLowLossThreshold then
        self:_increaseDifficulty()
        self.consecLowLosses = 0
    end
end
