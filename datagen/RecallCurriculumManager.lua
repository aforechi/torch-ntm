if not AbstractCurriculumManager then
    require("./CurriculumManager.lua")
end

local RecallCurriculumManager, parent = torch.class('RecallCurriculumManager', 'AbstractCurriculumManager')

function RecallCurriculumManager:__init(min_len, max_len, 
                                        min_key, max_key)
    parent.__init(1.0, 5)
    self.minLen     = min_len
    self.maxLen     = max_len
    self.minKeyLen  = min_key
    self.maxKeyLen  = max_key
    self.minQueries = min_queries
    self.maxQueries = max_queries

    self.len        = self.minLen
    self.keyLen     = self.minKeyLen
    self.numQueries = self.minQueries
end

function RecallCurriculumManager:increaseDifficulty(loss)
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

function RecallCurriculumManager:_increaseDifficulty()

end
