local AbstractDataGenerator = torch.class("AbstractDataGenerator")

function AbstractDataGenerator:__init(state)
    assert(false, 'abstract class does not define impl')
end

function AbstractDataGenerator:getRandomDataPoint()
    assert(false, 'abstract class does not define impl')
    return initialInput, queryInput, queryTarget, targetMask
end
