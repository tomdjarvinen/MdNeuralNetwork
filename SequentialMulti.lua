--
--SequentialMulti.lua
--Author: Thomas Jarvinen
--This is a simple child of the nn.Sequential class, to provide an override of the updateOutput method.
--Instead of computing output over a single input, it sums the output over all inputs in the given table.
--Open Question: Do I need to override other methods?
--
require 'torch';
require 'nn';

local SequentialMulti, _ = torch.class('nn.SequentialMulti', 'nn.Sequential')

function SequentialMulti:updateOutput(input)

    local j, currentInput = next(input, nil)
    local finalOutput
    while j do
        local currentOutput = currentInput
        for i=1,#self.modules do
            currentOutput = self:rethrowErrors(self.modules[i], i, 'updateOutput', currentOutput)
        end
        finalOutput = finalOutput + currentOutput
    end
    self.output = finalOutput
    return finalOutput

end