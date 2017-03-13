require 'torch';
require 'nn';
require 'math';
require 'dftPreprocess';
require 'nngraph';

--Initialize dataset
--TODO: Save dataSet sequentially
--dataSet = dftPreprocess('strainData.txt')
--function dataSet:size()return 41 end

function inference(net, system)
    local i, atom = next(system, 1) --Skip first element, contains target value
    local accumulatedInference = torch.Tensor(1)
    while i do
        accumulatedInference = accumulatedInference + net:forward(atom)
        i, atom = next(system, i)
    end
    return accumulatedInference
end
--TODO: generalize to allow for multiple atom types
function SGD(net, trainSet, criterion, learningRate)
    net:zeroGradParameters()
    local i, system = next(trainSet, nil)
    while i do
        local sum = inference(net, system)
        local j,target = next(system,nil)
        local error = target - sum
        local j, atom = next(system,j)
        while j do

            local val = net:forward(atom)
            criterion:forward(val,val+error)
            net:backward(atom,criterion:backward(val, (val + error)))
            j,atom = next(system,j)
        end
        net:updateParameters(learningRate)
        net:zeroGradParameters()
        i, system = next(trainSet,i)
    end
end
--TODO: implement
function shuffleBatch(dataSet, batchSize)

end
function makeNet(numSymmetry, numHidden, numNodes)
    net = nn.Sequential();
    net:add(nn.Linear(numSymmetry,numNodes))
    net:add(nn.Sigmoid())
    for i=2,(numHidden-1) do
        net:add(nn.Linear(numNodes,numNodes))
        net:add(nn.Sigmoid())
    end
    net:add(nn.Linear(numNodes,1))
    net:add(nn.Sigmoid())
    return net
end


function getMeanError(net,Set)
    local count, error = 0, 0
    local i, node = next(Set, nil)
    while i do
        error = error + math.abs(inference(net,node)[1]-node[1][1])
        i, node = next(Set, i)
        count = count +1
    end
    return (error)/count
end

--parallelNet: Returns a neural network which has a set of parallel inputs for each atom in a system
--Inputs:
--  numTypes: table indexed with atomic # of each type of atom being evaluated
--      Each index contains the number of nets of that type which are needed
--  nets: table of tables:
--      Outer table: indexed with atom types from numTypes.  These indexes reference a table of neural network objects
--          These objects should be cloned networks with shared parameters indexed to the natural numbers.
--Outputs:  Returns a network with SUM(numTypes[i]*numSymmetry[i]) inputs, and one output, where i is all the indexes to the numTypes table
--Inputs to the net should be indexed as a 2-d tensor with size [numSymmetry][numAtoms]
--      TODO: This may need to change, because it does not allow different atom types to have a different # of inputs (unless the Tensor object can support this)
--      TODO: It is possible that this can be implemented more efficiently as a gated expert network; research how they are implemented
function parallelNet(numTypes, nets)
    local net = nn.Sequential()
    local c = nn.Parallel(2,1)    --Takes in parallel inputs indexed along the second dimension of input tensor (Q: can it take a table?)
    local i, numAtom = next(numTypes, nil)
    local atomCount = 0 --Need to track the total number of atoms in order to sum their outputs
    while i do
        for j = 1, numAtom do
            c:add(nets[i][j])
            atomCount = atomCount + 1
        end
        i, numAtom = next(numTypes, i)
    end
    net:add(c)
    net:add(nn.Linear(atomCount,1)) --Output is a weighted sum (atomCount)atoms, each having 1 output
    return net
end
