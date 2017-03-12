require 'torch';
require 'nn';
require 'math';
require 'dftPreprocess';
require 'nngraph';

--Initialize dataset
--TODO: Save dataSet sequentially
dataSet = dftPreprocess('strainData.txt')
function dataSet:size()return 41 end

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
function parallelNet(numAtoms,netPrimitive)
    net = nn.Sequential()
    m = nn.Parallel()
    netPrimitive:zeroGradParameters()
    local netTable = {}
    for i = 1, numAtoms do
        netTable[i] = netPrimitive:clone(netPrimitive)
        netTable[i]:share(netPrimitive,'bias','weight')
        m.add(netTable[i])
    end
    net:add(m)
    net:add(nn.Linear(numAtoms,1))
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
