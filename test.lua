--
-- Created by IntelliJ IDEA.
-- User: tom
-- Date: 3/12/17
-- Time: 3:08 PM
-- To change this template use File | Settings | File Templates.
--
require 'Net'
--Set output file
io.output("Tests/AnneallingTest2")
--Initialize Networks

function makeRepo(base, numAtoms)
    local repo = {}
    for i = 1,numAtoms do --Makes 10 copies of the base Net, all with shared parameters (parameters are shared references, not just values)
        repo[i]=base:clone()   --Create network with same structure as our base
        repo[i]:share(base,'bias','weight') --Share BaseNet's parameters as references
    end
    local parallelNetTable = {} --Table containing neural networks indexed with the # of parallel atoms
    local dummyNumAtomsTable = {}
    dummyNumAtomsTable[14] = 0  --Atomic # of silicon is 14
    local dummyNNtable = {}           --Need to wrap NetRepository in a table so that parallelNet will work correctly
    dummyNNtable[14] = repo
    for i = 1,numAtoms do --
        dummyNumAtomsTable[14] = i
        parallelNetTable[i] = parallelNet(dummyNumAtomsTable,dummyNNtable)
    end
    return parallelNetTable
end
function testArch(trainSet,testSet,criterion,learningRate,anneallingRate,numIterations,netRepo)
    local k = 0
    local Results = "Network: "..tostring(netRepo[1]).."Criterion: "..tostring(criterion).."\nLearningRate: "..learningRate.."Annealling Rate: "..anneallingRate.."\n"
    for i = 1,numIterations do
        parallelSGD(trainSet,netRepo,criterion,learningRate)
        k = k + 1
        learningRate = learningRate*anneallingRate
        Results = Results.."Run = "..k.." Training set error: "..getMeanErrorParallel(trainSet,netRepo).." Test Set error: "..getMeanErrorParallel(testSet,netRepo).."\n"
    end
    return Results
end

--Initialize parallel networks for each # of atoms in our test set.
--This will allow us to pass our dataSet into the tables as references in order to rapidly train our network.


--Initialize our data
dataSet = dftPreprocess('strainData.txt')
function dataSet:size()return 41 end
mean,stdev = meanStdev(dataSet)
dataSet = normalize(dataSet,mean,stdev)
trainSet,testSet = splitSet(dataSet,0.7)
twoDTrain = dataSetTo2dTensor(trainSet)
twoDTest = dataSetTo2dTensor(testSet)

BaseNet = makeNet(40, 2, 50)
NetRepository = makeRepo(BaseNet, 10)
io.write("Data Set Used: Only strain data.  70% of data is allocated as a training set, 30% is left for validation.\n")
io.write("This run will test further test variations on learning rates.\n")
learningRate,anneallingRate = 0.01,1
learningRate = 0.01
local target, result
while learningRate > 0.001 do
    anneallingRate = 1
    for i = 1, 4 do
        io.write(testArch(twoDTrain,twoDTest,nn.MSECriterion(),learningRate,anneallingRate,20,NetRepository))
        io.write("\nSome forward passes through the network:\n")
        for i = 1,5 do
            target = twoDTest[i]["output"][1]
            result = parallelForward(twoDTest[i],NetRepository)[1]
            io.write("Target: ",target," Result: ", result, " Denormalized Target: ", (target*stdev[1]) + mean[1], " Denormalized Result: ", (result*stdev[1]) + mean[1])
        end

        anneallingRate = anneallingRate - 0.2
    end
    learningRate = learningRate - 0.001
end