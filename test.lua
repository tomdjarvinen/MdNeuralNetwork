--
-- Created by IntelliJ IDEA.
-- User: tom
-- Date: 3/12/17
-- Time: 3:08 PM
-- To change this template use File | Settings | File Templates.
--
require 'Net'
--Set output file
io.output("Tests/OverfitTest2")
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
    local Results = "Network: "..tostring(netRepo[1]).."Criterion: "..tostring(criterion).."\nLearningRate: "..learningRate.." Annealling Rate: "..anneallingRate.."\n"

    for i = 1,numIterations do
        print("hi")
        parallelSGD(trainSet,netRepo,criterion,learningRate)
        k = k + 1
        learningRate = learningRate*anneallingRate
        Results = Results.."Run = "..k.." Training set Percent error: %"..getMeanPercentError(trainSet,netRepo).." Test Set Percent error: %"..getMeanPercentError(testSet,netRepo).."\n"
    end
    return Results
end
function denormalize(value, mean, stdev)
    return value * stdev + mean
end

--Initialize our data
local dataSet = dftPreprocess('bigStrain.txt')
--local bigStrain = dftPreprocess('bigStrain.txt')
--If you import multiple datasets using dftPreprocess, they must be local variables otherwise new imports will overwrite the old value references

--dataSet = TableConcat(dataSet,bigStrain)
function dataSet:size()return 41 end
mean,stdev = meanStdev(dataSet)
dataSet = normalize(dataSet,mean,stdev)
trainSet,testSet = splitSet(dataSet,0.7)
twoDTrain = dataSetTo2dTensor(trainSet)
twoDTest = dataSetTo2dTensor(testSet)


--Make some nets
io.write("Overfit Test \n")
io.write("Testing networks of depth 3-13 with 50 training epochs at training rate .01, annealing rate 0.95\n")
io.write("Training will be done on Strain+Big Strain datasets.  I want to see what effect adding more data has.")
repo = {}
for i = 3, 13 do
    repo[i] = makeRepo(makeNet(40,i,50),8)
    io.write(testArch(twoDTrain,twoDTest,nn.MSECriterion(),.01,0.95,50,repo[i]))
end


