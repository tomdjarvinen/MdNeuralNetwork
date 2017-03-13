--
-- Created by IntelliJ IDEA.
-- User: tom
-- Date: 3/12/17
-- Time: 3:08 PM
-- To change this template use File | Settings | File Templates.
--
require 'Net'
--Set output file
io.output("Tests/BigStrainTest3")
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
        parallelSGD(trainSet,netRepo,criterion,learningRate)
        k = k + 1
        learningRate = learningRate*anneallingRate
        Results = Results.."Run = "..k.." Training set error: "..getMeanErrorParallel(trainSet,netRepo).." Test Set error: "..getMeanErrorParallel(testSet,netRepo).."\n"
    end
    return Results
end
function denormalize(value, mean, stdev)
    return value * stdev + mean
end


--Initialize our data
dataSet = dftPreprocess('strainData.txt')

function dataSet:size()return 41 end
mean,stdev = meanStdev(dataSet)
dataSet = normalize(dataSet,mean,stdev)
trainSet,testSet = splitSet(dataSet,0.7)
twoDTrain = dataSetTo2dTensor(trainSet)
twoDTest = dataSetTo2dTensor(testSet)

bigStrain = dftPreprocess('bigStrain.txt')
bigStrain = normalize(bigStrain,mean,stdev) --Normalize with respect to same values as dataSet
twoDbigStrain = dataSetTo2dTensor(trainSet)

--Make some nets
net1 = makeNet(40,2,50)
net2 = makeNet(40,3,50)
net3 = makeNet(40,4,50)
repo1 = makeRepo(net1,8)
repo2 = makeRepo(net2,8)
repo3 = makeRepo(net3,8)
learningRate = 0.01
annealingRate = 0.6
io.write("Initial test using big strain dataSet\n")
io.write("Testing three neural networks trained exclusively on strain dataSet with 70% being used as data, and 30% for validation.\nTrain the networks: \n")
--Train the Nets
io.write(testArch(twoDTrain,twoDTest,nn.MSECriterion(),learningRate,annealingRate,6,repo1))
io.write(testArch(twoDTrain,twoDTest,nn.MSECriterion(),learningRate,annealingRate,6,repo2))
io.write(testArch(twoDTrain,twoDTest,nn.MSECriterion(),learningRate,annealingRate,6,repo2))

io.write("\nNow test the networks.\n Net one (50-50-1):\n")
io.write("Strain test set average error (denormalized): ", getMeanErrorDenormalized(twoDTest,repo1,mean[1],stdev[1]))
io.write("\nBig-Strain test set average error: ", getMeanErrorParallel(twoDbigStrain,repo1), " Denormalized: ", getMeanErrorDenormalized(twoDbigStrain,repo1,mean[1],stdev[1]))
io.write("\nSome forward passes: \n")
for i = 1,50 do
    io.write("Big strain[", (i*5), "] Target:Result ", twoDbigStrain[i*5]["output"][1],":", parallelForward(twoDbigStrain[i*5],repo1)[1], " Denormalized: ", denormalize(twoDbigStrain[i*5]["output"][1],mean[1],stdev[1]), ":", denormalize(parallelForward(twoDbigStrain[i*5],repo1)[1],mean[1],stdev[1]), "\n")
end
io.write("\nNow test the networks.\n Net two (50-50-50-1):\n")
io.write("Strain test set average error (denormalized): ", getMeanErrorDenormalized(twoDTest,repo2,mean[1],stdev[1]))
io.write("\nBig-Strain test set average error: ", getMeanErrorParallel(twoDbigStrain,repo2), " Denormalized: ", getMeanErrorDenormalized(twoDbigStrain,repo2,mean[1],stdev[1]))
io.write("\nSome forward passes: \n")
for i = 1,50 do
    io.write("Big strain[", i*5, "] Target:Result ", twoDbigStrain[i*5]["output"][1],":", parallelForward(twoDbigStrain[i*5],repo2)[1], " Denormalized: ", denormalize(twoDbigStrain[i*5]["output"][1],mean[1],stdev[1]), ":", denormalize(parallelForward(twoDbigStrain[i*5],repo2)[1],mean[1],stdev[1]), "\n")
end
io.write("\nNow test the networks.\n Net two (50-50-50-50-1):\n")
io.write("Strain test set average error (denormalized): ", getMeanErrorDenormalized(twoDTest,repo3,mean[1],stdev[1]))
io.write("\nBig-Strain test set average error: ", getMeanErrorParallel(twoDbigStrain,repo3), " Denormalized: ", getMeanErrorDenormalized(twoDbigStrain,repo3,mean[1],stdev[1]))
io.write("\nSome forward passes: \n")
for i = 1,50 do
    io.write("Big strain[", i*5, "] Target:Result ", twoDbigStrain[i*5]["output"][1],":", parallelForward(twoDbigStrain[i*5],repo3)[1], " Denormalized: ", denormalize(twoDbigStrain[i*5]["output"][1],mean[1],stdev[1]), ":", denormalize(parallelForward(twoDbigStrain[i*5],repo3)[1],mean[1],stdev[1]), "\n")
end