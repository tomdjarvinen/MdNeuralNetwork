--
-- Created by IntelliJ IDEA.
-- User: tom
-- Date: 3/12/17
-- Time: 3:08 PM
-- To change this template use File | Settings | File Templates.
--
require 'Net'
--Set output file
io.output("Tests/lolol.txt")
--Initialize Networks

function testArch(trainSet,testSet,criterion,learningRate,anneallingRate,numIterations,netRepo)
    local k = 0
    local Results = "Network: "..tostring(netRepo[1]).."Criterion: "..tostring(criterion).."\nLearningRate: "..learningRate.." Annealling Rate: "..anneallingRate.."\n"

    for i = 1,numIterations do
        parallelSGD(trainSet,netRepo,criterion,learningRate)
        k = k + 1
        learningRate = learningRate*anneallingRate
        Results = Results.."Run = "..k.." Training set Percent error: %"..getMeanPercentError(trainSet,netRepo).." Test Set Percent error: %"..getMeanPercentError(testSet,netRepo).."\n"
    end
    return Results
end
function testArchBatch(trainSet,testSet,criterion,learningRate,anneallingRate,numIterations,netRepo,batchPortion)
    local k = 0
    local Results = "Network: "..tostring(netRepo[1]).."Criterion: "..tostring(criterion).."\nLearningRate: "..learningRate.." Annealling Rate: "..anneallingRate.."\n"

    for i = 1,numIterations do
        local batch = splitSet(trainSet,batchPortion)
        parallelBGD(batch,netRepo,criterion,learningRate)
        k = k + 1
        learningRate = learningRate*anneallingRate
        Results = Results.."Run = "..k.." Training set Percent error: %"..getMeanPercentError(trainSet,netRepo).." Test Set Percent error: %"..getMeanPercentError(testSet,netRepo).."\n"
    end
    return Results
end
function gaussianDenormalize(value, mean, stdev)
    return value * stdev + mean
end
function hardDenormalize(value,min,max)
    return (((value+1)*(max-min))/2)+min
end


--Initialize our data
local dataSet = dftPreprocess('strainData.txt')
--local trimmedSet = extractCluster(-11,-8, dataSet)
--print(#trimmedSet)
--local bigStrain = dftPreprocess('bigStrain.txt')
--dataSet = TableConcat(dataSet,bigStrain)
--local max, min = getMaxMin(trimmedSet)
--dataSet = hardNormalization(trimmedSet,max,min)
--for i = 1, #dataSet do
--    local j, atom = next(dataSet[i],nil)
--    j, atom = next(dataSet[i], j)
--    while j do
--        print(atom[4],atom[5],atom[6],atom[7],atom[8])
--        j, atom = next(dataSet[i],j)
--    end
--end
--function dataSet:size()return 41 end



--If you import multiple datasets using dftPreprocess, they must be local variables otherwise new imports will overwrite the old value references

--mean,stdev = meanStdev(dataSet)
--dataSet = gaussianNormalize(dataSet,mean,stdev)

--local trainSet,testSet = splitSet(trimmedSet,0.7)
--local twoDTrain = dataSetTo2dTensor(trainSet)
--local twoDTest = dataSetTo2dTensor(testSet)
--for i=1, #trainSet do
--    print("Input: ", twoDTrain[i]["input"])
--end
--local test2d = dataSetTo2dTensor(dataSet)
--for i = 1, #max do
--    print(max[i]..min[i])
--end
--local set2 = dftPreprocess('strainData.txt')
--dataSet = logOfSet(dataSet)
local max, min = getMaxMin(dataSet)
dataSet =hardNormalization(dataSet,max,min)

--local test = findOutliers(-1,-.995,dataSet)
--print(#test)
--local shrunkenSet = {}
--for i = 1, #test do
--    shrunkenSet[i]=set2[test[i]]
--end
--shrunkenSet = hardNormalization(dataSet,max,min)

--max, min = getMaxMin(dataSet)
--dataSet =hardNormalization(dataSet,max,min)
local trainSet,testSet = splitSet(dataSet,0.7)
local twoDTrain = dataSetTo2dTensor(trainSet)
local twoDTest = dataSetTo2dTensor(testSet)
--
--local numSymmetry = 40
--for i = 1,2 do
    --   removeSymmetryFromSet(twoDTrain,5)
    --removeSymmetryFromSet(twoDTest,1)
--    numSymmetry=numSymmetry-1
--end
local repo =  makeRepo(makeNet(40,3,50),8,14)
    io.write(testArch(twoDTrain,twoDTest,nn.MSECriterion(),.01,0.6,10,repo))
    for i = 1, #twoDTest do
        print(hardDenormalize(twoDTest[i]["output"][1],max[1],min[1]),hardDenormalize(parallelForward(twoDTest[i],repo)[1],max[1],min[1]))
    end
--for i = 1,#twoDTrain do
--    print("Target: " , twoDTrain[i]["output"][1], "Actual: ", parallelForward(twoDTrain[i],repo)[1])
--end

--for i = 1,10 do
--    print("Target: " , hardDenormalize(twoDTest[i]["output"][1],min[1],max[1]), hardDenormalize(parallelForward(twoDTest[i],repo)[1],min[1],max[1]))
--end
--for i = 1,#twoDTrain do
--    print("Target: " , twoDTrain[i]["output"][1], "Actual: ", parallelForward(twoDTrain[i],repo)[1])
--end

--for j = 3, 3 do
--    for i=1, #twoDTest do
--        for k=1,(twoDTest[i]["input"]:size(2))do
--            print(twoDTest[i]["input"][j][k])
--        end
--    end
--end

