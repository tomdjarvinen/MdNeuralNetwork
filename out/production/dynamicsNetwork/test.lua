--
-- Created by IntelliJ IDEA.
-- User: tom
-- Date: 3/12/17
-- Time: 3:08 PM
-- To change this template use File | Settings | File Templates.
--
require 'Net.lua'




basicNet = makeNet(41, 10, 100)
local test = inference(basicNet,dataSet[1])-dataSet[1][1][1]
print(test)


mean,stdev = meanStdev(dataSet)
dataSet = normalize(dataSet,mean,stdev)
trainSet,testSet = splitSet(dataSet,0.7)
learningRate = .001
net1 = basicNet:clone()
net2 = basicNet:clone()
for i = 1, 10 do
    SGD(basicNet, trainSet,nn.MSECriterion(),learningRate)
    SGD(net1, trainSet,nn.MSECriterion(),learningRate/2)
    SGD(net2, trainSet,nn.MSECriterion(),learningRate/10)

    print('Basic Train Set error: ', getMeanError(basicNet,trainSet), 'Test Set error', getMeanError(basicNet,testSet))
    print('net1 Train Set error: ', getMeanError(basicNet,trainSet), 'Test Set error', getMeanError(basicNet,testSet))
    print('net3 Train Set error: ', getMeanError(basicNet,trainSet), 'Test Set error', getMeanError(basicNet,testSet))

end


