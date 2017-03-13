--
-- Created by IntelliJ IDEA.
-- User: tom
-- Date: 3/12/17
-- Time: 3:08 PM
-- To change this template use File | Settings | File Templates.
--
require 'Net'
--Initialize Networks
BaseNet = makeNet(41, 2, 50)
NetRepository = {}
for i = 1,10 do --Makes 10 copies of the base Net, all with shared parameters (parameters are shared references, not just values)
    NetRepository[i]=BaseNet:clone()   --Create network with same structure as our base
    NetRepository[i]:share(BaseNet,'bias','weight') --Share BaseNet's parameters as references
end

TestNet2 = makeNet(41,3,50)
NetRepository2 = {}
for i = 1,10 do --Makes 10 copies of the base Net, all with shared parameters (parameters are shared references, not just values)
    NetRepository2[i]=TestNet2:clone()   --Create network with same structure as our base
    NetRepository2[i]:share(TestNet2,'bias','weight') --Share BaseNet's parameters as references
end


singleAtomRepo = {}
singleAtomRepo[1] = NetRepository
singleAtomReference = {}
singleAtomReference[1] = 2
singleNetTest = parallelNet(singleAtomReference, singleAtomRepo)
print(singleNetTest:forward(torch.randn(41,2)))
--print(singleNetTest:forward(tableTest))

--net:add(nn.Linear(numNets,1))
--net:zeroGradParameters()




--local testInput = torch.randn(41,2)
--print(parallelTest:forward(testInput))
--basicNet = makeNet(41, 10, 100)
--local test = inference(basicNet,dataSet[1])-dataSet[1][1][1]
--print(test)


--mean,stdev = meanStdev(dataSet)
--dataSet = normalize(dataSet,mean,stdev)
--trainSet,testSet = splitSet(dataSet,0.7)
--learningRate = .001
--net1 = basicNet:clone()
--net2 = basicNet:clone()
--for i = 1, 10 do
   -- SGD(basicNet, trainSet,nn.MSECriterion(),learningRate)
    --SGD(net1, trainSet,nn.MSECriterion(),learningRate/2)
    --   SGD(net2, trainSet,nn.MSECriterion(),learningRate/10)

--    print('Basic Train Set error: ', getMeanError(basicNet,trainSet), 'Test Set error', getMeanError(basicNet,testSet))
--    print('net1 Train Set error: ', getMeanError(basicNet,trainSet), 'Test Set error', getMeanError(basicNet,testSet))
--    print('net3 Train Set error: ', getMeanError(basicNet,trainSet), 'Test Set error', getMeanError(basicNet,testSet))
--
--end


