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
--Initialize parallel networks for each # of atoms in our test set.
--This will allow us to pass our dataSet into the tables as references in order to rapidly train our network.
parallelNetTable = {} --Table containing neural networks indexed with the # of parallel atoms
dummyNumAtomsTable = {}
dummyNumAtomsTable[14] = 0  --Atomic # of silicon is 14
dummyNNtable = {}           --Need to wrap NetRepository in a table so that parallelNet will work correctly
dummyNNtable[14] = NetRepository
for i = 1,10 do --
    dummyNumAtomsTable[14] = i
    parallelNetTable[i] = parallelNet(dummyNumAtomsTable,dummyNNtable)
end
dataSet = dftPreprocess('strainData.txt')
function dataSet:size()return 41 end
mean,stdev = meanStdev(dataSet)
testDelete = removeSymmetryFromSet(dataSet, 41 )
--print("Original", dataSet[1][2][10], dataSet[1][2][11])
--print(testDelete[3])
dataSet = normalize(dataSet,mean,stdev)
trainSet,testSet = splitSet(dataSet,0.7)
twoDTest = dataSetTo2dTensor(trainSet)






--TEST CODE FOR PARALLELNET METHOD: SO FAR IT SEEMS TO BE WORKING CORRECTLY
--TestNet2 = makeNet(41,3,50)
--NetRepository2 = {}
--for i = 1,10 do --Makes 10 copies of the base Net, all with shared parameters (parameters are shared references, not just values)
--    NetRepository2[i]=TestNet2:clone()   --Create network with same structure as our base
--    NetRepository2[i]:share(TestNet2,'bias','weight') --Share BaseNet's parameters as references
--end


--singleAtomRepo = {}
--singleAtomRepo[1] = NetRepository
--singleAtomReference = {}
--singleAtomReference[1] = 2
--singleNetTest = parallelNet(singleAtomReference, singleAtomRepo)
--print(singleNetTest:forward(torch.randn(41,2)))
--print(singleNetTest:forward(tableTest))

--net:add(nn.Linear(numNets,1))
--net:zeroGradParameters()




--local testInput = torch.randn(41,2)
--print(parallelTest:forward(testInput))
--basicNet = makeNet(41, 10, 100)
--local test = inference(basicNet,dataSet[1])-dataSet[1][1][1]
--print(test)

--TESTING CODE FOR SGD METHOD: THIS HAD WEIRD BEHAVIOR, I DON'T THINK MY TRAINING IS CORRECT
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


