-- dftPreprocess.lua
-- Author: Thomas Jarvinen
-- Purpose: This script contains a set of helper functions that format our data suitably for training and inference by the objects of Net.lua
--

require 'torch';
require 'math';
--Parses a String of values seperated by sep, and formats it as a table where each value is an index
function ParseCSVLine (line,sep)
    local res = {}
    local pos = 1
    sep = sep or ','
    while true do
        local c = string.sub(line,pos,pos)
        if (c == "") then break end
        if (c == '"') then
            -- quoted value (ignore separator within)
            local txt = ""
            repeat
                local startp,endp = string.find(line,'^%b""',pos)
                txt = txt..string.sub(line,startp+1,endp-1)
                pos = endp + 1
                c = string.sub(line,pos,pos)
                if (c == '"') then txt = txt..'"' end
            -- check first char AFTER quoted string, if it is another
            -- quoted string without separator, then append it
            -- this is the way to "escape" the quote char in a quote. example:
            --   value1,"blub""blip""boing",value3  will result in blub"blip"boing  for the middle
            until (c ~= '"')
            table.insert(res,txt)
            assert(c == sep or c == "")
            pos = pos + 1
        else
            -- no quotes used, just look for the first separator
            local startp,endp = string.find(line,sep,pos)
            if (startp) then
                table.insert(res,string.sub(line,pos,startp-1))
                pos = endp + 1
            else
                -- no separator found -> use rest of string and terminate
                table.insert(res,string.sub(line,pos))
                break
            end
        end
    end
    return res
end
--TODO: actually implement getSize function
function getSize(symmetryDefinition)
    return 41
end
--Parses a single CSV file into a table where each index contains one line of data in from the file
--this line of data is stored as a table, where each comma seperated value is an index
--Output: each index points to a table of tables of format {result, atom1, atom2,...,atomN}
function ImportDFTData(filepath)
    io.input(filepath)
    dataTable = {};
    local line = io.read("*line")
    numInputs = getSize(line)
    local j = 1
    while line do
        local i = 1
        local table = {}
        table[i] = ParseCSVLine(line,",")[1]
        line = io.read("*line")
        while line ~= " "  do
            i=i+1
            table[i] = ParseCSVLine(line,",")
            line = io.read("*line")
        end
        dataTable[j]= table
        j = j+1
        line = io.read("*line")
    end
    return dataTable, numInputs
end
--Formats a dataTable from a CSV file into a table of format acceptable to our neural net.
--Input: {{result},{atom1},...,{atomN}}
--Output: {1=output=Torch.tensor(1), 2=Torch.tensor(# of inputs to net),...,i=Torch.tensor(# of inputs to net))
--Returns a table containing inputs and outputs from a single run of DFT data
function DataToTensor(dataTable, numInputs)
    dataSet = {}
    local energy = torch.Tensor(1)
    local i
    local tempTable
    i, tempTable = next(dataTable,nil) --grabs first index in datatable, the result table
    energy[1] = tempTable --Store energy
    dataSet[i]=energy
    i,tempTable=next(dataTable,i)           --Grab an atom
    while i do  --while dataTable has next, format it into tensors
        local inputSet = torch.Tensor(numInputs)
        for j=2,(numInputs+1) do    --Format it into a Tensor.  Note we are skipping indices 1, 43,44,45, as they are not relevant to our net
            inputSet[j-1] = tempTable[j]
        end
        dataSet[i]= inputSet
        i,tempTable=next(dataTable,i)           --Grab another atom
    end
    return dataSet
end
function dftPreprocess(fileName)
    local rawData,numInputs = ImportDFTData(fileName)
    dataSetFinal = {}
    local i,tempTable=next(rawData,nil)
    while i do
        dataSetFinal[i] = DataToTensor(tempTable, numInputs)
        i, tempTable = next(rawData,i)
    end
    return dataSetFinal
end
--Returns mean and standard deviation for dataSet.
--Input: dataset. should be of the same format returned by dftPreprocess()
--Output: Two tables, mean and stdev.  [1] = target, [2]-[getSize] = symmettry functions
function meanStdev(Set)
    local mean = {}
    local sum = {}
    local sumS = {}
    local meanS = {}
    local stdev = {}
    --initialize table values
    for i =1, 41 do
        sum[i] = 0
        sumS[i]=0
    end
    local setSize = 0 --Tracks number of datapoints in input set
    --Outer loop parses through all units of data (e.g. each DFT run)
    local i, dataUnit = next(Set, nil)
    while i do --calculate means
        if(type(dataUnit) == 'table') then
            local k, atom = next(dataUnit, nil)
            sum [1] = sum[1]+ atom[1]
            sumS[1] = sumS[1] + atom[1]
        --Parse through all atoms in this unit
            k, atom = next(dataUnit,k)
            while k do
                --parse through all symmetry functions in this atom (skipping first index which is atom label)
                for j = 2, 40 do
                    sum[j] = sum[j] + atom[j]
                    sumS[j] = sum[j] + atom[j]^2
                end
                k, atom = next(dataUnit,k)
            end
            setSize = setSize + 1
            i, dataUnit = next(Set, i)
        else
            Set[i] = nil
            i = nil
        end
    end
    --sum, sumS, setSize have been calculated, use these to calculate mean, stdev
    for i =1, 41 do
        mean[i]=sum[i]/setSize
        meanS[i] = sumS[i]/setSize
        stdev[i] = math.sqrt(math.abs(meanS[i]-mean[i]^2))
    end
    return mean, stdev
end
--Normalize Set by subtracting the mean, then dividing by standard deviation
--      Input: DataSet with format matching the output of DataToTensor
--      mean: table containing mean for each value
--      stdev: table containing standard deviation of each value
--      Output: returns normalized dataSet
--Note: this method seems to be bugged.
function gaussianNormalize(Set, mean, stdev)
    local i, dataUnit = next(Set, nil)  --grabs a test set
    while i do --normalize
        if(type(dataUnit) == 'table') then
            local k, atom = next(dataUnit, nil)  --grabs input tensor
            atom[1] = atom[1]-mean[1]
            if(stdev[1]~=0)then
                atom[1] = atom[1]/stdev[1]
            end
            k, atom = next(dataUnit,k)  --grabs first atom
            while k do
                for j=2,41 do
                    atom[j] = atom[j]-mean[j]
                    if(mean[j]~= 0) then
                        atom[j] = atom[j]/stdev[j]
                    end
                end
                k, atom = next(dataUnit,k)
            end
        end
        i, dataUnit = next(Set, i)
    end
    return Set
end
--Normalize Set to the range of values [-1,1] using formula 12 from Behler paper
--  For each index in set, use formula: 2(Val-Valmax)/(Valmax-Valmin)-1
--Inputs:
--  set: table of values with format: {1=output=Torch.tensor(1), 2=Torch.tensor(# of inputs to net),...,i=Torch.tensor(# of inputs to net))
--  max: a table containing maximimum values for each index of the input set (skip input[1], which is always atomic #
--  min: same as max, except minimum values
--Output:
--  set: Same as input except with values squeezed to [-1,1]
function hardNormalization(set,max,min)
    for i = 1, #set do
        local j, out = next(set[i], nil) --Puts output of this datapoint in out.  This is of type Tensor[1]
        if(max[j]-min[j]~=0)then
            set[i][j][1] = (2*(out[1]-min[j])/(max[j]-min[j]))-1
        end
        local input
        j, input = next(set[i], j) --Grabs first input of this datapoint. This should be of format Tensor[#of inputs + 1]
        while j do
            for k = 2, 41 do        --Skip first value, it is just the atomic #
                if(max[k]-min[k]~=0)then
                    set[i][j][k] = (2*(input[k]-min[k])/(max[k]-min[k]))-1
                end

            end
            j,input = next(set[i], j)
        end
    end
    return set
end
function logOfSet(set)
    for i = 1, #set do
        local j, out = next(set[i], nil) --Puts output of this datapoint in out.  This is of type Tensor[1]
        local input
        j, input = next(set[i], j) --Grabs first input of this datapoint. This should be of format Tensor[#of inputs + 1]
        while j do
            for k = 2, 41 do        --Skip first value, it is just the atomic #
                if(set[i][j][k] ~=0)then
                set[i][j][k] = math.log(set[i][j][k])
                end

            end
            j,input = next(set[i], j)
        end
    end
    return set
end
--Performs a linear search for maximum and minimum values of each index of Set, and returns a table containing these values.
--Inputs:
--  set: a set of values of the format returned by dftPreprocess
--      set: table of values with format: {1=output=Torch.tensor(1), 2=Torch.tensor(# of inputs to net),...,i=Torch.tensor(# of inputs to net))
-- Outputs:
--  max: a table containing maximimum values for each index of the input set (skip input[1], which is always atomic #
--  min: same as max, except minimum values
--TODO: Figure out how to do the dataSet:size method properly, then update the innermost for loop to use it.
function getMaxMin(set)
    local max = {}
    local min = {}
    for i = 1, #set do

       local j, out = next(set[i], nil) --Puts output of this datapoint in out.  This is of type Tensor[1]
        if(max[j] == nil)then
            max[j] = out[1]
        elseif(out[1] > max[j])then
            max[j] = out[1]
        end
        if(min[j] == nil)then
            min[j] = out[1]
        elseif(out[1] < min[j])then
            min[j] = out[1]
        end
        local input
        j, input = next(set[i], j) --Grabs first input of this datapoint. This should be of format Tensor[#of inputs + 1]

        while j do
            for k = 2, 41 do
                if(max[k] == nil)then
                    max[k] = input[k]
                elseif(input[k] > max[k])then
                    max[k] = input[k]
                end
                if(min[k] == nil)then
                    min[k] = input[k]
                elseif(input[k] < min[k])then
                    min[k] = input[k]
                    if(k==41)then
                    end
                end
            end
            j,input = next(set[i], j)
        end
    end
    return max,min
end
--splits a set into two seperate sets at random
--The ratio of one set to another is defined by ratio
--Input: ratio, # between 0 and one, which defines what proportion of the return set
--will belong to the trainSet
function splitSet(Set, ratio)
    local testSet, trainSet = {}, {}
    local i, unit = next(Set, nil)
    local testCount, trainCount=1,1
    while i do
        if(torch.uniform()>ratio) then
                testSet[testCount] = unit
                testCount = testCount +1
        else
            trainSet[trainCount] = unit
            trainCount  = trainCount + 1
        end

        i, unit = next(Set, i)
    end
    return trainSet,testSet
end
--dataSetTo2dTensor: Parses through a dataSet, so it can be reformatted by the dataPointTo2dTensor method
--Input: dataSet: table of tables formatted as dataPoints (see dataPointTo2dTensor for details)
--Output: dataSet2d: table of tables formatted as data2D  (see dataPointTo2dTensor for details)
function dataSetTo2dTensor(dataSet)
    local i, testCase = next(dataSet,nil)
    local dataSet2D = {}
    while i do
        dataSet2D[i] = dataPointTo2dTensor(testCase)
        i, testCase = next(dataSet,i)
    end
    return dataSet2D
end
--dataPointTo2dTensor: reformats dataSet data structure (as output by dftPreprocess) into a format suitable for parallelTable
--Inputs:
--  dataPoint: dft test that has been loaded into a Lua table
--      dataPoint has format: {1=output=Torch.tensor(1), 2=Torch.tensor(# of inputs to net),...,i=Torch.tensor(# of inputs to net))
--Output: returns data2d, a restructuring of the input dataPoint. It is a lua table. Data2d has the following indices:
--      Index: numAtoms     Contains:Lua table w/ # of each type of atom in the system, Indexed by atomic # of the atoms
--          E.g. a test case with 10 Si atoms and 4 Sulfur atoms would have format numAtoms[14] = 10, numAtoms[16] = 4
--      Index: input        Contains: Tensor[numSymmetry][SUM(all values in numAtoms].  This object is ordered by the numAtoms index.
--          E.g. for the test case above input[*][1]-input[*][10] will contain Si data, and input[*][11]-input[*][14] will contain S data
--      Index: output       Contains: Tensor[1] object containing energy of system (target output of NN).
function dataPointTo2dTensor(dataPoint)
    local numAtoms = {}
    local symmetry
    local i, output = next(dataPoint,nil) --Get first value in dataPoint: output
    local input
    i,input = next(dataPoint,i)           --Get first input
    while i do  --parse through all inputs
        if(numAtoms[input[1]] == nil) then  --First Element of the input is the atomic #, if We don't yet have this atom type, then add an index for it.
            numAtoms[input[1]] = 1
        else                                --If we already have this type, increment our count
            numAtoms[input[1]] = numAtoms[input[1]] + 1
        end
        if(symmetry == nil) then            --If it doesn't exist yet, initialize it with the input
            symmetry = removeFirstElement(input)
        else                                --Else, concatenate current input
            symmetry = torch.cat(symmetry,removeFirstElement(input),2)      --Want formated as Tensor of size [numSymmettry][numAtoms], so concat in second dimension
        end
        i, input = next(dataPoint,i)        --Move to next element
    end
    local data2d = {}
    data2d["numAtoms"] = numAtoms
    data2d["input"] = symmetry
    data2d["output"] = output
    return data2d
end
--Returns the input Tensor with its first element removed
function removeFirstElement(tensor)
    return tensor:narrow(1,2,tensor:size()[1]-1)
end
--dataSetOrderTest: Validates assumption that input data to dataPointTo2dTensor will be ordered by atom type
--  Input:  dataPoint (see dataPointTo2dTensor for details)
--  Output: return 1 if everything is correct, return 0 if not
--  TODO: Implement this method (deferring implementation until we get to two-atom systems)
function dataSetOrderTest(dataPoint)
end
--removeSymmetryFunction: returns a dataSet with the row at index removed
--Inputs:
--  dataSet: table of values with format: {1=output=Torch.tensor(1), 2=Torch.tensor(# of inputs to net),...,i=Torch.tensor(# of inputs to net))
function removeSymmetryFromSet(dataSet, index)
    local updatedSet = {}
    local i, testRun = next(dataSet,nil)
    while i do
        updatedSet[i]= removeSymmetry(testRun,index)
        i, testRun = next(dataSet,i)
    end
    return updatedSet
end
--removeSymmetry: returns the input data point with the symetry function at index removed
--If the index is invalid, it returns the original datapoint, a -1, and an error message
--If the index is valid, it returns the updated point and a 1
--Input: dataPoint (2dTensor form)
function removeSymmetry(dataPoint,index)
    local updatedPoint = {}
    local i, temp = next(dataPoint, nil)
    updatedPoint[i] = temp
    i, temp = next(dataPoint,i)
    local first, second
    while i do
        if index==1 then
            updatedPoint[i] = removeFirstElement(temp)
        elseif index == temp:size()[1] then
            updatedPoint[i] = temp:narrow(1,1,index-1)
        elseif index > temp:size()[1] then
            return dataPoint, -1, "Index too large"
        elseif index < 0 then
            return dataPoint, -1, "Negative Index"
        else
            first = temp:narrow(1,1,index-1)
            second = temp:narrow(1,index+1,temp:size()[1]-index)
            updatedPoint[i] = torch.cat(first,second,1)
        end
        i, temp = next(dataPoint,i)
    end
    return updatedPoint,1,"success"
end
function TableConcat(t1,t2)
    for i=1,#t2 do
        t1[#t1+1] = t2[i]
    end
    return t1
end
function extractCluster(minTarget,maxTarget,dataSet)
    local i, datapoint = next(dataSet,nil)
    local index = 1
    local cluster = {}
    while i do
        if(datapoint[1][1] > minTarget and datapoint[1][1] < maxTarget) then
            cluster[index] = datapoint
            index = index + 1
        end
        i, datapoint = next(dataSet,i)
    end
    return cluster
end
function findOutliers(minTarget,maxTarget,dataSet)
    local i, datapoint = next(dataSet,nil)
    local index = 1
    local isOutlier;
    local cluster = {}
    while i do  --Go through all test cases
        isOutlier = 0;
        local j, atom = next(datapoint,nil) --should grab output
        j, atom = next(datapoint,j)         --should grab first atom
        while j do  --Go through all atoms in test case
            for k = 1,40 do --Go throught all symettry functions in atom
                if(atom[k] > minTarget and atom[k] < maxTarget) then
                    isOutlier = 1;
                end
            end
            j, atom = next(datapoint,j)
        end
        if isOutlier == 1 then
            cluster[index] = i
            index = index + 1
        end
        i, datapoint = next(dataSet,i)
    end
    return cluster
end
function findMaxIndex(dataSet2D)
    local max= {}
    for i = 1, #dataSet2D do
        for k = 1, dataSet2D[1]["input"]:size()[2] do
            for j = 1, dataSet2D[1]["input"]:size()[1] do
                if max[j]==nil then
                    local maxTable = {}
                    maxTable["index"] = 0
                    maxTable["value"] = -1
                    max[j] = maxTable
                end
                if max[j]["value"] < dataSet2D[i]["input"][j][k] then
                    max[j]["value"] = dataSet2D[i]["input"][j][k]
                    max[j]["index"] = i
                end
            end
        end
    end
    return max
end