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
function getSize(symettryDefinition)
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
    for i =1, Set:size()+1 do
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
                for j = 2, Set:size() do
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
function normalize(Set, mean, stdev)
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


