function X = dftDataGrabber1(a,b,fileName)
	fileFormat = '%e';
	B = [];
	C = [];
	D = [' '];
	sizeC = [5,1]
	sizeB = [45 Inf]	
	for i = a:b 
		fileID = fopen(strcat('sf', num2str(i), '.dat'));
		C = [transpose(fscanf(fileID, fileFormat, sizeC))];%readfrom file, write to arrayB		
		B = [transpose(fscanf(fileID, fileFormat, sizeB))];%readfrom file, write to arrayB
		fclose(fileID);	
		dlmwrite(fileName, C,'-append');
		dlmwrite(fileName, B,'-append');
		dlmwrite(fileName, D,'-append');			
	end	
	X = B
	end
