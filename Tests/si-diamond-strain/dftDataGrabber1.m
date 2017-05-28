function X = dftDataGrabber1(fileName)
	fileFormat = '%e';
	B = [];
	C = [];
	D = [' '];
	sizeC = [5,1]
	sizeB = [45 Inf]
	for i = 0:9
		fileID = fopen(strcat(num2str(i), '-02-0',num2str(i), '.dat'));
		C = [transpose(fscanf(fileID, fileFormat, sizeC))];%readfrom file, write to arrayB
		B = [transpose(fscanf(fileID, fileFormat, sizeB))];%readfrom file, write to arrayB
		fclose(fileID);
		dlmwrite(fileName, C,'-append');
		dlmwrite(fileName, B,'-append');
		dlmwrite(fileName, D,'-append');
	end
	fileID = fopen('10-02-10.dat');
	C = [transpose(fscanf(fileID, fileFormat, sizeC))];%readfrom file, write to arrayB
	B = [transpose(fscanf(fileID, fileFormat, sizeB))];%readfrom file, write to arrayB
	fclose(fileID);
	dlmwrite(fileName, C,'-append');
	dlmwrite(fileName, B,'-append');
	dlmwrite(fileName, D,'-append');
	for i = 11:19
		j = i-10;
		fileID = fopen(strcat(num2str(i),'-02-m0',num2str(j), '.dat'));
		C = [transpose(fscanf(fileID, fileFormat, sizeC))];%readfrom file, write to arrayB
		B = [transpose(fscanf(fileID, fileFormat, sizeB))];%readfrom file, write to arrayB
		fclose(fileID);
		dlmwrite(fileName, C,'-append');
		dlmwrite(fileName, B,'-append');
		dlmwrite(fileName, D,'-append');
	end
	fileID = fopen('20-02-m10.dat');
	C = [transpose(fscanf(fileID, fileFormat, sizeC))];%readfrom file, write to arrayB
	B = [transpose(fscanf(fileID, fileFormat, sizeB))];%readfrom file, write to arrayB
	fclose(fileID);
	dlmwrite(fileName, C,'-append');
	dlmwrite(fileName, B,'-append');
	dlmwrite(fileName, D,'-append');
	fileID = fopen('21-06-01.dat');
	C = [transpose(fscanf(fileID, fileFormat, sizeC))];%readfrom file, write to arrayB
	B = [transpose(fscanf(fileID, fileFormat, sizeB))];%readfrom file, write to arrayB
	fclose(fileID);
	dlmwrite(fileName, C,'-append');
	dlmwrite(fileName, B,'-append');
	dlmwrite(fileName, D,'-append');
	fileID = fopen('22-06-010.dat');
	C = [transpose(fscanf(fileID, fileFormat, sizeC))];%readfrom file, write to arrayB
	B = [transpose(fscanf(fileID, fileFormat, sizeB))];%readfrom file, write to arrayB
	fclose(fileID);
	dlmwrite(fileName, C,'-append');
	dlmwrite(fileName, B,'-append');
	dlmwrite(fileName, D,'-append');
	for i = 23:30
		j = i-21;
		fileID = fopen(strcat(num2str(i),'-06-0',num2str(j), '.dat'));
		C = [transpose(fscanf(fileID, fileFormat, sizeC))];%readfrom file, write to arrayB
		B = [transpose(fscanf(fileID, fileFormat, sizeB))];%readfrom file, write to arrayB
		fclose(fileID);
		dlmwrite(fileName, C,'-append');
		dlmwrite(fileName, B,'-append');
		dlmwrite(fileName, D,'-append');
	end
	fileID = fopen('31-06-m01.dat');
	C = [transpose(fscanf(fileID, fileFormat, sizeC))];%readfrom file, write to arrayB
	B = [transpose(fscanf(fileID, fileFormat, sizeB))];%readfrom file, write to arrayB
	fclose(fileID);
	dlmwrite(fileName, C,'-append');
	dlmwrite(fileName, B,'-append');
	dlmwrite(fileName, D,'-append');
	fileID = fopen('32-06-m010.dat');
	C = [transpose(fscanf(fileID, fileFormat, sizeC))];%readfrom file, write to arrayB
	B = [transpose(fscanf(fileID, fileFormat, sizeB))];%readfrom file, write to arrayB
	fclose(fileID);
	dlmwrite(fileName, C,'-append');
	dlmwrite(fileName, B,'-append');
	dlmwrite(fileName, D,'-append');
	for i = 33:40
		j=i-31;
		fileID = fopen(strcat(num2str(i),'-06-m0',num2str(j), '.dat'));
		C = [transpose(fscanf(fileID, fileFormat, sizeC))];%readfrom file, write to arrayB
		B = [transpose(fscanf(fileID, fileFormat, sizeB))];%readfrom file, write to arrayB
		fclose(fileID);
		dlmwrite(fileName, C,'-append');
		dlmwrite(fileName, B,'-append');
		dlmwrite(fileName, D,'-append');
	end
	fileID = fopen('41-10-01.dat');
	C = [transpose(fscanf(fileID, fileFormat, sizeC))];%readfrom file, write to arrayB
	B = [transpose(fscanf(fileID, fileFormat, sizeB))];%readfrom file, write to arrayB
	fclose(fileID);
	dlmwrite(fileName, C,'-append');
	dlmwrite(fileName, B,'-append');
	dlmwrite(fileName, D,'-append');
	fileID = fopen('42-10-010.dat');
	C = [transpose(fscanf(fileID, fileFormat, sizeC))];%readfrom file, write to arrayB
	B = [transpose(fscanf(fileID, fileFormat, sizeB))];%readfrom file, write to arrayB
	fclose(fileID);
	dlmwrite(fileName, C,'-append');
	dlmwrite(fileName, B,'-append');
	dlmwrite(fileName, D,'-append');
	for i = 43:50
		j = i - 41;
		fileID = fopen(strcat(num2str(i),'-10-0',num2str(j), '.dat'));
		C = [transpose(fscanf(fileID, fileFormat, sizeC))];%readfrom file, write to arrayB
		B = [transpose(fscanf(fileID, fileFormat, sizeB))];%readfrom file, write to arrayB
		fclose(fileID);
		dlmwrite(fileName, C,'-append');
		dlmwrite(fileName, B,'-append');
		dlmwrite(fileName, D,'-append');
	end
	fileID = fopen('51-10-m01.dat');
	C = [transpose(fscanf(fileID, fileFormat, sizeC))];%readfrom file, write to arrayB
	B = [transpose(fscanf(fileID, fileFormat, sizeB))];%readfrom file, write to arrayB
	fclose(fileID);
	dlmwrite(fileName, C,'-append');
	dlmwrite(fileName, B,'-append');
	dlmwrite(fileName, D,'-append');
	fileID = fopen('52-10-m010.dat');
	C = [transpose(fscanf(fileID, fileFormat, sizeC))];%readfrom file, write to arrayB
	B = [transpose(fscanf(fileID, fileFormat, sizeB))];%readfrom file, write to arrayB
	fclose(fileID);
	dlmwrite(fileName, C,'-append');
	dlmwrite(fileName, B,'-append');
	dlmwrite(fileName, D,'-append');
	for i = 53:60
		j = i - 51
		fileID = fopen(strcat(num2str(i),'-10-m0',num2str(j), '.dat'));
		C = [transpose(fscanf(fileID, fileFormat, sizeC))];%readfrom file, write to arrayB
		B = [transpose(fscanf(fileID, fileFormat, sizeB))];%readfrom file, write to arrayB
		fclose(fileID);
		dlmwrite(fileName, C,'-append');
		dlmwrite(fileName, B,'-append');
		dlmwrite(fileName, D,'-append');
	end
	X = B
	end
