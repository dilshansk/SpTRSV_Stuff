#!/usr/bin/python3

import sys
import os

#run a command in the terminal
def run_shell_command(command):
    os.system("{cmd}".format(cmd=command))

def uncompressFile(fileName):
    myCommand = "tar -xf " + fileName
    os.system(myCommand)

def getInputFileName(fileName):
    folderName = fileName.replace(".tar.gz","")
    inputFileName = folderName+"/"+folderName+".mtx"
    return inputFileName

def printProgress(NNZ, count, foundPercetage):
    percentage = int(round(((float(count)/float(NNZ))*100.0),-1))
    if percentage >= foundPercetage:
        print("Analyzed %d" % percentage + "%")
        foundPercetage+=10
    return foundPercetage

def clearDirectory(fileName):
    folderName = fileName.replace(".tar.gz","")
    myCommand = "rm -rf " + folderName
    os.system(myCommand)


def readToCOO(inputFileName):
    file1 = open(inputFileName, 'r')
    
    count = 0

    readHader = 0
    diagCount = 0

    rows = 0
    cols = 0
    NNZ = 0
    triangleNNZ = 0
    sparsity = 0.0

    foundPercetage = 0

    #COO arrays
    rowIdxArr = [] 
    colIdxArr = []
    valsArr = []

    headerContent = []

    while True:

        # Get next line from file
        line = file1.readline()
    
        # if line is empty
        # end of file is reached
        if not line:
            break
        else:
            if '%%' in line: #first line
                vals = line.split(" ")
                if (vals[2]!="coordinate"):
                    print("[WARNING]::Matrix is given in Array Format. Doesn't look like a sparse matrix...!")
                if((vals[3]=="complex") | (vals[3]=="pattern")):
                    print("Values of this matrix has been given as ", val[3], " which we don't process at the moment.")
                    sys.exit()
                headerContent.append(line)
            elif '%' in line: #comments
                headerContent.append(line)
                continue;
            else:
                if(readHader):
                    count += 1
                    vals = line.split(" ")
                    rowIdx = int(vals[0])
                    colIdx = int(vals[1])
                    value=float(vals[2])
                    if (colIdx <= rowIdx):
                        if rowIdx==colIdx:
                            absVal = vals[2].replace('-',"")
                            if absVal!=0.0:
                                # print("row=", rowIdx, "col=", colIdx, "val=",value)
                                diagCount=diagCount+1
                            else:
                                print("[WARNING] For row index ", rowIdx, " a non zero value not found. Value = ", value)
                    
                        rowIdxArr.append(rowIdx)
                        colIdxArr.append(colIdx)
                        valsArr.append(value)
                        triangleNNZ+=1
                    
                    #print progress
                    foundPercetage = printProgress(NNZ, count, foundPercetage)

                else:
                    attr = line.split(" ")
                    rows = int(attr[0])
                    cols = int(attr[1])
                    NNZ = int(attr[2])
                    sparsity = float(NNZ)/(float(rows*cols))
                    print("=====Original Matrix Statistics=====")
                    print("rows="+str(rows)+" cols="+str(cols)+" NNZ="+str(NNZ)+" Sparsity="+str(sparsity))
                    print("====================================\n")
                    if(rows!=cols):
                        print("Not a square matrix!")
                        sys.exit()
                    readHader=1
    print("\n")
    if diagCount==rows:
        print("[INFO]::This is a traingular matrix")
    else:
        print("[WARNING]::Missing diagonal values. This is not a traingular matrix.")
    print("\n")
    file1.close()
    return rows, cols, triangleNNZ, rowIdxArr, colIdxArr, valsArr, headerContent


def extractDiagValInGivenRange(outputFileName, headerContent, rows, cols, NNZ, rowIdxArr, colIdxArr, valsArr, startRowLimit, endRowLimit):
    file1 = open(outputFileName, 'w')

    for line in headerContent:
        file1.write(line)
    
    #write row size, col size and nnz(=maxRowLimit in this case)
    matrixSize = endRowLimit - startRowLimit
    line = str(matrixSize) + " " + str(matrixSize) + " " + str(matrixSize) + "\n"
    file1.write(line)

    #loop through non zeros and write only if diagonal
    for i in range(NNZ):
        if ( (startRowLimit<=rowIdxArr[i]) & (rowIdxArr[i]<endRowLimit) & (startRowLimit<=colIdxArr[i]) & (colIdxArr[i]<endRowLimit) & (rowIdxArr[i] == colIdxArr[i]) ): #within range and diagonal value. need to write
            modifiedRowIdx = rowIdxArr[i] - (startRowLimit-1)
            modifiedColIdx = colIdxArr[i] - (startRowLimit-1)
            line = str(modifiedRowIdx) + " " + str(modifiedColIdx) + " " + str(valsArr[i]) + "\n"
            file1.write(line)

    file1.close()

#Given row range, this function dumps the mtx file for that triangle
def extractInGivenRowRange(outputFileName, headerContent, rows, cols, NNZ, rowIdxArr, colIdxArr, valsArr, startRowLimit, endRowLimit):
    file1 = open(outputFileName, 'w')

    for line in headerContent:
        file1.write(line)

    writeLines = []
    tileNNZ = 0

    #loop through non zeros and write only if diagonal
    for i in range(NNZ):
        if ( (startRowLimit<=rowIdxArr[i]) & (rowIdxArr[i]<endRowLimit) & (startRowLimit<=colIdxArr[i]) & (colIdxArr[i]<endRowLimit) ):  #within range. need to write
            modifiedRowIdx = rowIdxArr[i] - (startRowLimit-1)
            modifiedColIdx = colIdxArr[i] - (startRowLimit-1)
            line = str(modifiedRowIdx) + " " + str(modifiedColIdx) + " " + str(valsArr[i]) + "\n"
            writeLines.append(line)
            tileNNZ+=1

    #write row size, col size and nnz(=rows in this case)
    matrixSize = endRowLimit - startRowLimit
    line = str(matrixSize) + " " + str(matrixSize) + " " + str(tileNNZ) + "\n"
    file1.write(line)

    for i in range(tileNNZ):
        file1.write(writeLines[i])

    file1.close()
    

fileName = sys.argv[1]

if 'tar.gz' in fileName:
    #uncompress tar.gz file
    uncompressFile(fileName)

    #get matrix file name
    inputFileName = getInputFileName(fileName)
else:
    inputFileName = fileName

# diagonalOutputFileName = fileName.split(".")[0] + "_diagonal.mtx"
# manualRangeOutputFileName = fileName.split(".")[0] + "_manualRange.mtx"
diagonalOutputFileName = "diagonal.mtx"
manualRangeOutputFileName = "manualRange.mtx"
run_shell_command("rm -rf " + diagonalOutputFileName)
run_shell_command("rm -rf " + manualRangeOutputFileName)

#Matrix limits
startRowLimit = 1
maxRowLimit = 8192
endRowLimit = startRowLimit + maxRowLimit

#read the values to COO
print("===Starting reading values to COO===\n")
rows, cols, NNZ, rowIDX, colIDX, vals, headerContent = readToCOO(inputFileName)
print("===Reading values to COO successful===")

#Extract only diagonal values
print("\n===Starting extracting diagonal value only triangle===")
extractDiagValInGivenRange(diagonalOutputFileName, headerContent, rows, cols, NNZ, rowIDX, colIDX, vals, startRowLimit, endRowLimit)
print("===Extracting diagonal value only triangle successful===")

#Extract triangle in given row index range
print("\n===Starting extracting triangle in the given range===")
print("Start row=" + str(startRowLimit) + ", End row=" + str(endRowLimit))
extractInGivenRowRange(manualRangeOutputFileName, headerContent, rows, cols, NNZ, rowIDX, colIDX, vals, startRowLimit, endRowLimit)
print("===Extracting triangle in the given range successful===")

if 'tar.gz' in fileName:
    #clean directory
    clearDirectory(fileName)

print ("\nRun Successful")

