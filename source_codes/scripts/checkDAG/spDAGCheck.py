#!/usr/local/bin/python3

import sys
import os
import matplotlib.pyplot as plt



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
            elif '%' in line: #comments
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
    return rows, cols, triangleNNZ, rowIdxArr, colIdxArr, valsArr


def COO_to_CSR(rows, cols, NNZ, rowIdxArr, colIdxArr, valsArr):
    rowWiseElements = []
    
    # csr_vals = [0.0] * NNZ
    # csr_colIdx = [0] * NNZ
    # csr_rowIdx = [0] * (rows+1)
    csr_vals = []
    csr_colIdx = []
    csr_rowIdx = [0] * (rows+1)
    for i in range(rows):
        colIdxValArr = []
        rowWiseElements.append(colIdxValArr)

    for i in range(NNZ):
        colIdxValPair = [colIdxArr[i]-1,valsArr[i]]
        rowWiseElements[rowIdxArr[i]-1].append(colIdxValPair)

    # print(rowWiseElements[3][0][1])

    data_index = 0
    for i in range(rows):
        for j in range(len(rowWiseElements[i])):
            if(j==0):
                csr_colIdx.append(rowWiseElements[i][j][0])
                csr_vals.append(rowWiseElements[i][j][1])
                csr_rowIdx[i+1] = csr_rowIdx[i] + 1
                data_index+=1
            else:
                jj = j
                while(rowWiseElements[i][j][0]<csr_colIdx[data_index - ((j-jj+1))]) & (jj>0):
                    jj-=1
                csr_colIdx.insert((data_index - (j-jj)), rowWiseElements[i][j][0])
                csr_vals.insert((data_index - (j-jj)), rowWiseElements[i][j][1]) 
                csr_rowIdx[i+1] = csr_rowIdx[i+1] + 1
                data_index+=1

    return csr_vals, csr_colIdx, csr_rowIdx
    
def calculateDepth(rows, cols, NNZ, csr_vals, csr_colIdx, csr_rowIdx):
    rowDepths = [0] * rows
    for i in range(rows):
        rowMaxDependency = 0;
        for j in range(csr_rowIdx[i],csr_rowIdx[i+1]):
            if(i!=csr_colIdx[j]):
                if(rowDepths[csr_colIdx[j]]>rowMaxDependency):
                    rowMaxDependency = rowDepths[csr_colIdx[j]]
            else:
                rowMaxDependency += 1
        rowDepths[i] = rowMaxDependency
    return rowDepths

def buildDAG (rows, rowDepths):
    numLevels = max(rowDepths)

    levelSize = [0] * numLevels
    levels = []
    for i in range(numLevels):
        rowsInLevel = []
        levels.append(rowsInLevel)
    
    for i in range (rows):
        rowLevel = rowDepths[i]
        levelSize[rowLevel-1] += 1
        levels[rowLevel-1].append(i)
    
    return levelSize, levels

def drawDAGBarChart(levelSize):
    x_axis_data = []
    y_axis_data = []
    for i in range(len(levelSize)):
        x_axis_data.append(i+1)
        y_axis_data.append(levelSize[i])
    plt.bar(x_axis_data, y_axis_data)
    plt.title('Rows(Unknowns) per-level plots for SpTRSV DAGs')
    plt.xlabel('Level Number')
    plt.ylabel('Number of unknowns')
    plt.xlim(0,len(levelSize))
    plt.ylim(0,max(levelSize))
    plt.autoscale(tight=None)
    plt.show()


fileName = sys.argv[1]

#uncompress tar.gz file
uncompressFile(fileName)

#get matrix file name
inputFileName = getInputFileName(fileName)

#read the values to COO
print("===Starting reading values to COO===\n")
rows, cols, NNZ, rowIDX, colIDX, vals = readToCOO(inputFileName)
print("===Reading values to COO successful===")

#convert COO to CSR
print("\n===Starting converting COO to CSR===")
csr_vals, csr_colIdx, csr_rowIdx = COO_to_CSR(rows, cols, NNZ, rowIDX, colIDX, vals)
print("===Converting COO to CSR successful===")

#calculate depths indicating level
print("\n===Starting depth calculation===")
rowDepths = calculateDepth(rows, cols, NNZ, csr_vals, csr_colIdx, csr_rowIdx)
print("===Depth calculation successful===")

#build DAG. Returns number of rows in each level. 2D array containing rows in each level
print("\n===Starting DAG calculation===")
levelSize, levels = buildDAG (rows, rowDepths)
print("===DAG calculation successful===")

#draw levels vs number of unknown graph
print("\n===Starting drawing the graph===")
drawDAGBarChart(levelSize)
print("===Graph drawing successful===")

#clean directory
clearDirectory(fileName)

# print ("rows= %d," %rows + "cols=%d, " %cols + "NNZ=%d" %NNZ)
# print (rowIDX)
# print (colIDX)
# print (vals)

# print ("\n")
# print(csr_vals)
# print(csr_colIdx)
# print(csr_rowIdx)

# print ("\n")
# print(rowDepths)

# print("\n")
# print(levelSize)
# print(levels)

print ("\nRun Successful")
