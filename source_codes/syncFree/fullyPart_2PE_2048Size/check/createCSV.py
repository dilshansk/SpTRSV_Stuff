#!/usr/bin/python3

import sys
import os
import csv

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
    

#Write traingle to csv
def outputToCSV(rows, cols, NNZ, csr_vals, csr_colIdx, csr_rowIdx):
    #Open output csv
    csv_file_name = "triangle_out.csv"
    run_shell_command("rm -rf " + csv_file_name)
    csv_f = open(csv_file_name, 'w', encoding='UTF8', newline='')
    writer = csv.writer(csv_f)

    for i in range(rows):
        csv_data = [None] * cols
        startIdx = csr_rowIdx[i]
        endIdx = csr_rowIdx[i+1]
        for j in range(startIdx, endIdx):
            csv_data[csr_colIdx[j]] = csr_vals[j]
        
        writer.writerow(csv_data)

    csv_f.close()


fileName = sys.argv[1]

if 'tar.gz' in fileName:
    #uncompress tar.gz file
    uncompressFile(fileName)

    #get matrix file name
    inputFileName = getInputFileName(fileName)
else:
    inputFileName = fileName

#read the values to COO
print("===Starting reading values to COO===\n")
rows, cols, NNZ, rowIDX, colIDX, vals = readToCOO(inputFileName)
print("===Reading values to COO successful===")

#convert COO to CSR
print("\n===Starting converting COO to CSR===")
csr_vals, csr_colIdx, csr_rowIdx = COO_to_CSR(rows, cols, NNZ, rowIDX, colIDX, vals)
print("===Converting COO to CSR successful===")

#Write to csv
outputToCSV(rows, cols, NNZ, csr_vals, csr_colIdx, csr_rowIdx)


if 'tar.gz' in fileName:
    #clean directory
    clearDirectory(fileName)

print ("\nRun Successful")
