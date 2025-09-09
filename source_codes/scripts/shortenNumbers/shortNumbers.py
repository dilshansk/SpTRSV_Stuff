#!/usr/bin/python

import sys
import os
import random



def uncompressFile(fileName):
    myCommand = "tar -xf " + fileName
    os.system(myCommand)

def getInputFileName(fileName):
    folderName = fileName.replace(".tar.gz","")
    inputFileName = folderName+"/"+folderName+".mtx"
    return inputFileName


def clearDirectory(fileName):
    folderName = fileName.replace(".tar.gz","")
    myCommand = "rm -rf " + folderName
    os.system(myCommand)


def readToCOO(inputFileName, outputFileName):
    file1 = open(inputFileName, 'r')
    file2 = open(outputFileName, 'w')

    readHader = 0

    random.seed(5)

    while True:

        # Get next line from file
        line = file1.readline()
    
        # if line is empty
        # end of file is reached
        if not line:
            break
        else:
            if '%' in line:
                file2.write(line)
            else:
                if(readHader):
                    vals = line.split(" ")
                    rowIdx = int(vals[0])
                    colIdx = int(vals[1])
                    value=float(vals[2])
                    reminder = value % 10.0
                    if reminder==0.0:
                        finalValue = random.uniform(-10.0,10.0)
                    else:
                        finalValue = reminder
                    finalValue = round(finalValue,5)
                    file2.write(str(rowIdx) + " " + str(colIdx) + " " + str(finalValue) + "\n")
                else:
                    file2.write(line)
                    readHader = 1
    
    file1.close()
    file2.close()
    



fileName = sys.argv[1]

#uncompress tar.gz file
uncompressFile(fileName)

#get matrix file name
inputFileName = getInputFileName(fileName)

#read the values to COO
outputFileName = "test.out"
print("===Starting to modify===\n")
readToCOO(inputFileName, outputFileName)
print("===Modifying successful===")

#clean directory
clearDirectory(fileName)

print ("\nRun Successful")
