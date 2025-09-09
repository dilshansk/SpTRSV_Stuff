#!/usr/bin/python3
# Using readline()

import sys
import os

fileName = sys.argv[1]
myCommand = "tar -xf " + fileName
folderName = fileName.replace(".tar.gz","")
inputFileName = folderName+"/"+folderName+".mtx"
os.system(myCommand)

file1 = open(inputFileName, 'r')
count = 0

readHader = 0
diagCount = 0

rows = 0
cols = 0
NNZ = 0

foundPercetage = 0

status = 0;

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
                print("[NOTE]::Matrix is given in Array Format. Doesn't look like a sparse matrix...!")
            if((vals[3]=="complex") | (vals[3]=="pattern")):
                print("Values of this matrix has been given as ", val[3], " which we don't process at the moment.")
                status = 1;
                break;
        elif '%' in line: #comments
            continue;
        else:
            if(readHader):
                count += 1
                vals = line.split(" ")
                rowIdx = int(vals[0])
                colIdx = int(vals[1])
                if rowIdx==colIdx:
                    vals[2] = vals[2].replace('-',"")
                    value=float(vals[2])
                    if value!=0.0:
                        # print("row=", rowIdx, "col=", colIdx, "val=",value)
                        diagCount=diagCount+1
                    else:
                        print("For row index ", rowIdx, " a non zero value not found. Value = ", value)
                        break;
                # if(((count//NNZ)*100)%10==0):
                percentage = ((count/NNZ)*100)
                if percentage >= foundPercetage:
                    print("Analyzed %d" % percentage + "%")
                    foundPercetage+=10
            else:
                attr = line.split(" ")
                print("rows="+attr[0]+" cols="+attr[1]+" NNZ="+attr[2])
                rows = int(attr[0])
                cols = int(attr[1])
                if(rows!=cols):
                    print("Not a square matrix!")
                    break
                NNZ = int(attr[2])
                readHader=1

file1.close()

myCommand = "rm -rf " + folderName
os.system(myCommand)


if(status==1):
    print("\nMatrix checks Failed")
else:
    if diagCount==rows:
        print("\nThis is a traingular matrix")
    else:
        print("\nMissing diagonal values. This is not a traingular matrix.")