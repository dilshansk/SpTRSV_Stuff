#include "sptrsv.h"
#include <gflags/gflags.h>
#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <vector>
#include <math.h>
#include <sys/time.h>

#include <string>
#include <fstream>
#include <sstream>
#include <iomanip>

void sptrsv_kernel(tapa::mmap<const t_WIDE> value0,
                  tapa::mmap<const t_WIDE> value1,
                  tapa::mmap<const float> bval0,
                  tapa::mmap<const float> bval1,
                  tapa::mmap<float> xSolved, 
                  unsigned int nnz0, unsigned int nnz1, unsigned int size);

DEFINE_string(bitstream, "", "path to bitstream file, run csim if empty");


// using namespace std;

//read only triangular part as COO format
void readMatrixCOO(const std::string& filename, std::vector<float>& values, std::vector<int>& rowIdx, std::vector<int>& colIdx, int& rows, int& cols, int& nnz) {
    // open file
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Error: could not open file " << filename << std::endl;
        exit(0);
    }

    // file.precision(64);

    // read header
    std::string line;
    getline(file, line);
    while (line[0] == '%') {
        getline(file, line);
    }

    // parse header
    int orig_nnz=0;
    std::istringstream iss(line);
    iss >> rows >> cols >> orig_nnz;

    printf("Original Matrix Paramters :: rows=%d, columns=%d, nnz=%d\n", rows, cols, orig_nnz);

    // allocate memory
    // values.resize(nnz);
    // rowIdx.resize(nnz);
    // colIdx.resize(nnz);
    
    nnz = 0; //we only read triangle part. Therefore need to calculate nnz separately

    // read data
    int row, col;
    double value;
    for (int i = 0; i < orig_nnz; i++) {
        file >> row >> col  >> value;
        if (col<=row){
            values.push_back(value);
            rowIdx.push_back(row - 1);
            colIdx.push_back(col - 1);
            nnz++;
            // values[i] = value;
            // rowIdx[i] = row - 1;
            // colIdx[i] = col - 1;
        }
    }

    printf("Triangle Matrix Paramters :: rows=%d, columns=%d, nnz=%d\n", rows, cols, nnz);

    file.close();
}

//Convert triangle part in COO format to CSR
void COO_to_CSR(std::vector<float>& coo_values, std::vector<int>& coo_rowIdx, std::vector<int>& coo_colIdx, std::vector<float>& csr_values, std::vector<int>& csr_colIdx, std::vector<int>& csr_rowIdx,  int& rows, int& cols, int& nnz) {
    
    //allocate memory csr format
    // csr_values.resize(nnz);
    // csr_colIdx.resize(nnz);
    csr_rowIdx.resize(rows+1);

    //initialize row index
    for (int i = 0; i < rows+1; i++)
    {
        csr_rowIdx[i] = 0;
    }
    
    //First we get row wise data. Precautionary measure in case data is not ordered according to the row index or col index
    std::vector<std::vector<int>> rowWiseColIdx; //to hold row wise col indexes
    std::vector<std::vector<float>> rowWiseVals; //to hold row wise values
    rowWiseColIdx.resize(rows);
    rowWiseVals.resize(rows);

    for (int i = 0; i < nnz; i++)
    {
        rowWiseColIdx[coo_rowIdx[i]].push_back(coo_colIdx[i]);
        rowWiseVals[coo_rowIdx[i]].push_back(coo_values[i]);
    }

    // printf("second val=%f\n",rowWiseVals[2][0]);
    

    int data_index = 0;
    for (int i = 0; i < rows; i++)
    {
        for (int j = 0; j < rowWiseColIdx[i].size(); j++)
        {
            if(j==0){
                // printf("colIdx=%d, val=%f\n", rowWiseColIdx[i][j], rowWiseVals[i][j]);
                // printf("[before]::size of arr=%d\n", csr_values.size());
                csr_colIdx.push_back(rowWiseColIdx[i][j]);
                csr_values.push_back(rowWiseVals[i][j]);
                // printf("[before]::size of arr=%d\n", csr_values.size());
                csr_rowIdx[i+1] = csr_rowIdx[i] + 1;
                data_index++;
            }
            else{
                float jj=j;
                while ((rowWiseColIdx[i][j]<csr_colIdx[data_index-(j-jj+1)]) & (jj>0))
                {
                    jj--;
                }

                csr_colIdx.insert(csr_colIdx.begin() + (data_index - (j-jj)), rowWiseColIdx[i][j]);
                csr_values.insert(csr_values.begin() + (data_index - (j-jj)), rowWiseVals[i][j]);
                csr_rowIdx[i+1] = csr_rowIdx[i+1] + 1;
                data_index++;
            }
        }    
    }

}

/*This function create input stream s.t. it combine diagonal values whenever possible
    This was coded for 2 PE, 2 MAC unit design. Say previous row diagonal value is written to 0th index. Then 1st index has a space to carry another value value. What this does is, if the next row 
    is again a diagonal value(i.e., row size of 1), then it insert the new row diagonal value to that empty location.
    If the next row size is greater than 1, then it contains non diagonal value. It sends the non diagonal value in next location, if it's colum index matches with it(remind that mac unit need them in order).
    Otherwise, it will pad value '1' and add the non diag value to next location. Reason for padding '1' is to let that value consider as a non diagonal value and to be discarded as value is 0.
    At the moment this was developed, this started to give buggy results, because kernel code has a row synchronization method. Sending diagonal values parallely cause, solution status arriving late sometimes(for the 2nd value,
    so the synchronization doesn't work as expected. 
*/
void preProcessing_to_parallelDiagVal_Send(int& rows, int& cols, int& nnz, std::vector<float>& csr_values, std::vector<int>& csr_colIdx, std::vector<int>& csr_rowIdx, std::vector<float>& b_vec, std::vector<t_WIDE> *value_vec, std::vector<float> *b_vec_part, int *streamsize){

    std::vector<float> PEWiseVal[NUM_PE];
    std::vector<int> PEWiseRowIdx[NUM_PE];
    std::vector<int> PEWiseColIdx[NUM_PE];
    std::vector<int> PEWiseRowLength[NUM_PE];
    std::vector<t_DW> PEWiseCombinedVal[NUM_PE];

    int cummulativeLength[NUM_PE];
    int writtenLength[NUM_PE];

    float matVal_float;
    int matVal_int;
    t_HW colIdx;
    t_HW rowIdx;
    t_DW combinedVal;

    for (int i = 0; i < NUM_PE; i++)
    {
        writtenLength[i] = 0;
        cummulativeLength[i] = 0;
        streamsize[i] = 0;
    }
    
    for (int i = 0; i < rows; i++)
    {

        float b_val = b_vec[i];
        b_vec_part[i%NUM_PE].push_back(b_val);

        int startIdx = csr_rowIdx[i];
        int endIdx = csr_rowIdx[i+1];
        PEWiseRowLength[i%NUM_PE].push_back(endIdx-startIdx);

        for (int j = startIdx; j < endIdx; j++)
        {
            PEWiseVal[i%2].push_back(csr_values[j]);
            PEWiseColIdx[i%2].push_back(csr_colIdx[j]);
            PEWiseRowIdx[i%2].push_back(i);
            // printf("row=%d, col=%d, val=%f\n", i, csr_colIdx[j], csr_values[j]);
        }
    }

    for (int i = 0; i < NUM_PE; i++)
    {
        for (int j = 0; j < rows/NUM_PE; j++)
        {
            t_DW processingRowSize = PEWiseRowLength[i][j];
            if(processingRowSize==1){
                if ((writtenLength[i] % CONCAT_FACTOR) == 1) //If we are in an odd location, we can insert into a previous location. Hardcoded 1 should be modified for different concat factor
                {
                    int modifiedIdx = writtenLength[i] - 1;
                    matVal_float = PEWiseVal[i][cummulativeLength[i]];
                    matVal_int = (*(int *)(&matVal_float));
                    colIdx = PEWiseColIdx[i][cummulativeLength[i]];
                    rowIdx = PEWiseRowIdx[i][cummulativeLength[i]];
                    combinedVal = (((t_DW)matVal_int)<<32) | (colIdx<<16) | rowIdx;
                    PEWiseCombinedVal[i][modifiedIdx-1] = combinedVal;
                    PEWiseCombinedVal[i].push_back(processingRowSize);
                    writtenLength[i]+=1;
                }
                else{
                    matVal_float = PEWiseVal[i][cummulativeLength[i]];
                    matVal_int = (*(int *)(&matVal_float));
                    colIdx = PEWiseColIdx[i][cummulativeLength[i]];
                    rowIdx = PEWiseRowIdx[i][cummulativeLength[i]];
                    combinedVal = (((t_DW)matVal_int)<<32) | (colIdx<<16) | rowIdx;

                    PEWiseCombinedVal[i].push_back(combinedVal);
                    PEWiseCombinedVal[i].push_back(1);   //to allign size with diag val
                    PEWiseCombinedVal[i].push_back(processingRowSize);
                    writtenLength[i]+=3;
                }
            }
            else{
                for (int k = 0; k < processingRowSize; k++)
                {
                    int idx = cummulativeLength[i] + k;
                    if(PEWiseRowIdx[i][idx]==PEWiseColIdx[i][idx]){   //diag element
                        matVal_float = PEWiseVal[i][idx];
                        matVal_int = (*(int *)(&matVal_float));
                        colIdx = PEWiseColIdx[i][idx];
                        rowIdx = PEWiseRowIdx[i][idx];
                        combinedVal = (((t_DW)matVal_int)<<32) | (colIdx<<16) | rowIdx;

                        PEWiseCombinedVal[i].push_back(combinedVal);
                        PEWiseCombinedVal[i].push_back(1);
                        PEWiseCombinedVal[i].push_back(processingRowSize);
                        writtenLength[i]+=3;
                    }
                    else{
                        if((writtenLength[i]%2)==(PEWiseColIdx[i][idx]%2)){
                            matVal_float = PEWiseVal[i][idx];
                            matVal_int = (*(int *)(&matVal_float));
                            colIdx = PEWiseColIdx[i][idx];
                            rowIdx = PEWiseRowIdx[i][idx];
                            combinedVal = (((t_DW)matVal_int)<<32) | (colIdx<<16) | rowIdx;

                            PEWiseCombinedVal[i].push_back(combinedVal);
                            writtenLength[i]+=1;
                        }
                        else{
                            matVal_float = PEWiseVal[i][idx];
                            matVal_int = (*(int *)(&matVal_float));
                            colIdx = PEWiseColIdx[i][idx];
                            rowIdx = PEWiseRowIdx[i][idx];
                            combinedVal = (((t_DW)matVal_int)<<32) | (colIdx<<16) | rowIdx;

                            PEWiseCombinedVal[i].push_back(1);
                            PEWiseCombinedVal[i].push_back(combinedVal);
                            writtenLength[i]+=2;
                        }
                    }
                }
                
            }
            cummulativeLength[i] += (int)processingRowSize;
        }
        
    }

    for (int i = 0; i < NUM_PE; i++)
    {
        if (writtenLength[i]%CONCAT_FACTOR != 0)
        {
            for (int j = writtenLength[i]%CONCAT_FACTOR; j < CONCAT_FACTOR; j++)
            {
                PEWiseCombinedVal[i].push_back(1);
                writtenLength[i]+=1;
            }            
        }
    }

    for (int i = 0; i < NUM_PE; i++)
    {
        int length = writtenLength[i];
        for (int j = 0; j < length/NUM_PE; j++)
        {
            t_WIDE wideData;
            for (int k = 0; k < NUM_PE; k++)
            {
                wideData.range(DATA_SIZE*(k+1)-1, DATA_SIZE*k) = PEWiseCombinedVal[i][j*NUM_PE+k];
            }
            value_vec[i].push_back(wideData);
            streamsize[i]+=1;
        }            
        
    }
}

/*
    This preprocessing function was used when tapa::vec_t was used to send combined data. It has MAC wise queues
    to store relevant data and later combine them as one PE stream.
*/
void preProcessing_vectorWise(int& rows, int& cols, int& nnz, std::vector<float>& csr_values, std::vector<int>& csr_colIdx, std::vector<int>& csr_rowIdx, std::vector<float>& b_vec, std::vector<t_WIDE> *value_vec, std::vector<float> *b_vec_part, int *streamsize){

    for (int i = 0; i < NUM_PE; i++)
    {
        streamsize[i] = 0;
    }

    
    std::queue<t_DW> macWiseData[CONCAT_FACTOR];
    
    t_DW valColIdxRowIdx;
    
    t_WIDE concatData;

    //First assign rows to individual streams for each PE
    for (t_HW i = 0; i < rows; i++)
    {
        float b_val = b_vec[i];
        b_vec_part[i%NUM_PE].push_back(b_val);

        t_HW row_start = csr_rowIdx[i];
        t_HW row_end = csr_rowIdx[i+1];
        t_DW row_size = row_end - row_start;
        t_HW rowIdx = (i);

        for(t_HW j=row_start; j<row_end-1; j++){
            float val = csr_values[j];
            t_HW colIdx = csr_colIdx[j];

            int val_in_int = (*(int *)(&val));

            valColIdxRowIdx = ((t_DW)val_in_int << 32) | (colIdx << 16) | (rowIdx);

            macWiseData[colIdx%CONCAT_FACTOR].push(valColIdxRowIdx);   //assign column data to individual vectors based on the column index
        }

        // if(row_size>8){
        //     for (int k = 0; k < 8; k++) //to support adder tree
        //     {
        //         insertData = 0;
        //         indivStreams[i%totalIndivStreams].push_back(insertData);
        //     }
        // }

        //Inserting the last row value followed by number of non zeros in the row
        float val = csr_values[row_end-1];
        t_HW colIdx = csr_colIdx[row_end-1];
        int val_in_int = (*(int *)(&val));
        valColIdxRowIdx = ((t_DW)val_in_int << 32) | (colIdx << 16) | (rowIdx);
        macWiseData[colIdx%CONCAT_FACTOR].push(valColIdxRowIdx); //insert diagonal value
        macWiseData[colIdx%CONCAT_FACTOR].push(row_size); //insert row size. i.e., nnz in the row

        //combine CONCAT_FACTOR time streams to make widened data
        int maxLength = std::max(macWiseData[0].size(), macWiseData[1].size());
        for (int j = 0; j < maxLength; j++)
        {
            for (int k = 0; k < CONCAT_FACTOR; k++)
            {
                if (!macWiseData[k].empty())
                {
                    concatData[k] = macWiseData[k].front();
                    macWiseData[k].pop();
                }
                else{
                    concatData[k] = 1;
                }
            }

            value_vec[i%NUM_PE].push_back(concatData);
            streamsize[i%NUM_PE]++;
        }
    }
}


/*
    In this preprosseing, let's say one mac stream is 2 or more shorter than the other mac unit. Then this fill up
    shorter mac stream with dummies until (maxLength-2), then add diagVal and number of nonzeros.
*/
void preProcessing_pushdiagValParallelToNonDiag(int& rows, int& cols, int& nnz, std::vector<float>& csr_values, std::vector<int>& csr_colIdx, std::vector<int>& csr_rowIdx, std::vector<float>& b_vec, std::vector<t_WIDE> *value_vec, std::vector<float> *b_vec_part, int *streamsize){

    std::vector<float> PEWiseVal[NUM_PE];
    std::vector<int> PEWiseRowIdx[NUM_PE];
    std::vector<int> PEWiseColIdx[NUM_PE];
    std::vector<int> PEWiseRowLength[NUM_PE];
    std::vector<t_DW> PEWiseCombinedVal[NUM_PE];

    int cummulativeLength[NUM_PE];
    int writtenLength[NUM_PE];

    std::queue<t_DW> macWiseData[CONCAT_FACTOR];
    int macWiseLength[CONCAT_FACTOR];


    float matVal_float;
    int matVal_int;
    t_HW colIdx;
    t_HW rowIdx;
    t_DW combinedVal;

    for (int i = 0; i < NUM_PE; i++)
    {
        writtenLength[i] = 0;
        cummulativeLength[i] = 0;
        streamsize[i] = 0;
    }

    /*
    1. assign rows cyclicly
    2. combine {val, colIdx, rowIdx}
    3. after diag value, append row size
    4. pad '1' s as necessary. '1' to avoid being treated as diag value and to be discarded
            - Rowwise synchronized
            - Padding dummy values after sending diag val.
            - Padding dummy values after sending number of non zeros in the row
    */   
    for (int i = 0; i < rows; i++)
    {

        // printf("row=%d\n",i);

        float b_val = b_vec[i];
        b_vec_part[i%NUM_PE].push_back(b_val);

        for (int j = 0; j < CONCAT_FACTOR; j++)
        {
            macWiseLength[j] = 0;
        }
        

        int startIdx = csr_rowIdx[i];
        int endIdx = csr_rowIdx[i+1];

        //gathering data as macwise
        for (int j = startIdx; j < endIdx-1; j++)
        {
            // printf("[appendingNonDiag] rowIdx=%d\n",i);
            t_HW rowIdx = i;
            t_HW colIdx = csr_colIdx[j];
            float val = csr_values[j];
            int val_in_int = (*(int *)(&val));

            t_DW combinedVal = (((t_DW)val_in_int)<<32) | (colIdx<<16) | rowIdx;
            
            macWiseData[(colIdx%CONCAT_FACTOR)].push(combinedVal);
            macWiseLength[(colIdx%CONCAT_FACTOR)]+=1;
        }

        t_HW rowIdx = i;
        t_HW colIdx = csr_colIdx[endIdx-1];
        float val = csr_values[endIdx-1];
        int val_in_int = (*(int *)(&val));

        t_DW diagVal = (((t_DW)val_in_int)<<32) | (colIdx<<16) | rowIdx;
        t_DW nnzInRow = (t_DW)(endIdx - startIdx);

        int maxMacLength=0;
        int maxMacIdx = 0;
        int minMacLength=0xFFFFFFF;
        int minMacIdx = 0;

        //find min and max lengths of macwise data
        for (int j = 0; j < CONCAT_FACTOR; j++)
        {
            if (maxMacLength < macWiseLength[j])
            {
                maxMacLength = macWiseLength[j];
                maxMacIdx = j;
            }

            if (macWiseLength[j] < minMacLength)
            {
                minMacLength = macWiseLength[j];
                minMacIdx = j;
            }
        }

        // printf("rowIdx=%d, mac0_length=%d, mac1_length=%d, min=%d, max=%d\n",i, macWiseLength[0], macWiseLength[1], minMacLength, maxMacLength);

        if ((maxMacLength-minMacLength)>=2) //the streams are unbalanced. Need padding. Also we can place diag val and size while balancing
        {
            // printf("[diff2] came here rowIdx=%d\n",i);

            for (int j = 0; j < CONCAT_FACTOR; j++) //make sure every stream is filled up at least maxlen-2
            {
                if (macWiseLength[j]<(maxMacLength-2))
                {
                    int startLen = macWiseLength[j];
                    for (int k = startLen; k < (maxMacLength-2); k++)
                    {
                        macWiseData[j].push(1);
                        macWiseLength[j]+=1;
                    }
                }
            }
            
            //appending diag val and nnz to the stream which had less values initially.
            macWiseData[minMacIdx].push(diagVal);
            macWiseData[minMacIdx].push(nnzInRow);
            macWiseLength[minMacIdx]+=2;


            //filling up every mac unit stream till max
            for (int j = 0; j < CONCAT_FACTOR; j++) 
            {
                if (macWiseLength[j]<(maxMacLength))
                {
                    int startLen = macWiseLength[j];
                    for (int k = startLen; k < maxMacLength; k++)
                    {
                        macWiseData[j].push(1);
                        macWiseLength[j]+=1;
                    }
                }
            }
            
        }
        else if ((maxMacLength-minMacLength)==1)    //The max gap is 1. We can add the diag val and nnz, then max length become max+1
        {
            // printf("[diff1] came here rowIdx=%d\n",i);

            macWiseData[minMacIdx].push(diagVal);
            macWiseData[minMacIdx].push(nnzInRow);
            macWiseLength[minMacIdx]+=2;

            //filling up every mac unit stream till max+1
            for (int j = 0; j < CONCAT_FACTOR; j++) 
            {
                if (macWiseLength[j]<(maxMacLength+1))
                {
                    int startLen = macWiseLength[j];
                    for (int k = startLen; k < maxMacLength+1; k++)
                    {
                        macWiseData[j].push(1);
                        macWiseLength[j]+=1;
                    }
                }
            }
        }
        else{   //every stream is same size. No matter where you add the diag val and nnz
            // printf("[balancedMacs] came here rowIdx=%d\n",i);

            macWiseData[minMacIdx].push(diagVal);
            macWiseData[minMacIdx].push(nnzInRow);
            macWiseLength[minMacIdx]+=2;

            //filling up every mac unit stream till max+2 since now with diag val and nnz, max length is max+2
            for (int j = 0; j < CONCAT_FACTOR; j++) 
            {
                // printf("[balancing] rowIdx=%d macWiseLength[%d]=%d, maxLength=%d\n",i,j,macWiseLength[j],(maxMacLength+2));
                if (macWiseLength[j]<(maxMacLength+2))
                {
                    int startLen = macWiseLength[j];
                    for (int k = startLen; k < (maxMacLength+2); k++)
                    {
                        // printf("[inside]::row=%d, macIdx=%d, iter=%d\n",i,j,k);
                        macWiseData[j].push(1);
                        macWiseLength[j]+=1;
                    }
                }
            }
        }
        
        // for (int j = 0; j < CONCAT_FACTOR; j++)
        // {
        //     printf("mac=%d, length=%d\n",j,macWiseLength[j]);
        // }
        

        //At this point the row is assigned to mac streams and well balanced. Need to assign to respective val stream

        for (int j = 0; j < macWiseLength[0]; j++)  //every length is same.
        {
            t_WIDE concatData;
            for (int k = 0; k < CONCAT_FACTOR; k++)
            {
                concatData.range((k+1)*DATA_SIZE-1, DATA_SIZE*k) = macWiseData[k].front();
                macWiseData[k].pop();

            }
            value_vec[i%NUM_PE].push_back(concatData);
            streamsize[i%NUM_PE]+=1;
        }
    }
}

/*
    This preprosseing create the mac streams with all non diag vals. Then add diag val and number of non zeros
*/
void preProcessing_noFlags(int& rows, int& cols, int& nnz, std::vector<float>& csr_values, std::vector<int>& csr_colIdx, std::vector<int>& csr_rowIdx, std::vector<float>& b_vec, std::vector<t_WIDE> *value_vec, std::vector<float> *b_vec_part, int *streamsize){

    std::vector<float> PEWiseVal[NUM_PE];
    std::vector<int> PEWiseRowIdx[NUM_PE];
    std::vector<int> PEWiseColIdx[NUM_PE];
    std::vector<int> PEWiseRowLength[NUM_PE];
    std::vector<t_DW> PEWiseCombinedVal[NUM_PE];

    int cummulativeLength[NUM_PE];
    int writtenLength[NUM_PE];

    std::queue<t_DW> macWiseData[CONCAT_FACTOR];
    int macWiseLength[CONCAT_FACTOR];


    float matVal_float;
    int matVal_int;
    t_HW colIdx;
    t_HW rowIdx;
    t_DW combinedVal;

    for (int i = 0; i < NUM_PE; i++)
    {
        writtenLength[i] = 0;
        cummulativeLength[i] = 0;
        streamsize[i] = 0;
    }

    /*
    1. assign rows cyclicly
    2. combine {val, colIdx, rowIdx}
    3. after diag value, append row size
    4. pad '1' s as necessary. '1' to avoid being treated as diag value and to be discarded
            - Rowwise synchronized
            - Padding dummy values after sending diag val.
            - Padding dummy values after sending number of non zeros in the row
    */   
    for (int i = 0; i < rows; i++)
    {

        // printf("row=%d\n",i);

        float b_val = b_vec[i];
        b_vec_part[i%NUM_PE].push_back(b_val);

        for (int j = 0; j < CONCAT_FACTOR; j++)
        {
            macWiseLength[j] = 0;
        }
        

        int startIdx = csr_rowIdx[i];
        int endIdx = csr_rowIdx[i+1];

        //gathering data as macwise
        for (int j = startIdx; j < endIdx-1; j++)
        {
            // printf("[appendingNonDiag] rowIdx=%d\n",i);
            t_HW rowIdx = i;
            t_HW colIdx = csr_colIdx[j];
            float val = csr_values[j];
            int val_in_int = (*(int *)(&val));

            t_DW combinedVal = (((t_DW)val_in_int)<<32) | (colIdx<<16) | rowIdx;
            
            macWiseData[(colIdx%CONCAT_FACTOR)].push(combinedVal);
            macWiseLength[(colIdx%CONCAT_FACTOR)]+=1;
        }

        t_HW rowIdx = i;
        t_HW colIdx = csr_colIdx[endIdx-1];
        float val = csr_values[endIdx-1];
        int val_in_int = (*(int *)(&val));

        t_DW diagVal = (((t_DW)val_in_int)<<32) | (colIdx<<16) | rowIdx;
        t_DW nnzInRow = (t_DW)(endIdx - startIdx);

        int maxMacLength=0;

        //find max lengths of macwise data
        for (int j = 0; j < CONCAT_FACTOR; j++)
        {
            if (maxMacLength < macWiseLength[j])
            {
                maxMacLength = macWiseLength[j];
            }
        }

        for (int j = 0; j < CONCAT_FACTOR; j++) //make sure every stream is filled up at least maxlen-2
        {
            if (macWiseLength[j]<maxMacLength)
            {
                int startLen = macWiseLength[j];
                for (int k = startLen; k < maxMacLength; k++)
                {
                    macWiseData[j].push(i);
                    macWiseLength[j]+=1;
                }
            }
        }
        macWiseData[(colIdx%CONCAT_FACTOR)].push(diagVal);
        macWiseData[(colIdx%CONCAT_FACTOR)].push(maxMacLength*2);
        macWiseLength[(colIdx%CONCAT_FACTOR)]+=2;

        for (int j = 0; j < CONCAT_FACTOR; j++) //make sure every stream is filled up at least maxlen-2
        {
            if (macWiseLength[j]<(maxMacLength+2))
            {
                int startLen = macWiseLength[j];
                for (int k = startLen; k < (maxMacLength+2); k++)
                {
                    macWiseData[j].push(1);
                    macWiseLength[j]+=1;
                }
            }
        }
        
        //At this point the row is assigned to mac streams and well balanced. Need to assign to respective val stream

        for (int j = 0; j < macWiseLength[0]; j++)  //every length is same.
        {
            t_WIDE concatData;
            for (int k = 0; k < CONCAT_FACTOR; k++)
            {
                concatData.range((k+1)*DATA_SIZE-1, DATA_SIZE*k) = macWiseData[k].front();
                macWiseData[k].pop();

            }
            value_vec[i%NUM_PE].push_back(concatData);
            streamsize[i%NUM_PE]+=1;
        }
    }
}

/*
    This preprosseing create the mac streams with all non diag vals. Then add diag val and number of non zeros
    In addition this preprocessing includes flags to indicate the end of row and absolute dummy data.
    Input structure is as follows
    <32 - fp32 val> | <15-colIdx> | <15-rowIdx> | <1-rowEnd> | <1-dummy>

    This preprocessing sends 
        - seperate dummy data to indicate row end (which is not required)
        - seprat dummy data to balance streams when sending the diag val
*/
void preProcessing_withAdditionalDummyData(int& rows, int& cols, int& nnz, std::vector<float>& csr_values, std::vector<int>& csr_colIdx, std::vector<int>& csr_rowIdx, std::vector<float>& b_vec, std::vector<t_WIDE> *value_vec, std::vector<float> *b_vec_part, unsigned int *streamsize){

    std::vector<float> PEWiseVal[NUM_PE];
    std::vector<int> PEWiseRowIdx[NUM_PE];
    std::vector<int> PEWiseColIdx[NUM_PE];
    std::vector<int> PEWiseRowLength[NUM_PE];
    std::vector<t_DW> PEWiseCombinedVal[NUM_PE];

    int cummulativeLength[NUM_PE];
    int writtenLength[NUM_PE];

    std::queue<t_DW> macWiseData[CONCAT_FACTOR];
    int macWiseLength[CONCAT_FACTOR];

    int allDummyDataCounter = 0;


    float matVal_float;
    int matVal_int;
    t_HW colIdx;
    t_HW rowIdx;
    t_DW combinedVal;

    t_DW rowEndDummyData;

    for (int i = 0; i < NUM_PE; i++)
    {
        writtenLength[i] = 0;
        cummulativeLength[i] = 0;
        streamsize[i] = 0;
    }

    /*
    1. assign rows cyclicly
    2. combine {val, colIdx, rowIdx}
    3. after diag value, append row size
    4. pad '1' s as necessary. '1' to avoid being treated as diag value and to be discarded
            - Rowwise synchronized
            - Padding dummy values after sending diag val.
            - Padding dummy values after sending number of non zeros in the row
    */   
    for (int i = 0; i < rows; i++)
    {

        // printf("row=%d\n",i);

        float b_val = b_vec[i];
        b_vec_part[i%NUM_PE].push_back(b_val);

        for (int j = 0; j < CONCAT_FACTOR; j++)
        {
            macWiseLength[j] = 0;
        }
        

        int startIdx = csr_rowIdx[i];
        int endIdx = csr_rowIdx[i+1];

        //gathering data as macwise
        for (int j = startIdx; j < endIdx-1; j++)
        {
            // printf("[appendingNonDiag] rowIdx=%d\n",i);
            t_HW rowIdx = i;
            t_HW colIdx = csr_colIdx[j];
            float val = csr_values[j];
            int val_in_int = (*(int *)(&val));

            t_DW combinedVal = (((t_DW)val_in_int)<<32) | (colIdx<<17) | (rowIdx << 2);
            
            macWiseData[(colIdx%CONCAT_FACTOR)].push(combinedVal);
            macWiseLength[(colIdx%CONCAT_FACTOR)]+=1;
        }

        t_HW rowIdx = i;
        t_HW colIdx = csr_colIdx[endIdx-1];
        float val = csr_values[endIdx-1];
        int val_in_int = (*(int *)(&val));

        t_DW diagVal = (((t_DW)val_in_int)<<32) | (colIdx<<17) | (rowIdx << 2);
        // t_DW nnzInRow = (t_DW)(endIdx - startIdx);

        int maxMacLength=0;

        //find max lengths of macwise data
        for (int j = 0; j < CONCAT_FACTOR; j++)
        {
            if (maxMacLength < macWiseLength[j])
            {
                maxMacLength = macWiseLength[j];
            }
        }

        for (int j = 0; j < CONCAT_FACTOR; j++) //make sure every stream is filled up at least maxlen
        {
            if (macWiseLength[j]<maxMacLength)
            {
                int startLen = macWiseLength[j];
                for (int k = startLen; k < maxMacLength; k++)
                {
                    macWiseData[j].push(i<<2);     //This is a dummy data which is still getting processed. It is {i,00}. i.e., val=0, colIdx=0, rowIdx=i, rowEnd=0, dummy=0
                    macWiseLength[j]+=1;

                    allDummyDataCounter+=1;
                }
            }
        }
        //now at this level all mac wise streams are balanced and all non diagonal values are stored

        //Sending row end with a dummy data, which will be still forwarded.
        if (i==0){
            rowEndDummyData = (1UL<<17) | (((t_DW)(i))<<2) | 2; //push a dummy data with end of row indicated, i.e., val=0, colIdx=1, rowIdx=i, rowEnd=1, dummy=0
        }
        else{
            rowEndDummyData = (((t_DW)(i))<<2) | 2;
        }
        macWiseData[0].push(rowEndDummyData);   //push a dummy data with end of row indicated, i.e., val=0, colIdx=0, rowIdx=i, rowEnd=1, dummy=0
        macWiseLength[0]+=1;
        macWiseData[1].push(rowEndDummyData);  //push a dummy data with end of row indicated, i.e., val=0, colIdx=0, rowIdx=i, rowEnd=1, dummy=0
        macWiseLength[1]+=1;
        allDummyDataCounter+=2;

        //Adding diag val
        t_DW absDummyData = ((t_DW)(i)<<2) | 1;
        macWiseData[0].push(diagVal);   //Push diagonal value
        macWiseLength[0]+=1;
        macWiseData[1].push(absDummyData);  //push an absolutely dummy data, i.e., val=0, colIdx=0, rowIdx=i, rowEnd=0, dummy=1
        macWiseLength[1]+=1;
        allDummyDataCounter+=1;
        
        //At this point the row is assigned to mac streams and well balanced. Need to assign to respective val stream

        for (int j = 0; j < macWiseLength[0]; j++)  //every length is same.
        {
            t_WIDE concatData;
            for (int k = 0; k < CONCAT_FACTOR; k++)
            {
                concatData.range((k+1)*DATA_SIZE-1, DATA_SIZE*k) = macWiseData[k].front();
                macWiseData[k].pop();

            }
            value_vec[i%NUM_PE].push_back(concatData);
            streamsize[i%NUM_PE]+=1;
        }
    }

    //stats
    int totalStreamLength = 0;
    
    for (int i = 0; i < NUM_PE; i++)
    {
        totalStreamLength+=streamsize[i]*CONCAT_FACTOR;
    }

    float dummyDataPercentage = (((float)allDummyDataCounter)/totalStreamLength)*100;
    

    printf("Number of dummy data added=%d, Total Stream Size=%d, Percentage of dummy data=%.2f %%\n", allDummyDataCounter, totalStreamLength, dummyDataPercentage);
}

/*
    This preprosseing create the mac streams with all non diag vals. Then add diag val and number of non zeros
    In addition this preprocessing includes flags to indicate the end of row and absolute dummy data.
    Input structure is as follows
    <32 - fp32 val> | <15-colIdx> | <15-rowIdx> | <1-rowEnd> | <1-dummy>
*/
void preProcessing(int& rows, int& cols, int& nnz, std::vector<float>& csr_values, std::vector<int>& csr_colIdx, std::vector<int>& csr_rowIdx, std::vector<float>& b_vec, std::vector<t_WIDE> *value_vec, std::vector<float> *b_vec_part, unsigned int *streamsize){

    std::vector<float> PEWiseVal[NUM_PE];
    std::vector<int> PEWiseRowIdx[NUM_PE];
    std::vector<int> PEWiseColIdx[NUM_PE];
    std::vector<int> PEWiseRowLength[NUM_PE];
    std::vector<t_DW> PEWiseCombinedVal[NUM_PE];

    int cummulativeLength[NUM_PE];
    int writtenLength[NUM_PE];

    std::deque<t_DW> macWiseData[CONCAT_FACTOR];
    int macWiseLength[CONCAT_FACTOR];

    int allDummyDataCounter = 0;


    float matVal_float;
    int matVal_int;
    t_HW colIdx;
    t_HW rowIdx;
    t_DW combinedVal;

    t_DW rowEndDummyData;

    for (int i = 0; i < NUM_PE; i++)
    {
        writtenLength[i] = 0;
        cummulativeLength[i] = 0;
        streamsize[i] = 0;
    }

    /*
    1. assign rows cyclicly
    2. combine {val, colIdx, rowIdx}
    3. after diag value, append row size
    4. pad '1' s as necessary. '1' to avoid being treated as diag value and to be discarded
            - Rowwise synchronized
            - Padding dummy values after sending diag val.
            - Padding dummy values after sending number of non zeros in the row
    */   
    for (int i = 0; i < rows; i++)
    {

        // printf("row=%d\n",i);

        float b_val = b_vec[i];
        b_vec_part[i%NUM_PE].push_back(b_val);

        for (int j = 0; j < CONCAT_FACTOR; j++)
        {
            macWiseLength[j] = 0;
        }
        

        int startIdx = csr_rowIdx[i];
        int endIdx = csr_rowIdx[i+1];

        //gathering data as macwise
        for (int j = startIdx; j < endIdx-1; j++)
        {
            // printf("[appendingNonDiag] rowIdx=%d\n",i);
            t_HW rowIdx = i;
            t_HW colIdx = csr_colIdx[j];
            float val = csr_values[j];
            int val_in_int = (*(int *)(&val));

            t_DW combinedVal = (((t_DW)val_in_int)<<32) | (colIdx<<17) | (rowIdx << 2);
            
            macWiseData[(colIdx%CONCAT_FACTOR)].push_back(combinedVal);
            macWiseLength[(colIdx%CONCAT_FACTOR)]+=1;
        }

        t_HW rowIdx = i;
        t_HW colIdx = csr_colIdx[endIdx-1];
        float val = csr_values[endIdx-1];
        int val_in_int = (*(int *)(&val));

        t_DW diagVal = (((t_DW)val_in_int)<<32) | (colIdx<<17) | (rowIdx << 2);
        // t_DW nnzInRow = (t_DW)(endIdx - startIdx);

        int maxMacLength=0;

        //find max lengths of macwise data
        for (int j = 0; j < CONCAT_FACTOR; j++)
        {
            if (maxMacLength < macWiseLength[j])
            {
                maxMacLength = macWiseLength[j];
            }
        }

        for (int j = 0; j < CONCAT_FACTOR; j++) //make sure every stream is filled up at least maxlen
        {
            if (macWiseLength[j]<maxMacLength)
            {
                int startLen = macWiseLength[j];
                for (int k = startLen; k < maxMacLength; k++)
                {
                    macWiseData[j].push_back(i<<2);     //This is a dummy data which is still getting processed. It is {i,00}. i.e., val=0, colIdx=0, rowIdx=i, rowEnd=0, dummy=0
                    macWiseLength[j]+=1;

                    allDummyDataCounter+=1;
                }
            }
        }
        //now at this level all mac wise streams are balanced and all non diagonal values are stored. So, if there are non diag vals, need to append
        //end of row signal to the last two vals. 
        
        if(maxMacLength==0){ //this row doesn't have any non diag val. need to append end of row to diag val
            //Adding diag val
            t_DW absDummyData = ((t_DW)(i)<<2) | (1UL << 1) | (1UL);
            diagVal = diagVal | (1UL << 1);
            macWiseData[0].push_back(diagVal);   //Push diagonal value
            macWiseLength[0]+=1;
            macWiseData[1].push_back(absDummyData);  //push an absolutely dummy data, i.e., val=0, colIdx=0, rowIdx=i, rowEnd=0, dummy=1
            macWiseLength[1]+=1;
            allDummyDataCounter+=1;
        }
        else{   //this row has diag vals. 
            //so need to append end of row signal to last data
            for(int i=0; i<CONCAT_FACTOR; i++){
                t_DW lastData = macWiseData[i].back();
                macWiseData[i].pop_back();
                lastData = lastData | (1UL <<1);
                macWiseData[i].push_back(lastData);
            }

            //Adding diag val
            t_DW absDummyData = ((t_DW)(i)<<2) | (1UL);
            macWiseData[0].push_back(diagVal);   //Push diagonal value
            macWiseLength[0]+=1;
            macWiseData[1].push_back(absDummyData);  //push an absolutely dummy data, i.e., val=0, colIdx=0, rowIdx=i, rowEnd=0, dummy=1
            macWiseLength[1]+=1;
            allDummyDataCounter+=1;
        }
        
        //At this point the row is assigned to mac streams and well balanced. Need to assign to respective val stream

        for (int j = 0; j < macWiseLength[0]; j++)  //every length is same.
        {
            t_WIDE concatData;
            for (int k = 0; k < CONCAT_FACTOR; k++)
            {
                concatData.range((k+1)*DATA_SIZE-1, DATA_SIZE*k) = macWiseData[k].front();
                macWiseData[k].pop_front();

            }
            value_vec[i%NUM_PE].push_back(concatData);
            streamsize[i%NUM_PE]+=1;
        }
    }

    //stats
    int totalStreamLength = 0;
    
    for (int i = 0; i < NUM_PE; i++)
    {
        totalStreamLength+=streamsize[i]*CONCAT_FACTOR;
    }

    float dummyDataPercentage = (((float)allDummyDataCounter)/totalStreamLength)*100;
    

    printf("Number of dummy data added=%d, Total Stream Size=%d, Percentage of dummy data=%.2f %%\n", allDummyDataCounter, totalStreamLength, dummyDataPercentage);
}

//read only triangular part as COO format
void readVec(const std::string& filename, std::vector<float>& b_vec, int& rows) {
    // open file
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Error: could not open file " << filename << std::endl;
        exit(0);
    }

    // read data
    double value;
    for (int i = 0; i < rows; i++) {
        file >> value;
        b_vec.push_back(value);
    }

    file.close();
}

//Initialize x vector to 0
void initX(std::vector<float>& x_vec, int& rows) {
    for (int i = 0; i < rows; i++) {
        x_vec.push_back(0.0);
    }
}

//write output to file
void writeXVec(const std::string& filename, std::vector<float>& x_vec, int& rows) {
    // open file
    std::ofstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Error: could not open file " << filename << std::endl;
        return;
    }

    // write data
    for (int i = 0; i < rows; i++) {
        file << x_vec[i];
        file << "\n";
    }

    file.close();
}

void sptrsv_kernel_baseline(std::vector<float>& csr_values, 
                  std::vector<int>& csr_colIdx,
                  std::vector<int>& csr_rowIdx, 
                  std::vector<float>& b_vec,
                  std::vector<float>& x_vec,
                  int rows){

  for (int i = 0; i < rows; i++)
  {
    int startIdx = csr_rowIdx[i];
    int endIdx = csr_rowIdx[i+1];

    float b_val = b_vec[i];
    float accum = 0.0;

    for (int j = startIdx; j < endIdx-1; j++)
    {
      accum = accum + (csr_values[j]*x_vec[csr_colIdx[j]]);
    }

    x_vec[i] = (b_val-accum)/csr_values[endIdx-1];

    // for (int j = startIdx; j < endIdx-1; j++)
    // {
    //   b_val = b_val - (csr_values[j]*x_vec[csr_colIdx[j]]);
    // }

    // x_vec[i] = (b_val)/csr_values[endIdx-1];

  }
  
}

float calculateRMSE(std::vector<float>& gold, std::vector<float>& test, int rows){
    float rmse = 0.0;

    for (int i = 0; i < rows; i++)
    {
        rmse += (gold[i]-test[i])*(gold[i]-test[i]);
        if (abs(gold[i]-test[i])>0.0001)
        {
            printf("idx=%d, gold=%f, test=%f\n", i, gold[i], test[i]);
        }
        
    }

    rmse/=rows;

    rmse = sqrt (rmse);

    return rmse;
}

int main(int argc, char* argv[])
{
    // float value[NNZ] = {0.1243, 0.9434, 0.3234, 0.6753, 1.4334, 5.2133, 0.4324, 4.9242, 3.4342, 4.5359, 2.2138, 2.6546, 3.2489, 4.3865};
    // int col_idx[NNZ] = {0, 1, 2, 0, 3, 4, 1, 3, 5, 4, 5, 6, 2, 7};
    // int row_idx[SIZE+1] = {0, 1, 2, 3, 5, 6, 9, 12, 14};

    // float b[SIZE] = {0.13673, 2.07548, 1.06722, 7.04979, 28.67315, 45.28348, 59.99895, 49.32257};
    gflags::ParseCommandLineFlags(&argc, &argv, /*remove_flags=*/true);

    std::string fileName = "input.mtx";

    printf("File Name is: %s\n\n", fileName.c_str());

    //read into COO format
    std::vector<float> coo_values;
    std::vector<int> coo_rowIdx;
    std::vector<int> coo_colIdx;

    int rows=0, cols=0, nnz=0;

    printf("# Start reading into COO format\n");
    readMatrixCOO(fileName, coo_values, coo_rowIdx, coo_colIdx, rows, cols, nnz);
    printf("# Reading into COO format completed succesfully\n\n");

    //convert into CSR format
    std::vector<float> csr_values;
    std::vector<int> csr_colIdx;
    std::vector<int> csr_rowIdx;
    
    printf("# Start converting to CSR format\n");
    COO_to_CSR(coo_values, coo_rowIdx, coo_colIdx, csr_values, csr_colIdx, csr_rowIdx, rows, cols, nnz);
    printf("# Converting to CSR completed succesfully\n\n");

    //read B vector
    std::vector<float> b_vec;
    std::string b_vecFileName = "b_vec.txt";
    readVec(b_vecFileName, b_vec, rows);

    //preprocessing   
    std::vector<t_WIDE> value_vec[NUM_PE];
    unsigned int streamSize[NUM_PE];
    std::vector<float> b_vec_part[NUM_PE];

    printf("# Start preprocessing\n");
    preProcessing(rows, cols, nnz, csr_values, csr_colIdx, csr_rowIdx, b_vec, value_vec, b_vec_part, streamSize);
    for (int i = 0; i < NUM_PE; i++)
    {
        printf("Stream %d size = %d\n", i, streamSize[i]);
    }
    printf("# Preprocessing completed succesfully\n\n");
    

    //Initialize X vector
    std::vector<float> x_vec;
    initX(x_vec, rows);

    // //initialize output B
    // std::vector<float> dummy_b_out_vec;
    // initX(dummy_b_out_vec, rows);
    
    printf("\nStarting kernel execution.\n");

    printf("=====Argument=%s=======\n",FLAGS_bitstream.c_str());

    int64_t kernel_time_ns = tapa::invoke(sptrsv_kernel, FLAGS_bitstream, 
            tapa::read_only_mmap<const t_WIDE>(value_vec[0]),
            tapa::read_only_mmap<const t_WIDE>(value_vec[1]), 
            tapa::read_only_mmap<const float>(b_vec_part[0]),  
            tapa::read_only_mmap<const float>(b_vec_part[1]), 
            tapa::write_only_mmap<float>(x_vec), 
            streamSize[0], streamSize[1],
            rows);
            

    printf("\nKernel execution sucessful.\n");
    std::clog << "kernel time: " << kernel_time_ns * 1e-9 << " s\n\n" << std::endl;

    

    // printf("\n\n");
    

    // printf("\nResults:\n");
    // for (int i = 0; i < rows; i++)
    // {
    //     printf("x[%d]=%f\n",i,x_vec[i]);
    // }
    
    //write output to a file
    std::string outputFileName = "outx.txt";
    writeXVec(outputFileName, x_vec, rows);

    //calculate RMSE
    std::vector<float> gold_vec;
    initX(gold_vec, rows);
    // std::string gold_vecFileName = "gold.txt";
    // readVec(gold_vecFileName, gold_vec, rows);
    struct timeval start, stop;
    float hostTime;
    gettimeofday(&start,NULL);
    sptrsv_kernel_baseline(csr_values, csr_colIdx, csr_rowIdx, b_vec, gold_vec, rows);
    gettimeofday(&stop,NULL);
    hostTime = (stop.tv_usec-start.tv_usec)*1.0e-6 + stop.tv_sec - start.tv_sec;
    std::clog << "Single thread host time: " << hostTime << " s\n\n" << std::endl; 


    float rmse = calculateRMSE(gold_vec, x_vec, rows);
    printf("RMSE=%f\n", rmse);

    if (rmse<=0.0001)
    {
        printf("Test Passed\n");
    }
    else{
        printf("Test Failed. High RMSE detected\n");
    }

    printf("\nRun completed\n");

    //Testing
    std::string rmseFile = "rmse.txt";
    std::ofstream rmse_file(rmseFile);
    if (!rmse_file.is_open()) {
        std::cerr << "Error: could not open file " << rmseFile << std::endl;
        return 1;
    }
    // write data
    rmse_file << rmse;
    rmse_file.close();
    //Testing end

    return 0;
}

