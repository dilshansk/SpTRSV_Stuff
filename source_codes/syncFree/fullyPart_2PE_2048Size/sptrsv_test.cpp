#include "sptrsv.h"
#include <gflags/gflags.h>
#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <vector>

#include <string>
#include <fstream>
#include <sstream>
#include <iomanip>

// void sptrsv_kernel(tapa::mmap<const float> value0, tapa::mmap<const int> col_idx0, tapa::mmap<const float> value1, tapa::mmap<const int> col_idx1, tapa::mmap<const float> b, tapa::mmap<float> x, int size1, int size2);
// void sptrsv_kernel(tapa::mmap<const float> value0, tapa::mmap<const int> col_idx0, tapa::mmap<const float> value1, tapa::mmap<const int> col_idx1, tapa::mmap<const float> b, tapa::mmap<float> x, int size1, int size2);
void sptrsv_kernel(tapa::mmap<const float> value0, tapa::mmap<const int> col_idx0, tapa::mmap<const float> value1, tapa::mmap<const int> col_idx1, tapa::mmap<const float> b, tapa::mmap<float> x, int rows, int size1, int size2);

DEFINE_string(bitstream, "", "path to bitstream file, run csim if empty");


// using namespace std;

//read only triangular part as COO format
void readMatrixCOO(const std::string& filename, std::vector<float>& values, std::vector<int>& rowIdx, std::vector<int>& colIdx, int& rows, int& cols, int& nnz) {
    // open file
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Error: could not open file " << filename << std::endl;
        return;
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

//Assigning values to different banks
void preProcessing(int& rows, int& cols, int& nnz, std::vector<float>& csr_values, std::vector<int>& csr_colIdx, std::vector<int>& csr_rowIdx, std::vector<float> *value_vec, std::vector<int> *colIdx_vec, int *streamsize){

    for (int i = 0; i < NU_PE; i++)
    {
        streamsize[i] = 0;
    }
    

    for (int i = 0; i < rows; i=i+NU_PE)
    {
        for (int j = 0; j < NU_PE; j++)
        {
            int row_start = csr_rowIdx[i+j];
            int row_end = csr_rowIdx[i+j+1];
            int row_size = row_end - row_start;
            
            value_vec[j].push_back((float)(i+j));
            colIdx_vec[j].push_back(row_size);

            streamsize[j]++;
            
            for(int k=row_start; k<row_end-1; k++){
                value_vec[j].push_back(csr_values[k]);
                colIdx_vec[j].push_back(csr_colIdx[k]);
                streamsize[j]++;
            }

            if(row_size>8){
                for (int k = 0; k < 8; k++) //to support adder tree
                {
                    value_vec[j].push_back(0.0);
                    colIdx_vec[j].push_back(0);
                    streamsize[j]++;
                }
            }

            value_vec[j].push_back(csr_values[row_end-1]);
            colIdx_vec[j].push_back(csr_colIdx[row_end-1]);
            streamsize[j]++;
            
        }
    }
}


//read only triangular part as COO format
void readBVec(const std::string& filename, std::vector<float>& b_vec, int& rows) {
    // open file
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Error: could not open file " << filename << std::endl;
        return;
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

int main(int argc, char** argv)
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

    //preprocessing   
    std::vector<float> value_vec[NU_PE];
    std::vector<int> colIdx_vec[NU_PE];
    int streamSize[NU_PE];
    printf("# Start preprocessing\n");
    preProcessing(rows, cols, nnz, csr_values, csr_colIdx, csr_rowIdx, value_vec, colIdx_vec, streamSize);
    printf("Stream 1 size=%d, Stream 2 size=%d\n",streamSize[0], streamSize[1]);
    printf("# Preprocessing completed succesfully\n\n");

    //read B vector
    std::vector<float> b_vec;
    std::string b_vecFileName = "b_vec.txt";
    readBVec(b_vecFileName, b_vec, rows);
    

    //Initialize X vector
    std::vector<float> x_vec;
    initX(x_vec, rows);

    // //initialize output B
    // std::vector<float> dummy_b_out_vec;
    // initX(dummy_b_out_vec, rows);
    
    printf("\nStarting kernel execution.\n");


    printf("=====Argument=%s=======\n",FLAGS_bitstream.c_str());

    int64_t kernel_time_ns = tapa::invoke(sptrsv_kernel, FLAGS_bitstream, 
            tapa::read_only_mmap<const float>(value_vec[0]), tapa::read_only_mmap<const int>(colIdx_vec[0]), tapa::read_only_mmap<const float>(value_vec[1]), tapa::read_only_mmap<const int>(colIdx_vec[1]), tapa::read_only_mmap<const float>(b_vec), tapa::write_only_mmap<float>(x_vec), rows, streamSize[0], streamSize[1]);

    printf("\nKernel execution sucessful.\n");
    std::clog << "kernel time: " << kernel_time_ns * 1e-9 << " s\n\n" << std::endl;

    // printf("\n\n");
    

    printf("\nResults:\n");
    for (int i = 0; i < rows; i++)
    {
        printf("x[%d]=%f\n",i,x_vec[i]);
    }
    
    //write output to a file
    std::string outputFileName = "outx.txt";
    writeXVec(outputFileName, x_vec, rows);

    printf("Run completed\n");

    return 0;
}