#include <iostream>
#include <stdio.h>
#include <tapa.h>
#include <vector>
#include <gflags/gflags.h>

#include <string>
#include <fstream>
#include <sstream>
#include <iomanip>

#define NNZ 14
#define SIZE 8

using namespace std;


//read only triangular part as COO format
void readMatrixCOO(const string& filename, vector<float>& values, vector<int>& rowIdx, vector<int>& colIdx, int& rows, int& cols, int& nnz) {
    // open file
    ifstream file(filename);
    if (!file.is_open()) {
        cerr << "Error: could not open file " << filename << endl;
        return;
    }

    // file.precision(64);

    // read header
    string line;
    getline(file, line);
    while (line[0] == '%') {
        getline(file, line);
    }

    // parse header
    int orig_nnz=0;
    istringstream iss(line);
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

void COO_to_CSR(vector<float>& coo_values, vector<int>& coo_rowIdx, vector<int>& coo_colIdx, vector<float>& csr_values, vector<int>& csr_colIdx, vector<int>& csr_rowIdx,  int& rows, int& cols, int& nnz) {
    
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
    vector<vector<int>> rowWiseColIdx; //to hold row wise col indexes
    vector<vector<float>> rowWiseVals; //to hold row wise values
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


int main(int argc, char** argv)
{


    string fileName = "bcsstk01.mtx";

    printf("File Name is: %s\n", fileName.c_str());

    //read into COO format
    std::vector<float> coo_values;
    std::vector<int> coo_rowIdx;
    std::vector<int> coo_colIdx;

    int rows=0, cols=0, nnz=0;

    printf("# Start reading into COO format\n");
    readMatrixCOO(fileName, coo_values, coo_rowIdx, coo_colIdx, rows, cols, nnz);
    printf("# Reading into COO format completed succesfully\n");

    //convert into CSR format
    std::vector<float> csr_values;
    std::vector<int> csr_colIdx;
    std::vector<int> csr_rowIdx;
    
    COO_to_CSR(coo_values, coo_rowIdx, coo_colIdx, csr_values, csr_colIdx, csr_rowIdx, rows, cols, nnz);

    
    printf("\nValues:\n");
    for (int i = 0; i < nnz; i++)
    {
        printf("%f,",csr_values[i]);
    }
    printf("\n");
    
    printf("ColIdx:\n");
    for (int i = 0; i < nnz; i++)
    {
        printf("%d,",csr_colIdx[i]);
    }
    printf("\n");

    printf("RowIdx:\n");
    for (int i = 0; i < rows+1; i++)
    {
        printf("%d,",csr_rowIdx[i]);
    }
    printf("\n");
    


    // float value[NNZ] = {0.1243, 0.9434, 0.3234, 0.6753, 1.4334, 5.2133, 0.4324, 4.9242, 3.4342, 4.5359, 2.2138, 2.6546, 3.2489, 4.3865};
    // int col_idx[NNZ] = {0, 1, 2, 0, 3, 4, 1, 3, 5, 4, 5, 6, 2, 7};
    // int row_idx[SIZE+1] = {0, 1, 2, 3, 5, 6, 9, 12, 14};

    // float b[SIZE] = {0.13673, 2.07548, 1.06722, 7.04979, 28.67315, 45.28348, 59.99895, 49.32257};

    

    //re arrange inputs to send row index and number of values before the col index and value.
    // std::vector<float> value_vec0;
    // std::vector<int> colIdx_vec0;

    // std::vector<float> value_vec1;
    // std::vector<int> colIdx_vec1;

    // std::vector<float> b_vec;
    // std::vector<float> x_vec;

    // printf("\nStarting modifying inputs.\n");

    // for (int i = 0; i < SIZE; i=i+2)
    // {

    //     //0th vector
    //     int row_start0 = row_idx[i];
    //     int row_end0 = row_idx[i+1];
    //     int row_size0 = row_end0 - row_start0;
        
    //     value_vec0.push_back((float)i);
    //     colIdx_vec0.push_back(row_size0);

    //     for(int j=row_start0; j<row_end0; j++){
    //         value_vec0.push_back(value[j]);
    //         colIdx_vec0.push_back(col_idx[j]);
    //     }

    //     //1st vector
    //     int row_start1 = row_idx[i+1];
    //     int row_end1 = row_idx[i+2];
    //     int row_size1 = row_end1 - row_start1;
        
    //     value_vec1.push_back((float)(i+1));
    //     colIdx_vec1.push_back(row_size1);

    //     for(int j=row_start1; j<row_end1; j++){
    //         // printf("colIdx=%d, val=%f\n",col_idx[j], value[j]);
    //         value_vec1.push_back(value[j]);
    //         colIdx_vec1.push_back(col_idx[j]);
    //     }

    //     b_vec.push_back(b[i]);
    //     b_vec.push_back(b[i+1]);
    //     x_vec.push_back(0.0);
    //     x_vec.push_back(0.0);

    // }
    
    // printf("\nStarting kernel execution.\n");

    // printf("\nKernel execution sucessful.\n");

    
    

    // printf("\nResults:\n");
    // for (int i = 0; i < SIZE; i++)
    // {
    //     printf("row=%d, value=%f\n",i,x_vec[i]);
    // }
    
    return 0;
}





