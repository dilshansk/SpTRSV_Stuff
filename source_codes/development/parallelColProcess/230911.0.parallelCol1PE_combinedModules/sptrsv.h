#include <ap_int.h>
#include <tapa.h>

//Input configs
// #define CONCAT_FACTOR 2
const int CONCAT_FACTOR=2;
// #define DATA_SIZE 64
const int DATA_SIZE=64;
// #define NUM_PE 2
const int NUM_PE=2;
// #define SUP_X_SIZE 8192
const int SUP_X_SIZE=8192;
// #define MEM_BANK_SIZE (SUP_X_SIZE/NUM_PE)
const int MEM_BANK_SIZE=(SUP_X_SIZE/NUM_PE);


//FIFO depths
// #define MAT_VAL_FIFO_DEPTH 64
const int MAT_VAL_FIFO_DEPTH=64;
// #define REQ_FIFO_DEPTH 1
const int REQ_FIFO_DEPTH=4;
// #define GEN_FIFO_DEPTH 1
const int GEN_FIFO_DEPTH=4;

//Data types
#define t_bit ap_uint<1>
#define t_WIDE ap_uint<128>
#define t_DW uint64_t
#define t_HW uint16_t



// There is a bug in Vitis HLS preventing fully pipelined read/write of struct
// via m_axi; using ap_uint can work-around this problem.
// template <typename T>
// using bits = ap_uint<tapa::widthof<T>()>;

// void sptrsv_kernel(float value[NNZ], int row[NNZ], int col[NNZ], float b[SIZE], float x[SIZE]);
// void sptrsv_kernel(tapa::mmap<const float> value0, tapa::mmap<const int> col_idx0, tapa::mmap<const float> value1, tapa::mmap<const int> col_idx1, tapa::mmap<const float> b, tapa::mmap<float> x, int size1, int size2);


