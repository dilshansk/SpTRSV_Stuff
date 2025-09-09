#include <tapa.h>
#include <ap_int.h>

//Input configs
#define SUP_SIZE 1024
#define NU_PE 2
#define SMALL_MEM 512
#define NU_SMALL_MEMS 2

//Data types
#define t_bit ap_uint<1>
#define t_WD ap_uint<1>

//PE state machine states
#define LOAD_B 0
#define INIT_ROW 1
#define MODIFY_B 2
#define SAMPLE_DATA 3
#define CHECK_X_REQ 4
#define CHECK_X_SOLVED 5
#define COMP_DIV 6
#define COMP_MODIFY_B 7
#define UPDATE_X 8

// void sptrsv_kernel(float value[NNZ], int row[NNZ], int col[NNZ], float b[SIZE], float x[SIZE]);
// void sptrsv_kernel(tapa::mmap<const float> value0, tapa::mmap<const int> col_idx0, tapa::mmap<const float> value1, tapa::mmap<const int> col_idx1, tapa::mmap<const float> b, tapa::mmap<float> x, int size1, int size2);

