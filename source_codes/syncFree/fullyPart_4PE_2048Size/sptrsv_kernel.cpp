#include "sptrsv.h"

void Mmap2Stream_b(tapa::mmap<const float> b_mmap, tapa::ostream<float>& b_stream, int rows) {
  for (uint64_t i = 0; i < rows; ++i) {
    #pragma HLS PIPELINE II=1
    b_stream << b_mmap[i];
  }
}

void Mmap2Stream_values(tapa::mmap<const float> val_mmap, tapa::mmap<const int> colIdx_mmap, tapa::ostream<float>& val_stream, tapa::ostream<int>& colIdx_stream, tapa::ostream<bool>& tlast_stream, int nnzs) {
  for (uint64_t i = 0; i < nnzs-1; ++i) {
    #pragma HLS PIPELINE II=1
    val_stream << val_mmap[i];
    colIdx_stream << colIdx_mmap[i];
    tlast_stream << false;
  }
  val_stream << val_mmap[nnzs-1];
  colIdx_stream << colIdx_mmap[nnzs-1];
  tlast_stream << true;
}

// void Mmap2Stream_value(tapa::mmap<const float> val_mmap, tapa::ostream<float>& val_stream, int size) {
//   for (uint64_t i = 0; i < size; ++i) {
//     #pragma HLS PIPELINE II=1
//     val_stream << val_mmap[i];
//   }
// }

// void Mmap2Stream_colIdx(tapa::mmap<const int> colIdx_mmap, tapa::ostream<int>& colIdx_stream, int size) {
//   for (uint64_t i = 0; i < size; ++i) {
//     #pragma HLS PIPELINE II=1
//     colIdx_stream << colIdx_mmap[i];
//   }
// }

void Stream2Mmap_solvedX(tapa::mmap<float> solvedX, tapa::istream<float>& solvedX_stream, int rows){
  for (uint64_t i = 0; i < rows; ++i) {
    #pragma HLS PIPELINE II=1
    solvedX[i] = solvedX_stream.read();
  }
}

void dummyBConsume(tapa::istream<float>& b_stream, int rows){ //to just read chain broadcasted b last. otherwise it hangs after the last PE
  for (uint64_t i = 0; i < rows; ++i) {
    #pragma HLS PIPELINE II=1
    float b_val = b_stream.read();
  }
}

// void dummyBPass(tapa::mmap<float> b_out_mmap, tapa::istream<float>& b_stream){
//   for (uint64_t i = 0; i < SIZE; ++i) {
//     #pragma HLS PIPELINE II=1
//     b_out_mmap[i] = b_stream.read();
//   }
// }


void sptrsv_PE(tapa::istream<float>& b_in_stream, 
              tapa::istream<float>& val_stream, 
              tapa::istream<int>& colIdx_stream,
              tapa::istream<bool>& tlast_stream,
              tapa::istream<float>& x_resp,
              tapa::ostream<float>& b_out_stream,
              tapa::ostream<int>& x_req_address,
              tapa::ostream<int>& solvedX_colIdx,
              tapa::ostream<float>& solvedX_val,
              int rows,
              int nnzs
              ){

  //initial state
  int status = INIT_ROW;

  //to store the column size
  int rowSize = 0;

  //to track processed nonZeros in the row
  int rowCounter = 0;

  //to keep track of the row number
  float float_rowNumber = 0.0; //will be read from float stream
  int rowNumber = 0;
  
  //To store local copy of vector b(RHS)
  float local_b[SUP_SIZE];
  #pragma HLS bind_storage variable=local_b type=RAM_1P impl=BRAM

  //To fascilitate intermediate variable as of solved X
  float solvedX = 0.0;

  //to hold data sampled from data streams
  float valSample = 0.0;
  int colIdxSample = 0;
  bool tlast_val = false;

  //To track x values
  t_bit x_valid_bit = 0;
  int x_valid = 0;
  float x_val = 0;

  //To keep track of procesing column index
  int colIdx = 0;

  //Sparse Triangle Value
  float matrix_val = 0.0;

  //To help intermediate steps of modifying b
  float modifyVal = 0.0;
  float b_val = 0.0;
  float modified_b = 0.0;
  float nonDiagSum = 0.0;

  //To control states
  bool processingRow = false;
  bool waitResp = false;

  //To support partial summ
  float part_sum_val[8];
  #pragma HLS bind_storage variable=part_sum_val type=RAM_2P
  int part_sum_rowIdx[8];
  #pragma HLS bind_storage variable=part_sum_rowIdx type=RAM_2P
  float adder_tree_init[8];
  #pragma HLS ARRAY_PARTITION variable=adder_tree_init type=complete
  int adder_tree_init_rowIdx[8];
  #pragma HLS ARRAY_PARTITION variable=adder_tree_init_rowIdx type=complete
  // ap_uint<3> part_sum_idx = 0;
  // float adder_tree_1[4];
  // #pragma HLS ARRAY_PARTITION variable=adder_tree_1 type=complete
  // float adder_tree_2[2];
  // #pragma HLS ARRAY_PARTITION variable=adder_tree_2 type=complete


  PE_INIT_B_1:for(int i=0; i<rows; i++){
    #pragma HLS PIPELINE II=1
    float b_val_in = b_in_stream.read();
    local_b[i] = b_val_in;
    b_out_stream << b_val_in; //Chain broadcasting to the next PE
  }

  PE_INIT_B_2:for (int i = rows; i < SUP_SIZE; i++)
  {
    #pragma HLS PIPELINE II=1
    local_b[i] = 0.0;
  }

  PART_SUM_INIT:for(int i=0; i<8; i++){
    #pragma HLS PIPELINE II=1
    part_sum_val[i] = 0.0;
    part_sum_rowIdx[i] = 0xFFFFFFFF;
    adder_tree_init[i] = 0.0;
    adder_tree_init_rowIdx[i] = 0xFFFFFFFF;
  }



  PE_MAIN:for (int i = 0; tlast_val==false; i++){
    #pragma HLS PIPELINE II=1 
    #pragma HLS dependence variable=part_sum_val inter RAW false
    if(!waitResp){
      valSample = val_stream.read();
      colIdxSample = colIdx_stream.read();
      tlast_val = tlast_stream.read();
    }
    if(!processingRow){
      float_rowNumber = valSample;
      rowSize = colIdxSample;
      rowNumber = (int)float_rowNumber;
      b_val = local_b[rowNumber];
      rowCounter = 0;

      // PARTSUM_INIT:for (int j = 0; j < 50; j++)
      // {
      //   #pragma HLS PIPELINE II=1
      //   part_sum[j] = 0.0;
      //   adder_tree_init[j%8] = 0.0;
      // }
      // part_sum_idx = 0;

      processingRow = true;//start processing row
    }
    else{
      colIdx = colIdxSample;
      matrix_val = valSample;
      if(colIdx == rowNumber){ //Diagonal value

        float adder_tree_1_0 = ((adder_tree_init_rowIdx[0]==rowNumber)? adder_tree_init[0]:0.0) + ((adder_tree_init_rowIdx[1]==rowNumber)? adder_tree_init[1]:0.0);
        float adder_tree_1_1 = ((adder_tree_init_rowIdx[2]==rowNumber)? adder_tree_init[2]:0.0) + ((adder_tree_init_rowIdx[3]==rowNumber)? adder_tree_init[3]:0.0);
        float adder_tree_1_2 = ((adder_tree_init_rowIdx[4]==rowNumber)? adder_tree_init[4]:0.0) + ((adder_tree_init_rowIdx[5]==rowNumber)? adder_tree_init[5]:0.0);
        float adder_tree_1_3 = ((adder_tree_init_rowIdx[6]==rowNumber)? adder_tree_init[6]:0.0) + ((adder_tree_init_rowIdx[7]==rowNumber)? adder_tree_init[7]:0.0);

        float adder_tree_2_0 = adder_tree_1_0 + adder_tree_1_1;
        float adder_tree_2_1 = adder_tree_1_2 + adder_tree_1_3;

        nonDiagSum = adder_tree_2_0 + adder_tree_2_1;
        modified_b = b_val - nonDiagSum;
        solvedX = modified_b/matrix_val;
        solvedX_colIdx << colIdx;
        solvedX_val << solvedX;

        processingRow = false;//done processing row
      }
      else{
        if(!waitResp){
          x_req_address << colIdx;
          waitResp = true;
        }
        if (!x_resp.empty())
        {
          waitResp = false;
          x_resp.try_read(x_val);
          modifyVal = x_val * matrix_val;
          float ps_val = part_sum_val[i&7];
          float ps_rIdx = part_sum_rowIdx[i&7];

          if(ps_rIdx==rowNumber){
            adder_tree_init[i&7] = ps_val + modifyVal;
            adder_tree_init_rowIdx[i&7] = ps_rIdx;
            part_sum_val[i&7] = ps_val + modifyVal;
          }
          else{
            adder_tree_init[i&7] = modifyVal;
            adder_tree_init_rowIdx[i&7] = rowNumber;
            part_sum_val[i&7] = modifyVal;
          }
          part_sum_rowIdx[i&7] =  rowNumber;
          waitResp = false;
        }
        
        // x_val = x_resp.read();

        // modifyVal = x_val * matrix_val;
        // float ps_val = part_sum_val[i&7];
        // float ps_rIdx = part_sum_rowIdx[i&7];

        // if(ps_rIdx==rowNumber){
        //   adder_tree_init[i&7] = ps_val + modifyVal;
        //   adder_tree_init_rowIdx[i&7] = ps_rIdx;
        //   part_sum_val[i&7] = ps_val + modifyVal;
        // }
        // else{
        //   adder_tree_init[i&7] = modifyVal;
        //   adder_tree_init_rowIdx[i&7] = rowNumber;
        //   part_sum_val[i&7] = modifyVal;
        // }
        // part_sum_rowIdx[i&7] =  rowNumber;
      }
      
    }
  }
  

}


void smallX(tapa::istream<bool>& sm_loopCondition,
            tapa::istream<float>& wr_val,
            tapa::istream<int>& wr_idx,
            tapa::istreams<int, NU_PE>& rd_idx,
            tapa::ostreams<float, NU_PE>& rd_val,
            tapa::ostream<bool>& sol_status,
            tapa::ostream<float>& final_out, int rows, int num){
  
  float part_x[SMALL_MEM];
  #pragma HLS ARRAY_PARTITION variable=part_x type=complete
  ap_uint<SMALL_MEM> part_x_flag;

  ap_uint<SMALL_MEM> expected_part_x_flag;

  bool doneSend = false;

  for (int i = 0; i < SMALL_MEM; i++)
  {
    #pragma HLS PIPELINE II=1
    if( i<(rows/NU_SMALL_MEMS) ){
      expected_part_x_flag.range(i,i) = 1;
    }
    else{
      expected_part_x_flag.range(i,i) = 0;
    }

    part_x_flag.range(i,i) = 0;

  }
  
  bool loopCondition = false;

  sol_status.write(false);

  for ( ; !loopCondition ; )
  {
    #pragma HLS PIPELINE II=1

    if ( !sm_loopCondition.empty() )
    {
      sm_loopCondition.try_read(loopCondition);
    }

    //write
    int wr_colIdxSample;
    float wr_valSample;
    if (!wr_val.empty())
    {
      wr_idx.try_read(wr_colIdxSample);
      wr_val.try_read(wr_valSample);
      part_x[wr_colIdxSample] = wr_valSample;
      part_x_flag.range(wr_colIdxSample, wr_colIdxSample) = 1;
    }
    

    for (int i = 0; i < NU_PE; i++)
    {
      #pragma HLS UNROLL
      //read
      int rd_idx_sample;
      float rd_val_sample;
      t_bit rd_flag_sample;
      int peekedReqAddress;
      bool isPeeked;

      if ( !rd_idx[i].empty() )
      {
        isPeeked = rd_idx[i].try_peek(peekedReqAddress); //peek the request
        if( (isPeeked) && (((part_x_flag.range(peekedReqAddress,peekedReqAddress)))==(t_bit)1) ){ //peek is successful & requested x is solved
          rd_idx[i].try_read(rd_idx_sample);
          rd_val_sample = part_x[rd_idx_sample];
          rd_val[i].try_write(rd_val_sample);
        }
      }
    }

    if ( (part_x_flag == expected_part_x_flag) & (!doneSend) )
    {
      sol_status.write(true);
      doneSend = true;
    }
    
  }

  for (int i = 0; i < (rows/NU_SMALL_MEMS); i++)
  {
    final_out.write(part_x[i]);
  }

}

void readXPart(tapa::istream<bool>& rd_loopConidtion,
              tapa::istreams<int, NU_PE>& x_req_address,
              tapa::istreams<float, NU_PE>& mem_0_vals,
              tapa::istreams<float, NU_PE>& mem_1_vals,
              tapa::istreams<float, NU_PE>& mem_2_vals,
              tapa::istreams<float, NU_PE>& mem_3_vals,
              tapa::ostreams<float, NU_PE>& x_resp_val,
              tapa::ostreams<int, NU_PE>& mem_0_req_addr,
              tapa::ostreams<int, NU_PE>& mem_1_req_addr,
              tapa::ostreams<int, NU_PE>& mem_2_req_addr,
              tapa::ostreams<int, NU_PE>& mem_3_req_addr
                ){

  bool mem0Requested[NU_PE];
  bool mem1Requested[NU_PE];
  bool mem2Requested[NU_PE];
  bool mem3Requested[NU_PE];

  for (int i = 0; i < NU_PE; i++)
  {
    #pragma HLS PIPELINE II=1
    mem0Requested[i] = false;
    mem1Requested[i] = false;
    mem2Requested[i] = false;
    mem3Requested[i] = false;
  }
  
  bool loopCondition = false;

  for (; !loopCondition ; )
  {
    #pragma HLS PIPELINE II=1

    if ( !rd_loopConidtion.empty() )
    {
      rd_loopConidtion.try_read(loopCondition);
    }
    

    for (int i = 0; i < NU_PE; i++)
    {
      #pragma HLS UNROLL
      int actualAddr;
      if ((!x_req_address[i].empty()) & (!mem0Requested[i]) & (!mem1Requested[i]) & (!mem2Requested[i]) & (!mem3Requested[i]) )
      {
        x_req_address[i].try_read(actualAddr);
        if(actualAddr%NU_SMALL_MEMS==0){      //Need changes if NU_SMALL_MEMS is not equal to 2
          int requestingAddr = actualAddr/NU_SMALL_MEMS;
          mem_0_req_addr[i].write(requestingAddr);
          mem0Requested[i] = true;
        }
        else if (actualAddr%NU_SMALL_MEMS==1) {
          int requestingAddr = (actualAddr-1)/NU_SMALL_MEMS;
          mem_1_req_addr[i].write(requestingAddr);
          mem1Requested[i] = true;
        }
        else if (actualAddr%NU_SMALL_MEMS==2) {
          int requestingAddr = (actualAddr-2)/NU_SMALL_MEMS;
          mem_2_req_addr[i].write(requestingAddr);
          mem2Requested[i] = true;
        }
        else if (actualAddr%NU_SMALL_MEMS==3) {
          int requestingAddr = (actualAddr-3)/NU_SMALL_MEMS;
          mem_3_req_addr[i].write(requestingAddr);
          mem3Requested[i] = true;
        }
      }

      float val_sample;
      if (mem0Requested[i] & (!mem_0_vals[i].empty()))
      {
        mem_0_vals[i].try_read(val_sample);
        mem0Requested[i] = false;
        x_resp_val[i].try_write(val_sample);
      }
      else if (mem1Requested[i] & (!mem_1_vals[i].empty()))
      {
        mem_1_vals[i].try_read(val_sample);
        mem1Requested[i] = false;
        x_resp_val[i].try_write(val_sample);
      }
      else if (mem2Requested[i] & (!mem_2_vals[i].empty()))
      {
        mem_2_vals[i].try_read(val_sample);
        mem2Requested[i] = false;
        x_resp_val[i].try_write(val_sample);
      }
      else if (mem3Requested[i] & (!mem_3_vals[i].empty()))
      {
        mem_3_vals[i].try_read(val_sample);
        mem3Requested[i] = false;
        x_resp_val[i].try_write(val_sample);
      }
    }
  }

  
}


void writeXPart(tapa::istream<bool>& wr_loopCondition,
                tapa::istreams<int, NU_PE>& x_solved_colIdx,
                tapa::istreams<float, NU_PE>& x_solved_val,
                tapa::ostreams<int, NU_PE>& wr_colIdx,
                tapa::ostreams<float, NU_PE>& wr_val
                ){

  bool loopCondition = false;

  for (; !loopCondition ; )
  {
    #pragma HLS PIPELINE II=1

    if( !wr_loopCondition.empty() ){
      wr_loopCondition.try_read(loopCondition);
    }

    for (int j = 0; j < NU_PE; j++)
    {
      #pragma HLS UNROLL
      int colIdx;
      int modified_colIdx;
      float val;
      if( !x_solved_colIdx[j].empty() ){
        x_solved_colIdx[j].try_read(colIdx);
        x_solved_val[j].try_read(val);

        if(colIdx%NU_SMALL_MEMS==0){
          modified_colIdx = colIdx/NU_SMALL_MEMS;
        }
        else if(colIdx%NU_SMALL_MEMS==1){
          modified_colIdx = (colIdx-1)/NU_SMALL_MEMS;
        }
        else if(colIdx%NU_SMALL_MEMS==2){
          modified_colIdx = (colIdx-2)/NU_SMALL_MEMS;
        }
        else if(colIdx%NU_SMALL_MEMS==3){
          modified_colIdx = (colIdx-3)/NU_SMALL_MEMS;
        }
        wr_colIdx[j].write(modified_colIdx);
        wr_val[j].write(val);
      }
    }
  }
  
}

void controlSmallMem(tapa::istream<bool>& mem_0_solved,
                    tapa::istream<bool>& mem_1_solved,
                    tapa::istream<bool>& mem_2_solved,
                    tapa::istream<bool>& mem_3_solved,
                    tapa::istream<float>& mem_0_vals,
                    tapa::istream<float>& mem_1_vals,
                    tapa::istream<float>& mem_2_vals,
                    tapa::istream<float>& mem_3_vals,
                    tapa::ostream<bool>& wr_terminate,
                    tapa::ostream<bool>& rd_terminate,
                    tapa::ostream<bool>& mem0_terminate,
                    tapa::ostream<bool>& mem1_terminate,
                    tapa::ostream<bool>& mem2_terminate,
                    tapa::ostream<bool>& mem3_terminate,
                    tapa::ostream<float>& x_out_stream, int rows){

  bool mem0Done = false;
  bool mem1Done = false;
  bool mem2Done = false;
  bool mem3Done = false;
  bool allMemDone = false;

  
  wr_terminate.write(false);
  rd_terminate.write(false);
  mem0_terminate.write(false);
  mem1_terminate.write(false);
  mem2_terminate.write(false);
  mem3_terminate.write(false);

  for ( ; !allMemDone ; )
  {
    #pragma HLS PIPELINE II=1

    if ( !mem_0_solved.empty() )
    {
      mem_0_solved.try_read(mem0Done);
    }

    if ( !mem_1_solved.empty() )
    {
      mem_1_solved.try_read(mem1Done);
    }

    if ( !mem_2_solved.empty() )
    {
      mem_2_solved.try_read(mem2Done);
    }

    if ( !mem_3_solved.empty() )
    {
      mem_3_solved.try_read(mem3Done);
    }

    // if( !wr_terminate.full() ){
    //   wr_terminate.try_write(false);
    // }
    
    // if( !rd_terminate.full() ){
    //   rd_terminate.try_write(false);
    // }

    // if ( mem0_terminate.full() ){
    //   mem0_terminate.try_write(false);
    // }

    // if ( mem1_terminate.full() ){
    //   mem1_terminate.try_write(false);
    // }

    allMemDone = mem0Done & mem1Done & mem2Done & mem3Done;

  }

  wr_terminate.write(true);
  rd_terminate.write(true);
  mem0_terminate.write(true);
  mem1_terminate.write(true);
  mem2_terminate.write(true);
  mem3_terminate.write(true);

  for (int i = 0; i < rows; i++)
  {
    #pragma HLS PIPELINE II=1

    if(i%NU_SMALL_MEMS==0){
      float mem0_val_sample = mem_0_vals.read();
      x_out_stream.write(mem0_val_sample);
    }
    else if(i%NU_SMALL_MEMS==1){
      float mem1_val_sample = mem_1_vals.read();
      x_out_stream.write(mem1_val_sample);
    }
    else if(i%NU_SMALL_MEMS==2){
      float mem2_val_sample = mem_2_vals.read();
      x_out_stream.write(mem2_val_sample);
    }
    else if(i%NU_SMALL_MEMS==3){
      float mem3_val_sample = mem_3_vals.read();
      x_out_stream.write(mem3_val_sample);
    }
  }

}

void manage_x(tapa::ostream<float>& x_out_stream,
              tapa::istreams<int, NU_PE>& x_req_address,
              tapa::istreams<int, NU_PE>& x_solved_colIdx,
              tapa::istreams<float, NU_PE>& x_solved_val,
              tapa::ostreams<float, NU_PE>& x_resp_val,
              int rows){

  tapa::streams<float, NU_PE> mem_0_rd_vals;
  tapa::streams<float, NU_PE> mem_1_rd_vals;
  tapa::streams<float, NU_PE> mem_2_rd_vals;
  tapa::streams<float, NU_PE> mem_3_rd_vals;

  tapa::streams<int, NU_PE> mem_0_rd_req_addr;
  tapa::streams<int, NU_PE> mem_1_rd_req_addr;
  tapa::streams<int, NU_PE> mem_2_rd_req_addr;
  tapa::streams<int, NU_PE> mem_3_rd_req_addr;
  
  tapa::streams<int, NU_PE> mem_wr_addr;
  tapa::streams<float, NU_PE> mem_wr_val;

  tapa::stream<bool> mem0_status;
  tapa::stream<bool> mem1_status;
  tapa::stream<bool> mem2_status;
  tapa::stream<bool> mem3_status;

  tapa::stream<bool> wr_terminate;
  tapa::stream<bool> rd_terminate;

  tapa::stream<bool> mem0_terminate;
  tapa::stream<bool> mem1_terminate;
  tapa::stream<bool> mem2_terminate;
  tapa::stream<bool> mem3_terminate;

  tapa::stream<float> mem_0_final_vals;
  tapa::stream<float> mem_1_final_vals;
  tapa::stream<float> mem_2_final_vals;
  tapa::stream<float> mem_3_final_vals;

  tapa::task()
    .invoke(controlSmallMem, mem0_status, mem1_status, mem2_status, mem3_status, mem_0_final_vals, mem_1_final_vals, mem_2_final_vals, mem_3_final_vals, wr_terminate, rd_terminate, mem0_terminate, mem1_terminate, mem2_terminate, mem3_terminate, x_out_stream, rows)
    .invoke(readXPart, rd_terminate, x_req_address, mem_0_rd_vals, mem_1_rd_vals, mem_2_rd_vals, mem_3_rd_vals, x_resp_val, mem_0_rd_req_addr, mem_1_rd_req_addr, mem_2_rd_req_addr, mem_3_rd_req_addr)
    .invoke(writeXPart, wr_terminate, x_solved_colIdx, x_solved_val, mem_wr_addr, mem_wr_val)
    .invoke(smallX, mem0_terminate, mem_wr_val[0], mem_wr_addr[0], mem_0_rd_req_addr, mem_0_rd_vals, mem0_status, mem_0_final_vals, rows, 0)
    .invoke(smallX, mem1_terminate, mem_wr_val[1], mem_wr_addr[1], mem_1_rd_req_addr, mem_1_rd_vals, mem1_status, mem_1_final_vals, rows, 1)
    .invoke(smallX, mem2_terminate, mem_wr_val[2], mem_wr_addr[2], mem_2_rd_req_addr, mem_2_rd_vals, mem2_status, mem_2_final_vals, rows, 2)
    .invoke(smallX, mem3_terminate, mem_wr_val[3], mem_wr_addr[3], mem_3_rd_req_addr, mem_3_rd_vals, mem3_status, mem_3_final_vals, rows, 3);
}

// void sptrsv_kernel(float* value, int* row, int* col, float* b, float* x)
void sptrsv_kernel(tapa::mmap<const float> value0, 
                  tapa::mmap<const int> col_idx0, 
                  tapa::mmap<const float> value1, 
                  tapa::mmap<const int> col_idx1, 
                  tapa::mmap<const float> value2, 
                  tapa::mmap<const int> col_idx2, 
                  tapa::mmap<const float> value3, 
                  tapa::mmap<const int> col_idx3, 
                  tapa::mmap<const float> b, 
                  tapa::mmap<float> x, 
                  int rows, int size1, int size2, int size3, int size4){

  // tapa::stream<float> b_stream1("b_stream1");
  // tapa::stream<float> b_stream2("b_stream2");
  // tapa::stream<float> b_stream3("b_stream3");
  tapa::streams<float, NU_PE+1> b_stream("b_stream");

  tapa::streams<float, NU_PE> val_stream("val_stream");
  tapa::streams<int, NU_PE> colIdx_stream("colIdx_stream");
  tapa::streams<bool, NU_PE> tlast_stream("tlast_stream");

  tapa::streams<float, NU_PE> x_resp_val_stream("x_resp_val_stream");

  tapa::streams<int, NU_PE> x_req_address_stream("x_req_address_stream");

  tapa::streams<int, NU_PE> x_solved_colIdx_stream("x_solved_colIdx_stream");
  tapa::streams<float, NU_PE> x_solved_val_stream("x_solved_val_stream");
  
  tapa::stream<float> x_out_stream("x_out_stream");


  tapa::task()
      .invoke(Mmap2Stream_b, b, b_stream[0], rows)
      .invoke<tapa::detach>(manage_x, x_out_stream, x_req_address_stream, x_solved_colIdx_stream, x_solved_val_stream, x_resp_val_stream, rows)
      // .invoke(Mmap2Stream_value, value0, val_stream[0], size1)
      // .invoke(Mmap2Stream_colIdx, col_idx0, colIdx_stream[0], size1)
      .invoke(Mmap2Stream_values, value0, col_idx0, val_stream[0], colIdx_stream[0], tlast_stream[0], size1)
      .invoke<tapa::detach>(sptrsv_PE, b_stream[0], val_stream[0], colIdx_stream[0], tlast_stream[0], x_resp_val_stream[0], b_stream[1], x_req_address_stream[0], x_solved_colIdx_stream[0], x_solved_val_stream[0], rows, size1)
      // .invoke(Mmap2Stream_value, value1, val_stream[1], size2)
      // .invoke(Mmap2Stream_colIdx, col_idx1, colIdx_stream[1], size2)
      .invoke(Mmap2Stream_values, value1, col_idx1, val_stream[1], colIdx_stream[1], tlast_stream[1], size2)
      .invoke<tapa::detach>(sptrsv_PE, b_stream[1], val_stream[1], colIdx_stream[1], tlast_stream[1], x_resp_val_stream[1], b_stream[2], x_req_address_stream[1], x_solved_colIdx_stream[1], x_solved_val_stream[1], rows, size2)

      .invoke(Mmap2Stream_values, value2, col_idx2, val_stream[2], colIdx_stream[2], tlast_stream[2], size3)
      .invoke<tapa::detach>(sptrsv_PE, b_stream[2], val_stream[2], colIdx_stream[2], tlast_stream[2], x_resp_val_stream[2], b_stream[3], x_req_address_stream[2], x_solved_colIdx_stream[2], x_solved_val_stream[2], rows, size3)

      .invoke(Mmap2Stream_values, value3, col_idx3, val_stream[3], colIdx_stream[3], tlast_stream[3], size4)
      .invoke<tapa::detach>(sptrsv_PE, b_stream[3], val_stream[3], colIdx_stream[3], tlast_stream[3], x_resp_val_stream[3], b_stream[4], x_req_address_stream[3], x_solved_colIdx_stream[3], x_solved_val_stream[3], rows, size4)

      .invoke(Stream2Mmap_solvedX, x, x_out_stream, rows)
      // .invoke(dummyBPass, b_out, b_stream3);
      .invoke(dummyBConsume, b_stream[4], rows);
}

