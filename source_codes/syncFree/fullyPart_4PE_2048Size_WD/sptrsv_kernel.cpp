#include "sptrsv.h"


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


void sptrsv_PE(tapa::istream<float>& val_stream, 
              tapa::istream<int>& colIdx_stream,
              tapa::istream<bool>& tlast_stream,
              tapa::istream<float>& x_resp,
              tapa::ostream<int>& x_req_address,
              tapa::ostream<int>& solvedX_colIdx,
              tapa::ostream<float>& solvedX_val,
              int rows,
              int nnzs
              ){



  //to keep track of the row number
  int rowNumber = 0;

  //To fascilitate intermediate variable as of solved X
  float solvedX = 0.0;

  //to hold data sampled from data streams
  float valSample = 0.0;
  int colIdxSample = 0;
  bool tlast_val = false;

  //To track x values
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
      b_val = valSample;
      rowNumber = colIdxSample;

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
        
      }
      
    }
  }
  

}


void smallX(tapa::istream<bool>& sm_loopCondition,
            tapa::istream<float>& wr_val,
            tapa::istream<int>& wr_idx,
            tapa::istream<int>& rd_idx_cluster0,
            tapa::ostream<float>& rd_val_cluster0,
            tapa::istream<int>& rd_idx_cluster1,
            tapa::ostream<float>& rd_val_cluster1,
            tapa::istream<int>& rd_idx_cluster2,
            tapa::ostream<float>& rd_val_cluster2,
            tapa::istream<int>& rd_idx_cluster3,
            tapa::ostream<float>& rd_val_cluster3,
            tapa::ostream<bool>& sol_status,
            tapa::ostream<float>& final_out, int rows){
  
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
    
    //read from clustor 0
    int rd_idx_sample_cluster0;
    float rd_val_sample_cluster_0;
    int peekedReqAddress_cluster_0;
    bool isPeeked_cluster_0;

    if ( !rd_idx_cluster0.empty() )
    {
      isPeeked_cluster_0 = rd_idx_cluster0.try_peek(peekedReqAddress_cluster_0); //peek the request
      if( (isPeeked_cluster_0) && (((part_x_flag.range(peekedReqAddress_cluster_0,peekedReqAddress_cluster_0)))==(t_bit)1) ){ //peek is successful & requested x is solved
        rd_idx_cluster0.try_read(rd_idx_sample_cluster0);
        rd_val_sample_cluster_0 = part_x[rd_idx_sample_cluster0];
        rd_val_cluster0.write(rd_val_sample_cluster_0);
      }
    }

    //read from clustor 1
    int rd_idx_sample_cluster1;
    float rd_val_sample_cluster_1;
    int peekedReqAddress_cluster_1;
    bool isPeeked_cluster_1;

    if ( !rd_idx_cluster1.empty() )
    {
      isPeeked_cluster_1 = rd_idx_cluster1.try_peek(peekedReqAddress_cluster_1); //peek the request
      if( (isPeeked_cluster_1) && (((part_x_flag.range(peekedReqAddress_cluster_1,peekedReqAddress_cluster_1)))==(t_bit)1) ){ //peek is successful & requested x is solved
        rd_idx_cluster1.try_read(rd_idx_sample_cluster1);
        rd_val_sample_cluster_1 = part_x[rd_idx_sample_cluster1];
        rd_val_cluster1.write(rd_val_sample_cluster_1);
      }
    }

    //read from clustor 2
    int rd_idx_sample_cluster2;
    float rd_val_sample_cluster_2;
    int peekedReqAddress_cluster_2;
    bool isPeeked_cluster_2;

    if ( !rd_idx_cluster2.empty() )
    {
      isPeeked_cluster_2 = rd_idx_cluster2.try_peek(peekedReqAddress_cluster_2); //peek the request
      if( (isPeeked_cluster_2) && (((part_x_flag.range(peekedReqAddress_cluster_2,peekedReqAddress_cluster_2)))==(t_bit)1) ){ //peek is successful & requested x is solved
        rd_idx_cluster2.try_read(rd_idx_sample_cluster2);
        rd_val_sample_cluster_2 = part_x[rd_idx_sample_cluster2];
        rd_val_cluster2.write(rd_val_sample_cluster_2);
      }
    }


    //read from clustor 3
    int rd_idx_sample_cluster3;
    float rd_val_sample_cluster_3;
    int peekedReqAddress_cluster_3;
    bool isPeeked_cluster_3;

    if ( !rd_idx_cluster3.empty() )
    {
      isPeeked_cluster_3 = rd_idx_cluster3.try_peek(peekedReqAddress_cluster_3); //peek the request
      if( (isPeeked_cluster_3) && (((part_x_flag.range(peekedReqAddress_cluster_3,peekedReqAddress_cluster_3)))==(t_bit)1) ){ //peek is successful & requested x is solved
        rd_idx_cluster3.try_read(rd_idx_sample_cluster3);
        rd_val_sample_cluster_3 = part_x[rd_idx_sample_cluster3];
        rd_val_cluster3.write(rd_val_sample_cluster_3);
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

void readXPerCluster(tapa::istream<bool>& rd_loopConidtion,
              tapa::istream<int>& x_req_address,
              tapa::istream<float>& mem_0_vals,
              tapa::istream<float>& mem_1_vals,
              tapa::istream<float>& mem_2_vals,
              tapa::istream<float>& mem_3_vals,
              tapa::ostream<float>& x_resp_val,
              tapa::ostream<int>& mem_0_req_addr,
              tapa::ostream<int>& mem_1_req_addr,
              tapa::ostream<int>& mem_2_req_addr,
              tapa::ostream<int>& mem_3_req_addr
                ){

  bool mem0Requested = false;
  bool mem1Requested = false;
  bool mem2Requested = false;
  bool mem3Requested = false;
  
  bool loopCondition = false;

  for (; !loopCondition ; )
  {
    #pragma HLS PIPELINE II=1

    if ( !rd_loopConidtion.empty() )
    {
      rd_loopConidtion.try_read(loopCondition);
    }

    int actualAddr;
    if ((!x_req_address.empty()) & (!mem0Requested) & (!mem1Requested) & (!mem2Requested) & (!mem3Requested))
    {
      x_req_address.try_read(actualAddr);
      if(actualAddr%NU_SMALL_MEMS==0){      //Need changes if NU_SMALL_MEMS is not equal to 2
        int requestingAddr = actualAddr/NU_SMALL_MEMS;
        mem_0_req_addr.write(requestingAddr);
        mem0Requested = true;
      }
      else if (actualAddr%NU_SMALL_MEMS==1) {
        int requestingAddr = (actualAddr-1)/NU_SMALL_MEMS;
        mem_1_req_addr.write(requestingAddr);
        mem1Requested = true;
      }
      else if (actualAddr%NU_SMALL_MEMS==2) {
        int requestingAddr = (actualAddr-2)/NU_SMALL_MEMS;
        mem_2_req_addr.write(requestingAddr);
        mem2Requested = true;
      }
      else if (actualAddr%NU_SMALL_MEMS==3) {
        int requestingAddr = (actualAddr-3)/NU_SMALL_MEMS;
        mem_3_req_addr.write(requestingAddr);
        mem3Requested = true;
      }
    }

    float val_sample;
    if ((!mem_0_vals.empty()) & (mem0Requested))
    {
      mem_0_vals.try_read(val_sample);
      x_resp_val.write(val_sample);
      mem0Requested = false;
    }
    else if ((!mem_1_vals.empty()) & (mem1Requested))
    {
      mem_1_vals.try_read(val_sample);
      x_resp_val.write(val_sample);
      mem1Requested = false;
    }
    else if ((!mem_2_vals.empty()) & (mem2Requested))
    {
      mem_2_vals.try_read(val_sample);
      x_resp_val.write(val_sample);
      mem2Requested = false;
    }
    else if ((!mem_3_vals.empty()) & (mem3Requested))
    {
      mem_3_vals.try_read(val_sample);
      x_resp_val.write(val_sample);
      mem3Requested = false;
    }

  }
}

void writeXPerCluster(tapa::istream<bool>& wr_loopCondition,
                tapa::istream<int>& x_solved_colIdx,
                tapa::istream<float>& x_solved_val,
                tapa::ostream<int>& wr_colIdx,
                tapa::ostream<float>& wr_val, int c_idx
                ){

  bool loopCondition = false;

  for (; !loopCondition ; )
  {
    #pragma HLS PIPELINE II=1

    if( !wr_loopCondition.empty() ){
      wr_loopCondition.try_read(loopCondition);
    }

    int colIdx;
    int modified_colIdx;
    float val;
    if( (!x_solved_colIdx.empty()) & (!x_solved_val.empty()) ){
      x_solved_colIdx.try_read(colIdx);
      x_solved_val.try_read(val);

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
      wr_colIdx.write(modified_colIdx);
      wr_val.write(val);
    }
  }

}

void controlSmallMem(tapa::istreams<bool, NU_SMALL_MEMS>& mem_solved,
                    tapa::istreams<float, NU_SMALL_MEMS>& mem_vals,
                    tapa::ostreams<bool, NU_SMALL_MEMS>& wr_terminate,
                    tapa::ostreams<bool, NU_SMALL_MEMS>& rd_terminate,
                    tapa::ostreams<bool, NU_SMALL_MEMS>& mem_terminate,
                    tapa::ostream<float>& x_out_stream, int rows){

  bool mem0Done = false;
  bool mem1Done = false;
  bool mem2Done = false;
  bool mem3Done = false;
  bool allMemDone = false;

  
  wr_terminate[0].write(false);
  wr_terminate[1].write(false);
  wr_terminate[2].write(false);
  wr_terminate[3].write(false);

  rd_terminate[0].write(false);
  rd_terminate[1].write(false);
  rd_terminate[2].write(false);
  rd_terminate[3].write(false);

  mem_terminate[0].write(false);
  mem_terminate[1].write(false);
  mem_terminate[2].write(false);
  mem_terminate[3].write(false);

  for ( ; !allMemDone ; )
  {
    #pragma HLS PIPELINE II=1

    if ( !mem_solved[0].empty() )
    {
      mem_solved[0].try_read(mem0Done);
    }

    if ( !mem_solved[1].empty() )
    {
      mem_solved[1].try_read(mem1Done);
    }

    if ( !mem_solved[2].empty() )
    {
      mem_solved[2].try_read(mem2Done);
    }

    if ( !mem_solved[3].empty() )
    {
      mem_solved[3].try_read(mem3Done);
    }

    allMemDone = mem0Done & mem1Done & mem2Done & mem3Done;

  }

  

  wr_terminate[0].write(true);
  wr_terminate[1].write(true);
  wr_terminate[2].write(true);
  wr_terminate[3].write(true);
  
  rd_terminate[0].write(true);
  rd_terminate[1].write(true);
  rd_terminate[2].write(true);
  rd_terminate[3].write(true);

  mem_terminate[0].write(true);
  mem_terminate[1].write(true);
  mem_terminate[2].write(true);
  mem_terminate[3].write(true);


  

  for (int i = 0; i < rows; i++)
  {
    #pragma HLS PIPELINE II=1

    if(i%NU_SMALL_MEMS==0){
      float mem0_val_sample = mem_vals[0].read();
      x_out_stream.write(mem0_val_sample);
    }
    else if(i%NU_SMALL_MEMS==1){
      float mem1_val_sample = mem_vals[1].read();
      x_out_stream.write(mem1_val_sample);
    }
    else if(i%NU_SMALL_MEMS==2){
      float mem2_val_sample = mem_vals[2].read();
      x_out_stream.write(mem2_val_sample);
    }
    else if(i%NU_SMALL_MEMS==3){
      float mem3_val_sample = mem_vals[3].read();
      x_out_stream.write(mem3_val_sample);
    }
  }

}

/*
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
*/

// void sptrsv_kernel(float* value, int* row, int* col, float* b, float* x)
void sptrsv_kernel(tapa::mmap<const float> value0, 
                  tapa::mmap<const int> col_idx0, 
                  tapa::mmap<const float> value1, 
                  tapa::mmap<const int> col_idx1, 
                  tapa::mmap<const float> value2, 
                  tapa::mmap<const int> col_idx2, 
                  tapa::mmap<const float> value3, 
                  tapa::mmap<const int> col_idx3,  
                  tapa::mmap<float> x, 
                  int rows, int size1, int size2, int size3, int size4){

  // tapa::stream<float> b_stream1("b_stream1");
  // tapa::stream<float> b_stream2("b_stream2");
  // tapa::stream<float> b_stream3("b_stream3");
  // tapa::streams<float, NU_PE+1> b_stream("b_stream");

  tapa::streams<float, NU_PE> val_stream("val_stream");
  tapa::streams<int, NU_PE> colIdx_stream("colIdx_stream");
  tapa::streams<bool, NU_PE> tlast_stream("tlast_stream");

  tapa::streams<float, NU_PE> x_resp_val_stream("x_resp_val_stream");

  tapa::streams<int, NU_PE> x_req_address_stream("x_req_address_stream");

  tapa::streams<int, NU_PE> x_solved_colIdx_stream("x_solved_colIdx_stream");
  tapa::streams<float, NU_PE> x_solved_val_stream("x_solved_val_stream");


  tapa::streams<bool, NU_SMALL_MEMS> mem_solved_status("mem_sol_stat_stream");
  tapa::streams<float, NU_SMALL_MEMS> mem_final_vals("mem_final_vals_stream");
  tapa::streams<bool, NU_SMALL_MEMS> wr_terminate_signals("wr_terminate_stream");
  tapa::streams<bool, NU_SMALL_MEMS> rd_terminate_signals("rd_terminate_stream");
  tapa::streams<bool, NU_SMALL_MEMS> small_mem_terminate("mem_terminate_stream");
  tapa::stream<float> final_x_out("final_x_stream");

  tapa::stream<int> wr_short_colIdx_c0("wr_short_colIdx_c0_stream");
  tapa::stream<int> wr_short_colIdx_c1("wr_short_colIdx_c1_stream");
  tapa::stream<int> wr_short_colIdx_c2("wr_short_colIdx_c2_stream");
  tapa::stream<int> wr_short_colIdx_c3("wr_short_colIdx_c3_stream");

  tapa::stream<float> wr_val_c0("wr_val_c0_stream");
  tapa::stream<float> wr_val_c1("wr_val_c1_stream");
  tapa::stream<float> wr_val_c2("wr_val_c2_stream");
  tapa::stream<float> wr_val_c3("wr_val_c3_stream");

  tapa::stream<float> mem0_to_c0_vals("mem0_to_c0_val_stream");
  tapa::stream<float> mem1_to_c0_vals("mem1_to_c0_val_stream");
  tapa::stream<float> mem2_to_c0_vals("mem2_to_c0_val_stream");
  tapa::stream<float> mem3_to_c0_vals("mem3_to_c0_val_stream");

  tapa::stream<float> mem0_to_c1_vals("mem0_to_c1_val_stream");
  tapa::stream<float> mem1_to_c1_vals("mem1_to_c1_val_stream");
  tapa::stream<float> mem2_to_c1_vals("mem2_to_c1_val_stream");
  tapa::stream<float> mem3_to_c1_vals("mem3_to_c1_val_stream");

  tapa::stream<float> mem0_to_c2_vals("mem0_to_c2_val_stream");
  tapa::stream<float> mem1_to_c2_vals("mem1_to_c2_val_stream");
  tapa::stream<float> mem2_to_c2_vals("mem2_to_c2_val_stream");
  tapa::stream<float> mem3_to_c2_vals("mem3_to_c2_val_stream");

  tapa::stream<float> mem0_to_c3_vals("mem0_to_c3_val_stream");
  tapa::stream<float> mem1_to_c3_vals("mem1_to_c3_val_stream");
  tapa::stream<float> mem2_to_c3_vals("mem2_to_c3_val_stream");
  tapa::stream<float> mem3_to_c3_vals("mem3_to_c3_val_stream");

  tapa::stream<int> c0_to_mem0_rdIdx("c0_to_mem0_rdIdx_stream");
  tapa::stream<int> c0_to_mem1_rdIdx("c0_to_mem1_rdIdx_stream");
  tapa::stream<int> c0_to_mem2_rdIdx("c0_to_mem2_rdIdx_stream");
  tapa::stream<int> c0_to_mem3_rdIdx("c0_to_mem3_rdIdx_stream");

  tapa::stream<int> c1_to_mem0_rdIdx("c1_to_mem0_rdIdx_stream");
  tapa::stream<int> c1_to_mem1_rdIdx("c1_to_mem1_rdIdx_stream");
  tapa::stream<int> c1_to_mem2_rdIdx("c1_to_mem2_rdIdx_stream");
  tapa::stream<int> c1_to_mem3_rdIdx("c1_to_mem3_rdIdx_stream");

  tapa::stream<int> c2_to_mem0_rdIdx("c2_to_mem0_rdIdx_stream");
  tapa::stream<int> c2_to_mem1_rdIdx("c2_to_mem1_rdIdx_stream");
  tapa::stream<int> c2_to_mem2_rdIdx("c2_to_mem2_rdIdx_stream");
  tapa::stream<int> c2_to_mem3_rdIdx("c2_to_mem3_rdIdx_stream");

  tapa::stream<int> c3_to_mem0_rdIdx("c3_to_mem0_rdIdx_stream");
  tapa::stream<int> c3_to_mem1_rdIdx("c3_to_mem1_rdIdx_stream");
  tapa::stream<int> c3_to_mem2_rdIdx("c3_to_mem2_rdIdx_stream");
  tapa::stream<int> c3_to_mem3_rdIdx("c3_to_mem3_rdIdx_stream");

  tapa::task()
      .invoke(controlSmallMem, mem_solved_status, mem_final_vals, wr_terminate_signals, rd_terminate_signals, small_mem_terminate, final_x_out, rows)
      .invoke<tapa::detach>(smallX, small_mem_terminate[0], wr_val_c0, wr_short_colIdx_c0, c0_to_mem0_rdIdx, mem0_to_c0_vals, c1_to_mem0_rdIdx, mem0_to_c1_vals, c2_to_mem0_rdIdx, mem0_to_c2_vals, c3_to_mem0_rdIdx, mem0_to_c3_vals, mem_solved_status[0], mem_final_vals[0], rows)
      .invoke<tapa::detach>(readXPerCluster, rd_terminate_signals[0], x_req_address_stream[0], mem0_to_c0_vals, mem1_to_c0_vals, mem2_to_c0_vals, mem3_to_c0_vals, x_resp_val_stream[0], c0_to_mem0_rdIdx, c0_to_mem1_rdIdx, c0_to_mem2_rdIdx, c0_to_mem3_rdIdx)
      .invoke<tapa::detach>(writeXPerCluster, wr_terminate_signals[0], x_solved_colIdx_stream[0],  x_solved_val_stream[0], wr_short_colIdx_c0, wr_val_c0, 0)

      .invoke<tapa::detach>(smallX, small_mem_terminate[1], wr_val_c1, wr_short_colIdx_c1, c0_to_mem1_rdIdx, mem1_to_c0_vals, c1_to_mem1_rdIdx, mem1_to_c1_vals, c2_to_mem1_rdIdx, mem1_to_c2_vals, c3_to_mem1_rdIdx, mem1_to_c3_vals, mem_solved_status[1], mem_final_vals[1], rows)
      .invoke<tapa::detach>(readXPerCluster, rd_terminate_signals[1], x_req_address_stream[1], mem0_to_c1_vals, mem1_to_c1_vals, mem2_to_c1_vals, mem3_to_c1_vals, x_resp_val_stream[1], c1_to_mem0_rdIdx, c1_to_mem1_rdIdx, c1_to_mem2_rdIdx, c1_to_mem3_rdIdx)
      .invoke<tapa::detach>(writeXPerCluster, wr_terminate_signals[1], x_solved_colIdx_stream[1],  x_solved_val_stream[1], wr_short_colIdx_c1, wr_val_c1, 1)

      .invoke<tapa::detach>(smallX, small_mem_terminate[2], wr_val_c2, wr_short_colIdx_c2, c0_to_mem2_rdIdx, mem2_to_c0_vals, c1_to_mem2_rdIdx, mem2_to_c1_vals, c2_to_mem2_rdIdx, mem2_to_c2_vals, c3_to_mem2_rdIdx, mem2_to_c3_vals, mem_solved_status[2], mem_final_vals[2], rows)
      .invoke<tapa::detach>(readXPerCluster, rd_terminate_signals[2], x_req_address_stream[2], mem0_to_c2_vals, mem1_to_c2_vals, mem2_to_c2_vals, mem3_to_c2_vals, x_resp_val_stream[2], c2_to_mem0_rdIdx, c2_to_mem1_rdIdx, c2_to_mem2_rdIdx, c2_to_mem3_rdIdx)
      .invoke<tapa::detach>(writeXPerCluster, wr_terminate_signals[2], x_solved_colIdx_stream[2],  x_solved_val_stream[2], wr_short_colIdx_c2, wr_val_c2, 2)

      .invoke<tapa::detach>(smallX, small_mem_terminate[3], wr_val_c3, wr_short_colIdx_c3, c0_to_mem3_rdIdx, mem3_to_c0_vals, c1_to_mem3_rdIdx, mem3_to_c1_vals, c2_to_mem3_rdIdx, mem3_to_c2_vals, c3_to_mem3_rdIdx, mem3_to_c3_vals, mem_solved_status[3], mem_final_vals[3], rows)
      .invoke<tapa::detach>(readXPerCluster, rd_terminate_signals[3], x_req_address_stream[3], mem0_to_c3_vals, mem1_to_c3_vals, mem2_to_c3_vals, mem3_to_c3_vals, x_resp_val_stream[3], c3_to_mem0_rdIdx, c3_to_mem1_rdIdx, c3_to_mem2_rdIdx, c3_to_mem3_rdIdx)
      .invoke<tapa::detach>(writeXPerCluster, wr_terminate_signals[3], x_solved_colIdx_stream[3],  x_solved_val_stream[3], wr_short_colIdx_c3, wr_val_c3, 3
      )

      .invoke(Mmap2Stream_values, value0, col_idx0, val_stream[0], colIdx_stream[0], tlast_stream[0], size1)
      .invoke<tapa::detach>(sptrsv_PE, val_stream[0], colIdx_stream[0], tlast_stream[0], x_resp_val_stream[0], x_req_address_stream[0], x_solved_colIdx_stream[0], x_solved_val_stream[0], rows, size1)

      .invoke(Mmap2Stream_values, value1, col_idx1, val_stream[1], colIdx_stream[1], tlast_stream[1], size2)
      .invoke<tapa::detach>(sptrsv_PE, val_stream[1], colIdx_stream[1], tlast_stream[1], x_resp_val_stream[1], x_req_address_stream[1], x_solved_colIdx_stream[1], x_solved_val_stream[1], rows, size2)

      .invoke(Mmap2Stream_values, value2, col_idx2, val_stream[2], colIdx_stream[2], tlast_stream[2], size3)
      .invoke<tapa::detach>(sptrsv_PE, val_stream[2], colIdx_stream[2], tlast_stream[2], x_resp_val_stream[2], x_req_address_stream[2], x_solved_colIdx_stream[2], x_solved_val_stream[2], rows, size3)

      .invoke(Mmap2Stream_values, value3, col_idx3, val_stream[3], colIdx_stream[3], tlast_stream[3], size4)
      .invoke<tapa::detach>(sptrsv_PE, val_stream[3], colIdx_stream[3], tlast_stream[3], x_resp_val_stream[3], x_req_address_stream[3], x_solved_colIdx_stream[3], x_solved_val_stream[3], rows, size4)

      .invoke(Stream2Mmap_solvedX, x, final_x_out, rows);
}

