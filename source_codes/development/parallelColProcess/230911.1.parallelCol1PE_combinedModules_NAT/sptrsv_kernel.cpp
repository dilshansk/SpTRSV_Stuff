#include "sptrsv.h"


void Mmap2Stream_values(tapa::mmap<const t_WIDE> val_mmap, tapa::ostream<t_WIDE>& val_stream, unsigned int nnzs, unsigned int PEIdx) {

  for (uint64_t i = 0; i < nnzs; ++i) {
    // printf("PEIdx=%d, nnz=%d, i=%d\n",PEIdx, nnzs ,i);
    #pragma HLS PIPELINE II=1
    val_stream.write(val_mmap[i]);
  }
}

void Mmap2Stream_b(tapa::mmap<const float> b_mmap, tapa::ostream<float>& b_stream, unsigned int rows) {

  for (int i = 0; i < rows/NUM_PE; ++i) {
    #pragma HLS PIPELINE II=1
    b_stream << b_mmap[i];
  }
}

void Stream2Mmap_solvedX(tapa::mmap<float> solvedX, tapa::istream<float>& solvedX_stream, unsigned int rows){
  for (int i = 0; i < rows; ++i) {
    #pragma HLS PIPELINE II=1
    solvedX[i] = solvedX_stream.read();
    // printf("[S2MX]::idx=%d, val=%f\n", i, solvedX[i]);
  }
}


void splitVal(tapa::istream<t_WIDE>& i_valStream,
              tapa::ostreams<t_DW, CONCAT_FACTOR>& o_valStream){

  bool run = true;  
  bool foundDiagVal = false;
  t_WIDE wd_valSample;
  for (;;)
  {
    #pragma HLS PIPELINE II=1

    if ((!i_valStream.empty()))
    {
      i_valStream.try_read(wd_valSample);
    
      for (int i = 0; i < CONCAT_FACTOR; i++)
      {
        #pragma HLS UNROLL
        t_DW dw_valSample = wd_valSample.range(DATA_SIZE*(i+1)-1, DATA_SIZE*i);
        o_valStream[i].write(dw_valSample);       
      }
    }
  }
}

void FW_unit_0(tapa::istream<t_DW>& i_valStream,
            tapa::ostream<float>& o_nDiagValToPE,
            tapa::ostream<t_HW>& o_nDiagRowIdxRowEndToPE,
            tapa::ostream<bool>& o_nDiagDummyValStatToPE,
            tapa::ostream<t_DW>& o_xReqVal,
            tapa::ostream<float>& o_diagValToHandler,
            tapa::ostream<t_HW>& o_diagRowIdxToHandler,
            tapa::ostream<bool>& o_diagWaitStatToHandler,
            unsigned int PEIdx){

  bool processingDiagVal = false;
  
  t_DW inputData = 0;
  
  t_HW rowIdx = 0;
  t_HW colIdx = 0;
  t_HW rowIdxRowEnd = 0;
  unsigned int val_in_int = 0;
  float val_in_float = 0.0;

  t_HW diag_rowIdx = 0;
  unsigned int diag_val_in_int = 0;
  t_HW row_NZ = 0;

  unsigned int colIdxRowIdx = 0;

  t_DW xRequest = 0;

  for (;;)
  {
    #pragma HLS PIPELINE II=1
    if (!i_valStream.empty())
    {
      i_valStream.try_read(inputData);

      rowIdx = (inputData>>2) & 0x7FFF;
      colIdx = (inputData >> 17) & 0x7FFF;
      rowIdxRowEnd = (inputData>>1) & 0xFFFF;
      val_in_int = (inputData>>32) & 0xFFFFFFFF;
      val_in_float = (*(float *)(&val_in_int));

      if (rowIdx == colIdx) //diagonal value
      {
        if((rowIdxRowEnd & 0x1)==1){  //only diag val row
          o_diagWaitStatToHandler.write(false); //can proceed without waiting for non diag sum, don't wait
          o_diagValToHandler.write(val_in_float);
          o_diagRowIdxToHandler.write(rowIdx);
        }
        else{
          o_diagWaitStatToHandler.write(true); //can not proceed without waiting for non diag sum. wait
          o_diagValToHandler.write(val_in_float);
          o_diagRowIdxToHandler.write(rowIdx);
        }
        // o_diagValToHandler.write(val_in_float);
        // o_diagRowIdxToHandler.write(rowIdx);
      }
      else //Non diag value.
      {
        xRequest = 0;
        if (val_in_int == 0)  //it's a dummy non diag val 
        {
          o_nDiagDummyValStatToPE.write(true);
          o_nDiagValToPE.write(val_in_float);
          o_nDiagRowIdxRowEndToPE.write(rowIdxRowEnd);
          // xRequest = (1UL<<31) | (((t_DW)colIdx)<<16) | rowIdx;  //appending a flag indicating a dummy data
        }
        else{
          xRequest = (((t_DW)colIdx)<<16) | rowIdx;  
          xRequest = (((t_DW)PEIdx)<<32) | xRequest;
          o_nDiagDummyValStatToPE.write(false);
          o_nDiagValToPE.write(val_in_float);
          o_nDiagRowIdxRowEndToPE.write(rowIdxRowEnd);
          o_xReqVal.write(xRequest);
        }

        // xRequest = (((t_DW)PEIdx)<<32) | xRequest;

        // int colIdxRowIdx = (((int)colIdx)<<16) | rowIdx;
        // o_nDiagValToPE.write(val_in_float);
        // o_nDiagRowIdxRowEndToPE.write(rowIdxRowEnd);
        // o_xReqVal.write(xRequest);
      }
    }
  }
}

void FW_unit_1(tapa::istream<t_DW>& i_valStream,
            tapa::ostream<float>& o_nDiagValToPE,
            tapa::ostream<t_HW>& o_nDiagRowIdxRowEndToPE,
            tapa::ostream<bool>& o_nDiagDummyValStatToPE,
            tapa::ostream<t_DW>& o_xReqVal,
            unsigned int PEIdx){

  bool processingDiagVal = false;
  
  t_DW inputData = 0;
  
  t_HW rowIdx = 0;
  t_HW colIdx = 0;
  t_HW rowIdxRowEnd = 0;
  unsigned int val_in_int = 0;
  float val_in_float = 0.0;

  t_HW diag_rowIdx = 0;
  unsigned int diag_val_in_int = 0;
  t_HW row_NZ = 0;

  unsigned int colIdxRowIdx = 0;

  t_DW xRequest = 0;

  for (;;)
  {
    #pragma HLS PIPELINE II=1
    if (!i_valStream.empty())
    {
      i_valStream.try_read(inputData);

      if ((inputData & 0x1) == 0)
      {
        rowIdx = (inputData>>2) & 0x7FFF;
        colIdx = (inputData >> 17) & 0x7FFF;
        rowIdxRowEnd = (inputData>>1) & 0xFFFF;
        val_in_int = (inputData>>32) & 0xFFFFFFFF;
        val_in_float = (*(float *)(&val_in_int));

        xRequest = 0;
        if (val_in_int == 0)  //it's a dummy non diag val 
        {
          o_nDiagDummyValStatToPE.write(true);
          o_nDiagValToPE.write(val_in_float);
          o_nDiagRowIdxRowEndToPE.write(rowIdxRowEnd);
          // xRequest = (1UL<<31) | (((t_DW)colIdx)<<16) | rowIdx;  //appending a flag indicating a dummy data
        }
        else{
          xRequest = (((t_DW)colIdx)<<16) | rowIdx;  
          xRequest = (((t_DW)PEIdx)<<32) | xRequest;
          o_nDiagDummyValStatToPE.write(false);
          o_nDiagValToPE.write(val_in_float);
          o_nDiagRowIdxRowEndToPE.write(rowIdxRowEnd);
          o_xReqVal.write(xRequest);
        }

          // xRequest = (((t_DW)PEIdx)<<32) | xRequest;

          // int colIdxRowIdx = (((int)colIdx)<<16) | rowIdx;
          // o_nDiagValToPE.write(val_in_float);
          // o_nDiagRowIdxRowEndToPE.write(rowIdxRowEnd);
          // o_xReqVal.write(xRequest);
      }
    }
  }
}

void multiplyUnit(tapa::istream<bool>& i_dummyValStat,
                  tapa::istream<float>& i_valStream,
                  tapa::istream<t_HW>& i_rowIdxRowEndStream,
                  tapa::istream<float>& i_xRespVal,
                  tapa::ostream<t_DW>& o_lxValRowIdx,
                  unsigned int PEIdx, unsigned int mulIdx){
  
  unsigned int inputColIdxRowIdx = 0;
  t_DW xRequest = 0;
  float matVal = 0.0;
  float x_respVal = 0.0;
  float lxVal = 0.0;
  t_HW rowIdxRowEnd = 0;
  t_DW lxValRowIdx = 0;

  bool isDummyPeek;
  bool isDummy;

  // tapa::stream<float, MAT_VAL_FIFO_DEPTH> matValFifo;
  // tapa::stream<t_HW, MAT_VAL_FIFO_DEPTH> matRowIdxFifo;

  for(;;){
    #pragma HLS PIPELINE II=1

    // lxValRowIdx = 0;

    if(i_dummyValStat.try_peek(isDummyPeek)){

      if((!isDummyPeek) & (!i_valStream.empty()) & (!i_rowIdxRowEndStream.empty()) & (!i_xRespVal.empty())){
        i_valStream.try_read(matVal);
        i_rowIdxRowEndStream.try_read(rowIdxRowEnd);
        i_dummyValStat.try_read(isDummy);
        i_xRespVal.try_read(x_respVal);

        lxVal = matVal * x_respVal;
        unsigned int lxVal_in_int = (*(unsigned int *)(&lxVal));
        lxValRowIdx = (((t_DW)lxVal_in_int) << 32) | ((t_DW)(rowIdxRowEnd));
        // printf("[MUL-ACTUAL %d]::Before %d\n", mulIdx, (rowIdxRowEnd>>1)&0x7FFF);
        o_lxValRowIdx.write(lxValRowIdx);
        // printf("[MUL-ACTUAL %d]::After %d\n", mulIdx, (rowIdxRowEnd>>1)&0x7FFF);
      }
      else if((isDummyPeek) & (!i_valStream.empty()) & (!i_rowIdxRowEndStream.empty())){
        i_valStream.try_read(matVal);
        i_rowIdxRowEndStream.try_read(rowIdxRowEnd);
        i_dummyValStat.try_read(isDummy);
        lxValRowIdx = ((t_DW)(rowIdxRowEnd));

        // printf("[MUL-DUMMY %d]::Before %d\n", mulIdx, (rowIdxRowEnd>>1)&0x7FFF);
        o_lxValRowIdx.write(lxValRowIdx);
        // printf("[MUL-DUMMY %d]::After %d\n", mulIdx, (rowIdxRowEnd>>1)&0x7FFF);
      }
    }

    /*
    if ( (!i_valStream.empty()) & (!i_rowIdxRowEndStream.empty()) & (!i_xRespVal.empty()) ){
      i_valStream.try_read(matVal);
      i_rowIdxRowEndStream.try_read(rowIdxRowEnd);
      i_xRespVal.try_read(x_respVal);
      // if (rowIdx==3)
      // {
        // printf("[MUL%d-%d][RESP]::rowIdx=%d, matVal=%f, resp=%f\n", PEIdx, mulIdx, (rowIdxRowEnd>>1)&0x7FFF, matVal, x_respVal);
      // }
      lxVal = matVal * x_respVal;
      unsigned int lxVal_in_int = (*(unsigned int *)(&lxVal));
      lxValRowIdx = (((t_DW)lxVal_in_int) << 32) | ((t_DW)(rowIdxRowEnd));
      o_lxValRowIdx.write(lxValRowIdx);
    }
    */
  }
}

void multipliers_To_Accumulator_CrossBar(tapa::istreams<t_DW, 2>& i_streams,
                                        tapa::ostream<t_DW>& o_stream, unsigned int PEIdx){  //At the moment this is not a cross bar. Since we only have 2 input streams, and we process row by row, just adding them together.
  
  float val_sum = 0.0;
  t_HW numOfAccum = 0;

  t_DW inputSamples_0 = 0;
  t_DW inputSamples_1 = 0;

  t_HW rowIdxRowEnd_0 = 0;
  t_HW rowIdxRowEnd_1 = 0;

  unsigned int val_in_int_0 = 0;
  unsigned int val_in_int_1 = 0;

  float val_0 = 0.0;
  float val_1 = 0.0;

  t_DW outputVal = 0;

  // int temp_bothCounter=0;
  // int temp_0Counter=0;
  // int temp_1Counter=0;

  for (;;)
  {
    #pragma HLS PIPELINE II=1
    /*
    if (!(i_streams[0].empty()))
    {
      i_streams[0].try_read(inputSamples_0);
    }

    if (!(i_streams[1].empty()))
    {
      i_streams[1].try_read(inputSamples_1);
    }
    */

    if ( (!(i_streams[0].empty())) & (!(i_streams[1].empty())) )
    {
      i_streams[0].try_read(inputSamples_0);
      i_streams[1].try_read(inputSamples_1);

      rowIdxRowEnd_0 = inputSamples_0 & 0xFFFF;
      rowIdxRowEnd_1 = inputSamples_1 & 0xFFFF;
      
      val_in_int_0 = (inputSamples_0 >> 32) & 0xFFFFFFFF;
      val_in_int_1 = (inputSamples_1 >> 32) & 0xFFFFFFFF;

      val_0 = (*(float *)(&val_in_int_0));
      val_1 = (*(float *)(&val_in_int_1));

      val_sum = val_0 + val_1;

      // if (rowIdx_0==3)
      // {
        // printf("[CB][BOTH]::rowIdx=%d, val_0=%f, val_1=%f\n", rowIdx_0, val_0, val_1);
      // }
      

      // numOfAccum = 2;
      unsigned int val_sum_in_int = (*(unsigned int *)(&val_sum));
        
      // t_DW outputVal = (((t_DW)val_sum_in_int) << 32) | (((t_DW)numOfAccum) << 16) | rowIdx_0;
      t_DW outputVal = (((t_DW)val_sum_in_int) << 32) | rowIdxRowEnd_0;
      o_stream.write(outputVal);

      // temp_bothCounter+=1;
      // printf("[CB%d]::Both=%d\n", PEIdx, temp_bothCounter);
    }
    
    /*
    else if ((inputSamples_0!=0)){
      numOfAccum = 1;
      // t_HW temp_rowIdx = inputSamples_0 & 0xFFFF;
      // if (temp_rowIdx==3)
      // {
        // int temp_val_int = (inputSamples_0 >> 32) & 0xFFFFFFFF;
        // float temp_val_float = (*(float *)(&inputSamples_0));
        // printf("[CB][SINGLE_0]::rowIdx=%d val=%f\n", temp_rowIdx, temp_val_float);
      // }
      t_DW outputVal = inputSamples_0 | (((t_DW)numOfAccum) << 16);
      o_stream.write(outputVal);
      // temp_0Counter+=1;
      // printf("[CB%d]::Stream0=%d\n", PEIdx, temp_0Counter);
    }
    else if ((inputSamples_1!=0)){
      numOfAccum = 1;
      t_DW outputVal = inputSamples_1 | (((t_DW)numOfAccum) << 16);
      // t_HW temp_rowIdx = inputSamples_1 & 0xFFFF;
      // if (temp_rowIdx==3)
      // {
        // int temp_val_int = (inputSamples_1 >> 32) & 0xFFFFFFFF;
        // float temp_val_float = (*(float *)(&inputSamples_1));
        // printf("[CB][SINGLE_1]::rowIdx=%d val=%f\n", temp_rowIdx, temp_val_float);
      // }
      o_stream.write(outputVal);
      // temp_1Counter+=1;
      // printf("[CB%d]::Stream1=%d\n", PEIdx, temp_1Counter);
    }
    */




    /*
    if ( (!i_streams[0].empty()) & (!i_streams[1].empty()) )
    {

      i_streams[0].try_read(inputSamples_0);
      i_streams[1].try_read(inputSamples_1);

      rowIdx_0 = inputSamples_0 & 0xFFFF;
      rowIdx_1 = inputSamples_1 & 0xFFFF;
      
      val_in_int_0 = (inputSamples_0 >> 32) & 0xFFFFFFFF;
      val_in_int_1 = (inputSamples_1 >> 32) & 0xFFFFFFFF;

      val_0 = (*(float *)(&val_in_int_0));
      val_1 = (*(float *)(&val_in_int_1));

      val_sum = val_0 + val_1;

      // if (rowIdx_0==3)
      // {
        // printf("[CB][BOTH]::rowIdx=%d, val_0=%f, val_1=%f\n", rowIdx_0, val_0, val_1);
      // }
      

      numOfAccum = 2;
      int val_sum_in_int = (*(int *)(&val_sum));
        
      outputVal = (((t_DW)val_sum_in_int) << 32) | (((t_DW)numOfAccum) << 16) | rowIdx_0;
      o_stream.write(outputVal);
      temp_bothCounter+=1;
      printf("[CB%d]::Both=%d\n", PEIdx, temp_bothCounter);
    }
    else if ((!i_streams[0].empty())){
      i_streams[0].try_read(inputSamples_0);
      numOfAccum = 1;
      // t_HW temp_rowIdx = inputSamples_0 & 0xFFFF;
      // if (temp_rowIdx==3)
      // {
        // int temp_val_int = (inputSamples_0 >> 32) & 0xFFFFFFFF;
        // float temp_val_float = (*(float *)(&inputSamples_0));
        // printf("[CB][SINGLE_0]::rowIdx=%d val=%f\n", temp_rowIdx, temp_val_float);
      // }
      outputVal = inputSamples_0 | (((t_DW)numOfAccum) << 16);
      o_stream.write(outputVal);
      temp_0Counter+=1;
      printf("[CB%d]::Stream0=%d\n", PEIdx, temp_0Counter);
    }
    else if ((!i_streams[1].empty())){
      i_streams[1].try_read(inputSamples_1);
      numOfAccum = 1;
      outputVal = inputSamples_1 | (((t_DW)numOfAccum) << 16);
      // t_HW temp_rowIdx = inputSamples_1 & 0xFFFF;
      // if (temp_rowIdx==3)
      // {
        // int temp_val_int = (inputSamples_1 >> 32) & 0xFFFFFFFF;
        // float temp_val_float = (*(float *)(&inputSamples_1));
        // printf("[CB][SINGLE_1]::rowIdx=%d val=%f\n", temp_rowIdx, temp_val_float);
      // }
      o_stream.write(outputVal);
      temp_1Counter+=1;
      printf("[CB%d]::Stream1=%d\n", PEIdx, temp_1Counter);
    }
    */
  }
}

void accumulator(tapa::istream<t_DW>& i_valStream,
                tapa::ostream<float>& o_sumRespStream){
  
  float part_sum_val[5];
  t_HW part_sum_rowIdx[5];

  float adder_tree_val[5];
  #pragma HLS ARRAY_PARTITION variable=adder_tree_val type=complete
  t_HW adder_tree_rowIdx[5];
  #pragma HLS ARRAY_PARTITION variable=adder_tree_rowIdx type=complete

  t_DW dataSample = 0;
  t_HW rowIdxSample = 0;
  t_HW rowEndSample = 0;
  unsigned int sum_in_int = 0;
  float sum_in_float = 0.0;

  float nonDiagSumVal = 0.0;
  unsigned int reqSumVal_in_int = 0;
  t_HW reqSumAccum = 0;
  unsigned int peekReqRowNZIdx = 0;
  unsigned int ReqRowNZIdx = 0;
  t_HW peekRowIdx = 0;
  t_HW peekRowNZ = 0;
  t_DW fullResp = 0;

  int i = 0;

  // for (int i = 0; i < 5; i++)
  // {
  //   part_sum_val[i] = 0.0;
  //   part_sum_rowIdx[i] = 0xFFFF;

  //   adder_tree_val[i] = 0.0;
  //   adder_tree_rowIdx[i] = 0xFFFF;
  // }

  float accumVal = 0.0;
  float frwdAccumVal = 0.0;
  t_HW accumRowIdx = 0xFFFF;
  
  for (;;)
  {
    #pragma HLS PIPELINE II=1
    // #pragma HLS dependence variable=accumVal inter RAW true distance=6
    if (!i_valStream.empty())
    {
      i_valStream.try_read(dataSample);
      rowIdxSample = (dataSample>>1) & 0x7FFF;
      rowEndSample = dataSample & 0x1;
      sum_in_int = (dataSample>>32) & 0xFFFFFFFF;

      sum_in_float = (*(float *)(&sum_in_int));

      // printf("[ACCUM]::rowIdx=%d, rowEnd=%d, sum=%f\n", rowIdxSample, rowEndSample, sum_in_float);

      if (rowIdxSample == accumRowIdx)
      {
        accumVal = accumVal + sum_in_float;
        // #pragma HLS bind_op variable=accumVal op=fadd impl=fulldsp latency=4
        
      }
      else{
        accumVal = sum_in_float;
      }
      accumRowIdx = rowIdxSample;


      if (rowEndSample==1)  //Row end reached. Invoke adder tree to take the sum
      {
        o_sumRespStream.write(accumVal);
        // i = (i==4)?0:(i+1);
      }
    }
  }
}

// void diagValDecider(tapa::istreams<float, CONCAT_FACTOR>& i_diagVal,
//                     tapa::istreams<t_HW, CONCAT_FACTOR>& i_diagRowIdx,
//                     tapa::istreams<bool, CONCAT_FACTOR>& i_diagWaitStatus,
//                     tapa::ostream<float>& o_diagVal,
//                     tapa::ostream<t_HW>& o_diagRowIdx,
//                     tapa::ostream<bool>& o_diagWaitStatus,
//                     unsigned int PEIdx){

//   t_HW expectingRow = PEIdx & 0xFFFF;

//   t_HW peekRowIdx = 0;

//   t_HW diagRowIdxSample = 0;
//   t_HW diagRowNZSample = 0;
//   float diagValSample = 0.0;
//   bool diagWaitStatus = true;

//   unsigned int diagRowNZIdxCombine = 0;

//   for (; ; )
//   {
//     if ((!i_diagRowIdx[0].empty()) & (!i_diagVal[0].empty()) & (!i_diagWaitStatus[0].empty()))
//     {
//       // i_diagRowIdx[0].try_peek(peekRowIdx);
//       // if (peekRowIdx==expectingRow)
//       // {
//         i_diagRowIdx[0].try_read(diagRowIdxSample);
//         i_diagVal[0].try_read(diagValSample);
//         i_diagWaitStatus[0].try_read(diagWaitStatus);

//         // printf("[DIAGDECIDER]::rowIdx=%d, val=%f\n", diagRowIdxSample, diagValSample);

//         o_diagVal.write(diagValSample);
//         o_diagRowIdx.write(diagRowIdxSample);
//         o_diagWaitStatus.write(diagWaitStatus);

//       // }
//     }
//   }
  
// }

void diagValHandler(tapa::istream<float>& i_bStream,
                    tapa::istream<float>& i_diagVal,
                    tapa::istream<t_HW>& i_diagRowIdx,
                    tapa::istream<bool>& i_diagWiatStatus,
                    tapa::istream<float>& i_sumRespStream,
                    tapa::ostream<t_HW>& o_rowIdx,
                    tapa::ostream<float>& o_xSolved,
                    unsigned int PEIdx){

  unsigned int diagRowNZIdxSample = 0;
  float diagValSample = 0.0;
  t_HW diagRowIdxSample = 0;
  float sumRespSample = 0.0;
  float bVal = 0.0;
  float modifiedBVal = 0.0;
  float xSolved = 0.0;

  t_HW diagRowNZ = 0;

  bool requested = false;

  bool peekWaitStatus = true;
  bool waitStatusSample = true;

  for (;;)
  {
    #pragma HLS PIPELINE II=1
    if(i_diagWiatStatus.try_peek(peekWaitStatus)){

      if ( (peekWaitStatus) & (!i_diagVal.empty()) & (!i_diagRowIdx.empty()) & (!i_sumRespStream.empty()) & (!i_bStream.empty())){

        i_diagVal.try_read(diagValSample);
        i_diagRowIdx.try_read(diagRowIdxSample);
        i_sumRespStream.try_read(sumRespSample);
        i_diagWiatStatus.try_read(waitStatusSample);
        i_bStream.try_read(bVal);

        modifiedBVal = bVal - sumRespSample;
        xSolved = modifiedBVal/diagValSample;

        // printf("[DIAG_HANDL]::rowIdx=%d, diagVal=%f, resp=%f\n", diagRowIdxSample, diagValSample, sumRespSample);

        o_rowIdx.write(diagRowIdxSample);
        o_xSolved.write(xSolved);
      }
      else if ( (!peekWaitStatus) & (!i_diagVal.empty()) & (!i_diagRowIdx.empty()) & (!i_bStream.empty())){
        i_diagVal.try_read(diagValSample);
        i_diagRowIdx.try_read(diagRowIdxSample);
        i_diagWiatStatus.try_read(waitStatusSample);
        i_bStream.try_read(bVal);

        xSolved = bVal/diagValSample;
        // printf("[DIAG_HANDL]::rowIdx=%d, diagVal=%f, resp=%f\n", diagRowIdxSample, diagValSample, sumRespSample);

        o_rowIdx.write(diagRowIdxSample);
        o_xSolved.write(xSolved);
      }
    }

  /*
    if ((!i_diagVal.empty()) & (!i_diagRowIdx.empty()) & (!i_sumRespStream.empty()))
    {
      i_diagVal.try_read(diagValSample);
      i_diagRowIdx.try_read(diagRowIdxSample);
      i_sumRespStream.try_read(sumRespSample);

      o_rowIdxToDiv.write(diagRowIdxSample);
      o_diagValToDiv.write(diagValSample);
      o_nonDiagSumToSub.write(sumRespSample);
      o_rowProcIndicator.write(true);
    }
    else if ((!i_diagRowNZIdx.empty()) & (!i_diagVal.empty()) & (!i_diagRowIdx.empty()))
    {
      i_diagRowNZIdx.try_read(diagRowNZIdxSample);
      diagRowNZ = diagRowNZIdxSample >> 16;
      if (diagRowNZ==1)
      {
        i_diagVal.try_read(diagValSample);
        i_diagRowIdx.try_read(diagRowIdxSample);

        o_rowIdxToDiv.write(diagRowIdxSample);
        o_diagValToDiv.write(diagValSample);
        o_nonDiagSumToSub.write(0.0);
        o_rowProcIndicator.write(true);
      }
      else{
        o_sumReqStream.write(diagRowNZIdxSample);
      }
    }
    */
    
  }
}

/*
void diagValHandler(tapa::istreams<float, CONCAT_FACTOR>& i_diagVal,
                    tapa::istreams<t_HW, CONCAT_FACTOR>& i_diagRowIdx,
                    tapa::istreams<t_HW, CONCAT_FACTOR>& i_diagRowNZ,
                    tapa::istream<t_DW>& i_sumRespStream,
                    tapa::ostream<bool>& o_rowProcIndicator,
                    tapa::ostream<t_HW>& o_sumReqStream,
                    tapa::ostream<t_HW>& o_rowIdxToDiv,
                    tapa::ostream<float>& o_diagValToDiv,
                    tapa::ostream<float>& o_nonDiagSumToSub,
                    int PEIdx){

  // t_DW diagDataSample[CONCAT_FACTOR];
  // t_HW diagDataRowIdx[CONCAT_FACTOR];
  // int diagDataVal_in_int[CONCAT_FACTOR];
  // float diagDataVal[CONCAT_FACTOR];
  // t_HW nonDiagNum[CONCAT_FACTOR];

  // t_DW peekSamples[CONCAT_FACTOR];
  t_HW peekRowIdx[CONCAT_FACTOR];

  t_HW selectedStream;
  t_HW processingRowIdx;
  t_HW processingNonDiagNum;
  float processingDiagVal;

  bool diagValProcessed = true;

  t_DW responseSample;
  t_HW responseNumAccum;
  int responseAccumSum_in_int;
  float responseAccumSum;

  // tapa::streams<float, CONCAT_FACTOR> diagValsFIFO;
  // tapa::streams<t_HW, CONCAT_FACTOR> diagValRowIdxFIFO;
  // tapa::streams<t_HW, CONCAT_FACTOR> nonDiagNumFIFO;

  t_HW expectingRow = PEIdx & 0xFFFF;

  for (;;)
  {  
    #pragma HLS PIPELINE II=1

    for (int j = 0; j < CONCAT_FACTOR; j++)
    {
      #pragma HLS UNROLL
      if ((!i_diagRowIdx[j].empty()) & (!i_diagRowNZ[j].empty()) & (!i_diagVal[j].empty()) & (diagValProcessed))
      {
        i_diagRowIdx[j].try_read(processingRowIdx);
        i_diagRowNZ[j].try_read(processingNonDiagNum);
        i_diagVal[j].try_read(processingDiagVal);
        diagValProcessed &= false;
        // printf("[TargetIdx]::rowIdx=%d, nonDiagNum=%d, DiagVal=%f\n", processingRowIdx, processingNonDiagNum, processingDiagVal);
      }
    }
    

    // if(diagValProcessed){
    // for (int j = 0; j < CONCAT_FACTOR; j++)
    // {
    //   #pragma HLS UNROLL
    //   if ((!i_diagRowIdx[j].empty()) & (!i_diagRowNZ[j].empty()) & (!i_diagVal[j].empty()) & (diagValProcessed))
    //   {
    //     t_HW peekedRowIdx;
    //     i_diagRowIdx[j].try_peek(peekedRowIdx);
    //     if (peekedRowIdx==expectingRow)
    //     {
    //       i_diagRowIdx[j].try_read(processingRowIdx);
    //       i_diagRowNZ[j].try_read(processingNonDiagNum);
    //       i_diagVal[j].try_read(processingDiagVal);
    //       diagValProcessed = false;

    //       // printf("[TargetIdx]::rowIdx=%d, nonDiagNum=%d, DiagVal=%f\n", processingRowIdx, processingNonDiagNum, processingDiagVal);
    //     }
    //   }
    // }
    // // }
    
    
    // if ((!i_diagRowIdx[0].empty()) & (!i_diagRowIdx[1].empty()) & (diagValProcessed))
    // {
    //   i_diagRowIdx[0].try_peek(peekRowIdx[0]);
    //   i_diagRowIdx[1].try_peek(peekRowIdx[1]);

    //   if(peekRowIdx[0] < peekRowIdx[1]){
    //     selectedStream = 0;
    //   }
    //   else{
    //     selectedStream = 1;
    //   }
    //   processingRowIdx = i_diagRowIdx[selectedStream].read();
    //   processingNonDiagNum = i_diagRowNZ[selectedStream].read();
    //   processingDiagVal = i_diagVal[selectedStream].read();
    //   diagValProcessed = false;

    //   printf("[Both]::rowIdx1=%d, rowIdx2=%d, processingRowIdx=%d, processingVal=%f\n",peekRowIdx[0],peekRowIdx[1],processingRowIdx,processingDiagVal);
    // }
    // else if ((!i_diagRowIdx[0].empty()) & (diagValProcessed) ){
    //   selectedStream = 0;
    //   processingRowIdx = i_diagRowIdx[selectedStream].read();
    //   processingNonDiagNum = i_diagRowNZ[selectedStream].read();
    //   processingDiagVal = i_diagVal[selectedStream].read();
    //   diagValProcessed = false;
    // }
    // else if ((!i_diagRowIdx[1].empty()) & (diagValProcessed) ){
    //   selectedStream = 1;
    //   processingRowIdx = i_diagRowIdx[selectedStream].read();
    //   processingNonDiagNum = i_diagRowNZ[selectedStream].read();
    //   processingDiagVal = i_diagVal[selectedStream].read();
    //   diagValProcessed = false;
    // }

    if(!diagValProcessed){
      if (processingNonDiagNum==1)
      {
        // expectingRow+=NUM_PE;
        o_nonDiagSumToSub.write(0.0);
        o_diagValToDiv.write(processingDiagVal);
        o_rowIdxToDiv.write(processingRowIdx);
        o_rowProcIndicator.write(true);
        diagValProcessed = true;
        // printf("[DiagHanlder][Case1]::rowIdx=%d, val=%f\n",processingRowIdx,processingDiagVal);
      }
      else{
        o_sumReqStream.write(processingRowIdx);
        // printf("[DiagHandler]::Send req = %d\n",processingRowIdx);
        responseSample = i_sumRespStream.read();
        // printf("[DiagHandler]::Rec resp accum = %d\n",responseSample & 0xFFFF);
        responseNumAccum = responseSample & 0xFFFF;
        // int temp_val_int = (responseSample >> 32) & 0xFFFFFFFF;
        // float temp_val = (*(float *)(&temp_val_int));
        // if (processingRowIdx==3)
        // {
          // printf("[DiagHandler]::reqIdx=%d, respAccum=%d, respVal=%f, Target=%d\n", processingRowIdx, responseNumAccum, temp_val, processingNonDiagNum);
        // }

        if (responseNumAccum == (processingNonDiagNum-1))
        {
          responseAccumSum_in_int = (responseSample>>32) & 0xFFFFFFFF;
          responseAccumSum = (*(float *)(&responseAccumSum_in_int));
          o_nonDiagSumToSub.write(responseAccumSum);
          o_diagValToDiv.write(processingDiagVal);
          o_rowIdxToDiv.write(processingRowIdx);
          o_rowProcIndicator.write(true);
          diagValProcessed = true;
          // expectingRow+=NUM_PE;
        }
      }
    }
  }
}
*/
/*
void subDivUnit(tapa::istream<float>& i_bStream,
                tapa::istream<float>& i_nonDiagSum,
                tapa::istream<float>& i_diagVal,
                tapa::istream<t_HW>& i_rowIdx,
                tapa::ostream<t_HW>& o_rowIdx,
                tapa::ostream<float>& o_xSolved){
  
  float b_val = 0.0;
  float nonDiagSum = 0.0;
  float diagVal = 0.0;
  t_HW rowIdx = 0;
  float xSolved = 0.0;
  float modifiedB = 0.0;

  for (;;)
  {
    #pragma HLS PIPELINE II=1

    if ((!i_bStream.empty()) & (!i_nonDiagSum.empty()) & (!i_diagVal.empty()) & (!i_rowIdx.empty()))
    {
      i_bStream.try_read(b_val);
      i_nonDiagSum.try_read(nonDiagSum);
      i_diagVal.try_read(diagVal);
      i_rowIdx.try_read(rowIdx);

      // printf("[div unit]::rowIdx=%d, b_val=%f, nonDiagSum=%f, diagVal=%f\n",rowIdx, b_val, nonDiagSum, diagVal);

      modifiedB = b_val - nonDiagSum;
      xSolved = modifiedB/diagVal;

      o_rowIdx.write(rowIdx);
      o_xSolved.write(xSolved);
    }
  }
}
*/

/*
void PEsToArbFilter(tapa::istream<t_HW>& i_latestRowIdx,
                    tapa::istreams<t_DW, NUM_PE>& i_PE_req,
                    tapa::ostreams<t_DW, NUM_PE>& o_filtered_PE_req, unsigned int idx){

  t_HW latestRowIdx = 0;
  t_HW colIdxFilter = 0;

  bool filterEnable = false;

  for (;;)
  {
    #pragma HLS PIPELINE II=1

    if (!i_latestRowIdx.empty())
    {
      i_latestRowIdx.try_read(latestRowIdx);
      colIdxFilter = latestRowIdx;
      filterEnable = true;
      // printf("[ARB]::latestRowIdx=%d\n", latestRowIdx);
    }

    for (int j = 0; j < NUM_PE; j++)
    {
      #pragma HLS UNROLL
      t_DW peekReqSample;
      t_HW peekColIdx;
      t_DW reqSample;
      if ((!i_PE_req[j].empty()))
      {
        i_PE_req[j].try_peek(peekReqSample);
        peekColIdx = (peekReqSample>>16) & 0x7FFF;
        if (((peekColIdx<=colIdxFilter) & (filterEnable)) | (((peekReqSample>>31) & 0x1)==1))
        {
          // printf("[FILTER %d]::rowIdx=%d, colIdx=%d, PEIdx=%d\n", idx, (unsigned int)(peekReqSample&0xFFFF), (unsigned int)((peekReqSample>>16)&0x7FFF), (unsigned int)((peekReqSample>>32)&0xFFFFFFFF));
          i_PE_req[j].try_read(reqSample);
          o_filtered_PE_req[j].write(reqSample & 0xFFFFFFFF7FFFFFFF);
        }
      }
    }
  }
}

void PEsToBankForward(tapa::istreams<t_DW, NUM_PE>& i_PE_req,
                      tapa::ostream<t_DW>& o_mem_req){
  
  t_DW peekSamples[NUM_PE];
  t_HW peekRowIdx[NUM_PE];

  bool emptyStatus[NUM_PE];

  t_DW reqFIFOSample = 0;

  // t_HW latestRowIdx;

  // tapa::streams<t_DW, NUM_PE> filtered_PE_req_FIFO;

  for (int i = 0; i < NUM_PE; i++)
  {
    emptyStatus[i] = false;
    peekSamples[i] = 0;
    peekRowIdx[i] = 0;
  }
  

  for (;;)
  {
    #pragma HLS PIPELINE II=1

    if ((!i_PE_req[0].empty()))
    {
      emptyStatus[0] = true;
    }

    if ((!i_PE_req[1].empty()))
    {
      emptyStatus[1] = true;
    }
    

    if ((emptyStatus[0]) & (emptyStatus[1]))
    {
      i_PE_req[0].try_peek(peekSamples[0]);
      i_PE_req[1].try_peek(peekSamples[1]);

      peekRowIdx[0] = peekSamples[0] & 0xFFFF;
      peekRowIdx[1] = peekSamples[1] & 0xFFFF;

      if (peekRowIdx[0] < peekRowIdx[1]) //zero is going
      {
        i_PE_req[0].try_read(reqFIFOSample);
        o_mem_req.write(reqFIFOSample);
      }
      else{
        i_PE_req[1].try_read(reqFIFOSample);
        o_mem_req.write(reqFIFOSample);
      }

      emptyStatus[0] = false;
      emptyStatus[1] = false;

    }
    else if (emptyStatus[0])
    {
      i_PE_req[0].try_read(reqFIFOSample);
      o_mem_req.write(reqFIFOSample);
      emptyStatus[0] = false;
    }
    else if (emptyStatus[1])
    {
      i_PE_req[1].try_read(reqFIFOSample);
      o_mem_req.write(reqFIFOSample);
      emptyStatus[1] = false;
    }
  }
}
*/

void memBank(tapa::istream<bool>& i_loopExit,
            tapa::istreams<t_DW, NUM_PE>& i_memRequest,
            tapa::istream<t_HW>& i_xSolved_rowIdx,
            tapa::istream<float>& i_xSolved_val,
            tapa::ostreams<float, NUM_PE>& o_memResponse,
            tapa::ostream<float>& o_xFinal, unsigned int size, unsigned int idx){


  // bool x_flag[MEM_BANK_SIZE];
  float x[MEM_BANK_SIZE];
  #pragma HLS bind_storage variable=x type=RAM_2P impl=bram

  // t_DW peekReqSample;
  // t_HW peekColIdx;
  // bool peekFlag;

  t_DW reqSample = 0;
  t_HW reqColIdx = 0;
  t_HW reqPEIdx = 0;
  float respVal = 0.0;
  unsigned int respVal_in_int = 0;
  t_DW fullResp = 0;

  t_HW xSolved_rowIdx_sample = 0;
  float xSolved_val_sample = 0.0;

  t_HW finalReadCounter = 0;
  // bool finalReadFlag;
  float finalReadVal = 0.0;
  bool finalSendingDone = false;

  t_HW lastSolvedIdx = 0;
  t_HW lastSolvedRowIdx = 0;

  // bool memWritten_status_buf[2];
  // t_HW memWriteRowIdx_buf[2];
  // t_HW flagUpdateRowIdx;

  bool memReadEnable = false;

  bool emptyStatus[NUM_PE];
  t_DW peekSamples[NUM_PE];
  t_HW peekRowIdx[NUM_PE];
  t_HW peekColIdx[NUM_PE];

  // for (int i = 0; i < 2; i++)
  // {
  //   memWritten_status_buf[i] = false;
  // }
  

  // for (int i = 0; i < MEM_BANK_SIZE; i++)
  // {
  //   #pragma HLS PIPELINE II=1
  //   x_flag[i] = false;
  // }
  
  /*
  for (int i = 0; i < MEM_BANK_SIZE; i++)
  {
    x[i] = 0.0;
  }
  */

  bool loopExit = false;

 for (int i = 0; i < NUM_PE; i++)
  {
    emptyStatus[i] = false;
    peekSamples[i] = 0;
    peekRowIdx[i] = 0;
    peekColIdx[i] = 0;
  }
  

  for (;(!loopExit);)
  {
    #pragma HLS PIPELINE II=1

    // printf("This is mem loop\n");

    //checking loop exit
    if(!i_loopExit.empty()){
      i_loopExit.try_read(loopExit);
    }

    //reading
    if ((memReadEnable))
    {
      if ((!i_memRequest[0].empty()))
      {
        emptyStatus[0] = true;
      }

      if ((!i_memRequest[1].empty()))
      {
        emptyStatus[1] = true;
      }
      

      if ((emptyStatus[0]) & (emptyStatus[1]))
      {
        i_memRequest[0].try_peek(peekSamples[0]);
        i_memRequest[1].try_peek(peekSamples[1]);

        peekRowIdx[0] = peekSamples[0] & 0xFFFF;
        peekRowIdx[1] = peekSamples[1] & 0xFFFF;
        peekColIdx[0] = (peekSamples[0]>>16) & 0x7FFF;
        peekColIdx[1] = (peekSamples[1]>>16) & 0x7FFF;

        if ((peekColIdx[0]<=lastSolvedRowIdx) & (peekColIdx[1]<=lastSolvedRowIdx))
        {
          if ((peekRowIdx[0] < peekRowIdx[1])){
            i_memRequest[0].try_read(reqSample);
            reqColIdx = (reqSample >> 16) & 0xFFFF;
            
            respVal = x[reqColIdx>>(NUM_PE-1)];
            o_memResponse[0].write(respVal);
          }
          else{
            i_memRequest[1].try_read(reqSample);
            reqColIdx = (reqSample >> 16) & 0xFFFF;
            
            respVal = x[reqColIdx>>(NUM_PE-1)];
            o_memResponse[1].write(respVal);
          }
        }
        else if ((peekColIdx[0]<=lastSolvedRowIdx)){
          i_memRequest[0].try_read(reqSample);
          reqColIdx = (reqSample >> 16) & 0xFFFF;
          
          respVal = x[reqColIdx>>(NUM_PE-1)];
          o_memResponse[0].write(respVal);
        }
        else if ((peekColIdx[1]<=lastSolvedRowIdx)){
          i_memRequest[1].try_read(reqSample);
          reqColIdx = (reqSample >> 16) & 0xFFFF;
          
          respVal = x[reqColIdx>>(NUM_PE-1)];
          o_memResponse[1].write(respVal);
        }
        emptyStatus[0] = false;
        emptyStatus[1] = false;
      }
      else if (emptyStatus[0])
      {
        i_memRequest[0].try_peek(peekSamples[0]);
        peekColIdx[0] = (peekSamples[0]>>16) & 0x7FFF;
        if ((peekColIdx[0]<=lastSolvedRowIdx)){
          i_memRequest[0].try_read(reqSample);
          reqColIdx = (reqSample >> 16) & 0xFFFF;
          
          respVal = x[reqColIdx>>(NUM_PE-1)];
          o_memResponse[0].write(respVal);
          emptyStatus[0] = false;
        }
      }
      else if (emptyStatus[1])
      {
        i_memRequest[1].try_peek(peekSamples[1]);
        peekColIdx[1] = (peekSamples[1]>>16) & 0x7FFF;
        if ((peekColIdx[1]<=lastSolvedRowIdx)){
          i_memRequest[1].try_read(reqSample);
          reqColIdx = (reqSample >> 16) & 0xFFFF;
          
          respVal = x[reqColIdx>>(NUM_PE-1)];
          o_memResponse[1].write(respVal);
          emptyStatus[1] = false;
        }
      }
    }
    // if (!i_memRequest.empty() & memReadEnable)
    /*
    if (!i_memRequest.empty())
    {
      // i_memRequest.try_peek(peekReqSample);
      // peekColIdx = (peekReqSample >> 16) & 0xFFFF;
      // peekFlag = x_flag[peekColIdx>>(NUM_PE-1)];

      // prunsigned intf("[MEM][PEEK]::PeekColIdx=%d, solved?=%d\n", peekColIdx, peekFlag);
      
      // if (peekFlag){
      i_memRequest.try_read(reqSample);
      reqColIdx = (reqSample >> 16) & 0xFFFF;
      reqPEIdx = (reqSample >> 32) & 0xFFFF;
      
      respVal = x[reqColIdx>>(NUM_PE-1)];
      respVal_in_int = (*(unsigned int *)(&respVal));

      fullResp = ((t_DW)respVal_in_int << 32) | reqPEIdx;
      o_memResponse.write(fullResp);


      // printf("[MEM %d][READ]::rowIdx=%d, reqColIdx=%d, PEIdx=%d, val=%f\n", idx, reqSample&0xFFFF , reqColIdx, (int)(fullResp&0xFFFF), respVal);
      // printf("[%d]::req=%d\n", idx, reqColIdx);

      // t_HW tempRowIdx =   reqSample & 0xFFFF;

        // printf("[MEM][READ]::rowIdx=%d, reqColIdx=%d, reqPEIdx=%d, respVal=%f\n", tempRowIdx, reqColIdx, reqPEIdx, respVal);
      // }
    }
    */
    // else{ //if it is not read try sending value to output
    //   // int counterVal = finalReadCounter;
    //   if (finalReadCounter<(size/NUM_PE))
    //   {
    //     finalReadFlag = x_flag[finalReadCounter];
    //     if (finalReadFlag)
    //     {
    //       finalReadVal = x[finalReadCounter];
    //       o_xFinal.write(finalReadVal);
    //       finalReadCounter = finalReadCounter + 1;
    //       // if (o_xFinal.try_write(finalReadVal))
    //       // {
    //       //   finalReadCounter = counterVal + 1;
    //       // }
    //     }
    //   }
    // }

    //value writing
    if((!i_xSolved_rowIdx.empty()) & (!i_xSolved_val.empty())){
      i_xSolved_rowIdx.try_read(xSolved_rowIdx_sample);
      i_xSolved_val.try_read(xSolved_val_sample);

      x[xSolved_rowIdx_sample>>(NUM_PE-1)] = xSolved_val_sample;
      lastSolvedIdx = (xSolved_rowIdx_sample>>(NUM_PE-1));
      // printf("[Mem][Write]::row=%d, val=%f\n", xSolved_rowIdx_sample, xSolved_val_sample);
      lastSolvedRowIdx = xSolved_rowIdx_sample;
      memReadEnable = true;
      finalSendingDone = false;

      // printf("[%d]::write=%d\n", idx, xSolved_rowIdx_sample>>(NUM_PE-1));

      // memWritten_status_buf[i&0x1] = true;
      // memWriteRowIdx_buf[i&0x1] = xSolved_rowIdx_sample;

      // if (x[xSolved_rowIdx_sample>>(NUM_PE-1)] == xSolved_val_sample)
      // {
      //   x_flag[xSolved_rowIdx_sample>>(NUM_PE-1)] = true;
      // }

      // o_latestRowIdx.write(xSolved_rowIdx_sample);
    }
    // else if ((finalReadCounter<(size/NUM_PE)) & (finalReadCounter<=lastSolvedIdx) & (memReadEnable) & (!finalSendingDone)) //if it is not read try sending value to output. as far as we are solving rows one after one, checking the whether read counter is below last solved Idx is fine. Or else need flags, if go random
    else if ((finalReadCounter<=lastSolvedIdx) & (memReadEnable) & (!finalSendingDone)) //if it is not read try sending value to output. as far as we are solving rows one after one, checking the whether read counter is below last solved Idx is fine. Or else need flags, if go random
    {
      finalReadVal = x[finalReadCounter];
      if(o_xFinal.try_write(finalReadVal)){
        // printf("[%d]::final=%d\n", idx, finalReadCounter);
        if (finalReadCounter==(size/NUM_PE-1))
        {
          finalReadCounter = 0;
          finalSendingDone = true;
        }
        else{
          finalReadCounter = finalReadCounter + 1;
        }
    }

      /*
      finalReadFlag = x_flag[finalReadCounter];
      if (finalReadFlag)
      {
        finalReadVal = x[finalReadCounter];
        o_xFinal.write(finalReadVal);
        finalReadCounter = finalReadCounter + 1;
        // if (o_xFinal.try_write(finalReadVal))
        // {
        //   finalReadCounter = counterVal + 1;
        // }
      }

      */
    }

    /*Since we stream b, the way we solve rows are pretty much in order. Therefore, no need to maunsigned intain a flag array. All the mem requests are 
    also filtered based on the last solved val
    */
    // //Flag update  
    // if (memWritten_status_buf[((i+1)&0x1)])
    // {
    //   flagUpdateRowIdx = memWriteRowIdx_buf[((i+1)&0x1)];
    //   x_flag[flagUpdateRowIdx>>(NUM_PE-1)] = true;
    //   o_latestRowIdx.write(flagUpdateRowIdx);
    //   memWritten_status_buf[((i+1)&0x1)] = false;
    // }
    
  }
}

/*
void memBankToPE(tapa::istream<t_DW>& i_response,
                tapa::ostreams<float, NUM_PE>& o_response, unsigned int idx){

  t_DW responseSample = 0;
  t_HW PEIdx = 0;
  unsigned int val_in_int = 0;
  float val = 0.0;

  for (;;)
  {
    #pragma HLS PIPELINE II=1
    if ((!i_response.empty()))
    {
      i_response.try_read(responseSample);
      PEIdx = responseSample & 0xFFFF;
      val_in_int = (responseSample >> 32) & 0xFFFFFFFF;
      val = (*(float *)(&val_in_int));
      o_response[PEIdx&0x1].write(val);
      // printf("[BTOPE %d]::targetPE=%d, val=%f\n", idx, PEIdx&0x1, val);
    }
  }
}
*/

void xCombine(tapa::istreams<float, NUM_PE>& i_x_from_mem,
              tapa::ostream<float>& o_x,
              tapa::ostream<bool>& o_doneSignal,
              unsigned int size){
  
  float readVal = 0.0;

  for (int i = 0; i < size/NUM_PE; i++)
  {
    for (int j = 0; j < NUM_PE; j++)
    {
      #pragma HLS PIPELINE II=1
      readVal = i_x_from_mem[j].read();
      // printf("[combine]::readVal=%f\n", readVal);
      o_x.write(readVal);
    }
  }

  o_doneSignal.write(true);

}

void memExitHandler(tapa::istream<bool>& i_doneSignal,
                    tapa::ostreams<bool, NUM_PE>& o_loopExit){

  bool doneSignal;

  doneSignal = i_doneSignal.read();

  for(int i=0; i<NUM_PE; i++){
    #pragma HLS UNROLL
    o_loopExit[i].write(doneSignal);
  }

}

void sptrsv_PE(tapa::istream<t_WIDE>& i_val_stream,
              tapa::istream<float>& i_b_stream,
              tapa::istream<float>& i_x_resp_from_mem0,
              tapa::istream<float>& i_x_resp_from_mem1,
              tapa::ostream<t_DW>& o_x_req_to_mem0,
              tapa::ostream<t_DW>& o_x_req_to_mem1,
              tapa::ostream<t_HW>& o_xSolved_rowIdx,
              tapa::ostream<float>& o_xSolved_val,
              unsigned int PEIdx
              ){

  tapa::streams<t_DW, CONCAT_FACTOR, GEN_FIFO_DEPTH> valStreams_split_to_FWUnit("valStreams_split_to_FWUnit");
  tapa::streams<float, CONCAT_FACTOR, MAT_VAL_FIFO_DEPTH> nDiagVal_FW_to_mul("nDiagVal_FW_to_mul");
  tapa::streams<t_HW, CONCAT_FACTOR, MAT_VAL_FIFO_DEPTH> nDiagRowIdxRowEnd_FW_to_mul("nDiagRowIdxRowEnd_FW_to_mul");
  tapa::streams<bool, CONCAT_FACTOR, MAT_VAL_FIFO_DEPTH> nDiagDummyValStat_to_mul("nDiagDummyValStat_to_mul");
  // tapa::streams<unsigned int, CONCAT_FACTOR, GEN_FIFO_DEPTH> nDiagColIdxRowIdx_FW_to_mul("nDiagColIdxRowIdx_FW_to_mul");
  tapa::stream<float, GEN_FIFO_DEPTH> diagVal_FW_to_diagHandler("diagVal_FW_to_diagHandler");
  tapa::stream<t_HW, GEN_FIFO_DEPTH> diagRowIdx_FW_to_diagHandler("diagRowIdx_FW_to_diagHandler");
  tapa::stream<bool, GEN_FIFO_DEPTH> diagWaitStatus_FW_to_diagHandler("diagWaitStatus_FW_to_diagHandler");
  // tapa::stream<float> diagRowVal_diagDecider_to_diagHandler("diagRowVal_diagDecider_to_diagHandler");
  // tapa::stream<t_HW> diagRowIdx_diagDecider_to_diagHandler("diagRowIdx_diagDecider_to_diagHandler");
  // tapa::stream<bool> diagWaitStatus_diagDecider_to_diagHandler("diagWaitStatus_diagDecider_to_diagHandler");
  tapa::streams<t_DW, CONCAT_FACTOR, GEN_FIFO_DEPTH> lx_mul_to_cb("lx_mul_to_cb");
  tapa::stream<t_DW, GEN_FIFO_DEPTH> sum_cb_to_accum("sum_cb_to_accum");
  tapa::stream<float, GEN_FIFO_DEPTH> sumResp_accum_to_diagHandler("sumResp_accum_to_diagHandler");
  // tapa::stream<t_HW, GEN_FIFO_DEPTH> rowIdx_diagHandler_to_div("rowIdx_diagHandler_to_div");
  // tapa::stream<float, GEN_FIFO_DEPTH> diagVal_diagHanlder_to_div("diagVal_diagHanlder_to_div");
  // tapa::stream<float, GEN_FIFO_DEPTH> nDiagSum_diagHandler_to_sub("nDiagSum_diagHandler_to_sub");

  tapa::task()
    .invoke<tapa::detach>(splitVal, i_val_stream, valStreams_split_to_FWUnit)
    .invoke<tapa::detach>(FW_unit_0, valStreams_split_to_FWUnit[0], nDiagVal_FW_to_mul[0], nDiagRowIdxRowEnd_FW_to_mul[0], nDiagDummyValStat_to_mul[0], o_x_req_to_mem0, diagVal_FW_to_diagHandler, diagRowIdx_FW_to_diagHandler, diagWaitStatus_FW_to_diagHandler, PEIdx)
    .invoke<tapa::detach>(FW_unit_1, valStreams_split_to_FWUnit[1], nDiagVal_FW_to_mul[1], nDiagRowIdxRowEnd_FW_to_mul[1], nDiagDummyValStat_to_mul[1], o_x_req_to_mem1, PEIdx)
    .invoke<tapa::detach>(multiplyUnit, nDiagDummyValStat_to_mul[0], nDiagVal_FW_to_mul[0], nDiagRowIdxRowEnd_FW_to_mul[0], i_x_resp_from_mem0, lx_mul_to_cb[0], PEIdx, 0)
    .invoke<tapa::detach>(multiplyUnit, nDiagDummyValStat_to_mul[1], nDiagVal_FW_to_mul[1], nDiagRowIdxRowEnd_FW_to_mul[1], i_x_resp_from_mem1, lx_mul_to_cb[1], PEIdx, 1)
    .invoke<tapa::detach>(multipliers_To_Accumulator_CrossBar, lx_mul_to_cb, sum_cb_to_accum, PEIdx)
    .invoke<tapa::detach>(accumulator, sum_cb_to_accum, sumResp_accum_to_diagHandler)
    // .invoke<tapa::detach>(diagValDecider, diagVal_FW_to_diagDecider, diagRowIdx_FW_to_diagDecider, diagWaitStatus_FW_to_diagDecider, diagRowVal_diagDecider_to_diagHandler, diagRowIdx_diagDecider_to_diagHandler, diagWaitStatus_diagDecider_to_diagHandler, PEIdx)
    .invoke<tapa::detach>(diagValHandler, i_b_stream, diagVal_FW_to_diagHandler, diagRowIdx_FW_to_diagHandler, diagWaitStatus_FW_to_diagHandler, sumResp_accum_to_diagHandler, o_xSolved_rowIdx, o_xSolved_val, PEIdx);
}

void sptrsv_kernel(tapa::mmap<const t_WIDE> value0,
                  tapa::mmap<const t_WIDE> value1,
                  tapa::mmap<const float> bval0,
                  tapa::mmap<const float> bval1,
                  tapa::mmap<float> xSolved, 
                  unsigned int nnz0, unsigned int nnz1, unsigned int size){

  tapa::streams<t_WIDE, NUM_PE, GEN_FIFO_DEPTH> valStreams("valStreams");
  tapa::streams<float, NUM_PE, GEN_FIFO_DEPTH> bStreams("bStreams");
  tapa::streams<t_DW, NUM_PE, GEN_FIFO_DEPTH> xReq_PEs_to_mem0("xReq_PEs_to_mem0");
  tapa::streams<t_DW, NUM_PE, GEN_FIFO_DEPTH> xReq_PEs_to_mem1("xReq_PEs_to_mem1");
  // tapa::streams<t_DW, NUM_PE, GEN_FIFO_DEPTH> xReq_filter0_to_mem0("xReq_filter0_to_mem0");
  // tapa::streams<t_DW, NUM_PE, GEN_FIFO_DEPTH> xReq_filter1_to_mem1("xReq_filter1_to_mem1");
  // tapa::streams<t_HW, NUM_PE, GEN_FIFO_DEPTH> latestSolvedIdx_mem_to_arb("latestSolvedIdx_mem_to_arb");
  #ifndef __SYNTHESIS__
    tapa::streams<float, NUM_PE, 2048> xResp_mem0_to_PEs("xResp_mem0_to_PEs");
    tapa::streams<float, NUM_PE, 2048> xResp_mem1_to_PEs("xResp_mem1_to_PEs");
  #else
    tapa::streams<float, NUM_PE, GEN_FIFO_DEPTH> xResp_mem0_to_PEs("xResp_mem0_to_PEs");
    tapa::streams<float, NUM_PE, GEN_FIFO_DEPTH> xResp_mem1_to_PEs("xResp_mem1_to_PEs");
  #endif
  tapa::streams<t_HW, NUM_PE, GEN_FIFO_DEPTH> xSolved_rowIdx("xSolved_rowIdx");
  tapa::streams<float, NUM_PE, GEN_FIFO_DEPTH> xSolved_val("xSolved_val");
  // tapa::stream<t_DW, GEN_FIFO_DEPTH> memReq_arb0_to_mem0("memReq_arb0_to_mem0");
  // tapa::stream<t_DW, GEN_FIFO_DEPTH> memReq_arb1_to_mem1("memReq_arb1_to_mem1");
  // tapa::stream<t_DW, GEN_FIFO_DEPTH> memResp_mem0_to_arb0("memResp_mem0_to_arb0");
  // tapa::stream<t_DW, GEN_FIFO_DEPTH> memResp_mem1_to_arb1("memResp_mem1_to_arb1");
  tapa::streams<float, NUM_PE, GEN_FIFO_DEPTH> xFinal_mem_to_combine("xFinal_mem_to_combine");
  tapa::stream<float, GEN_FIFO_DEPTH> xFinal_comb_to_store("xFinal_comb_to_store");
  tapa::stream<bool> doneSignal_comb_to_exitHandler("doneSignal_comb_to_exitHandler");
  tapa::streams<bool, NUM_PE> loopExit_exitHandler_to_mem("loopExit_exitHandler_to_mem");

  tapa::task()
    .invoke(Mmap2Stream_values, value0, valStreams[0], nnz0, 0)
    .invoke(Mmap2Stream_values, value1, valStreams[1], nnz1, 1)
    .invoke(Mmap2Stream_b, bval0, bStreams[0], size)
    .invoke(Mmap2Stream_b, bval1, bStreams[1], size)
    .invoke<tapa::detach>(sptrsv_PE, valStreams[0], bStreams[0], xResp_mem0_to_PEs[0], xResp_mem1_to_PEs[0], xReq_PEs_to_mem0[0], xReq_PEs_to_mem1[0], xSolved_rowIdx[0], xSolved_val[0], 0)
    .invoke<tapa::detach>(sptrsv_PE, valStreams[1], bStreams[1], xResp_mem0_to_PEs[1], xResp_mem1_to_PEs[1], xReq_PEs_to_mem0[1], xReq_PEs_to_mem1[1], xSolved_rowIdx[1], xSolved_val[1], 1)
    .invoke<tapa::detach>(memBank, loopExit_exitHandler_to_mem[0], xReq_PEs_to_mem0, xSolved_rowIdx[0], xSolved_val[0], xResp_mem0_to_PEs, xFinal_mem_to_combine[0], size, 0)
    .invoke<tapa::detach>(memBank, loopExit_exitHandler_to_mem[1], xReq_PEs_to_mem1, xSolved_rowIdx[1], xSolved_val[1], xResp_mem1_to_PEs, xFinal_mem_to_combine[1], size, 1)
    .invoke(xCombine, xFinal_mem_to_combine, xFinal_comb_to_store, doneSignal_comb_to_exitHandler, size)
    .invoke(memExitHandler, doneSignal_comb_to_exitHandler, loopExit_exitHandler_to_mem)
    .invoke(Stream2Mmap_solvedX, xSolved, xFinal_comb_to_store, size);
}

