rm -rf sptrsv
rm -rf sptrsv.hw.xo
rm -rf tapa_workdir
rm -rf sptrsv.hw_generate_bitstream.sh

platform=xilinx_u280_xdma_201920_3

g++ sptrsv_kernel.cpp sptrsv_test.cpp  -o sptrsv  -O2 -ltapa -lfrt -lglog -lgflags -lOpenCL -I/local-scratch/Xilinx/Vitis_HLS/2021.2/include/


##Synthesis to RTL###
tapac -o sptrsv.hw.xo sptrsv_kernel.cpp \
  --platform $platform \
  --top sptrsv_kernel \
  --work-dir tapa_workdir \
  --clock-period 4

##To run HW simulation(HW_EMU)

# v++ -o sptrsv.hw_emu.xclbin \
#   --link \
#   --target hw_emu \
#   --kernel sptrsv_kernel \
#   --platform $platform \
#   sptrsv.hw.xo



###Tapa+Autobridge command###
# tapac -o sptrsv.hw.xo sptrsv_kernel.cpp \
#        --platform xilinx_u280_xdma_201920_3 \
#        --top sptrsv_kernel \
#        --work-dir tapa_workdir \
#        --connectivity sptrsv.ini \
#        --floorplan-output constraint.tcl \
#        --read-only-args "value" \
#        --read-only-args "col_idx" \
#        --read-only-args "b" \
#        --wite-only-args "x" \
#        --enable-floorplan \
#        --enable-hbm-binding-adjustment \
#        --min-area-limit 0.45 \
#        --max-area-limit 0.65
