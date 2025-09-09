COMPILER_FLAGS = -O2 -ltapa -lfrt -lglog -lgflags -lOpenCL 
#-fsanitize=address -g

VITIS_HLS_INCL_DIR = /localssd/Xilinx/Vitis_HLS/2021.2/include/

platform = xilinx_u280_xdma_201920_3

gcc_compile : sptrsv_kernel.cpp sptrsv_test.cpp
	g++ sptrsv_kernel.cpp sptrsv_test.cpp  -o sptrsv  $(COMPILER_FLAGS) -I$(VITIS_HLS_INCL_DIR)
	./sptrsv

tapa_synth : sptrsv_kernel.cpp
	tapac -o sptrsv.hw.xo sptrsv_kernel.cpp \
  	--platform $(platform) \
  	--top sptrsv_kernel \
  	--work-dir tapa_workdir \
  	--clock-period 4

tapa_hw_emu : sptrsv.hw.xo
	./sptrsv --bitstream=sptrsv.hw.xo

vitis_hw_emu : sptrsv.hw.xo
	v++ -o sptrsv.hw_emu.xclbin \
	  --link \
	  --target hw_emu \
	  --kernel sptrsv_kernel \
	  --platform $(platform) \
	  sptrsv.hw.xo
	./sptrsv --bitstream=sptrsv.hw_emu.xclbin

tapa_full : sptrsv_kernel.cpp
	tapac -o sptrsv.hw.xo sptrsv_kernel.cpp \
	       --platform xilinx_u280_xdma_201920_3 \
	       --top sptrsv_kernel \
	       --work-dir tapa_workdir \
		   --clock-period 4 \
	       --connectivity sptrsv.ini \
	       --floorplan-output constraint.tcl \
	       --read-only-args "value0" \
	       --read-only-args "col_idx0" \
		   --read-only-args "value1" \
	       --read-only-args "col_idx1" \
	       --read-only-args "b" \
	       --write-only-args "x" \
	       --enable-floorplan \
	       --enable-hbm-binding-adjustment \
	       --min-area-limit 0.45 \
	       --max-area-limit 0.65

hw_build:
	make tapa_full
	chmod 755 sptrsv.hw_generate_bitstream.sh
	$(shell ./sptrsv.hw_generate_bitstream.sh)

clean:
	rm -rf sptrsv
	rm -rf sptrsv.hw.xo
	rm -rf tapa_workdir
	rm -rf sptrsv.hw_generate_bitstream.sh
	rm -rf sptrsv.hw_emu.xclbin
	rm -rf sptrsv.hw_emu.xclbin.info
	rm -rf sptrsv.hw_emu.xclbin.link_summary
	rm -rf _x
	rm -rf vitis_run_hw
	rm -rf outx.txt
	rm -rf constraint.tcl
	rm -rf *.log

.PHONY : clean
