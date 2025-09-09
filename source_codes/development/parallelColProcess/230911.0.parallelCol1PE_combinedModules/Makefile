COMPILER_FLAGS = -O2 -ltapa -lfrt -lglog -lgflags -lOpenCL 
#-fsanitize=address -g

VITIS_HLS_INCL_DIR = /local-scratch/Xilinx/Vitis_HLS/2021.2/include/

platform = xilinx_u280_xdma_201920_3

TEST_FILE = sptrsv_test.cpp
KERNEL_FILE = sptrsv_kernel.cpp
TOP = sptrsv_kernel

gcc_compile : $(KERNEL_FILE) $(TEST_FILE)
	g++ $(KERNEL_FILE) $(TEST_FILE)  -o sptrsv  $(COMPILER_FLAGS) -I$(VITIS_HLS_INCL_DIR)
	./sptrsv

tapa_synth : $(KERNEL_FILE)
	tapac -o sptrsv.hw.xo $(KERNEL_FILE) \
  	--platform $(platform) \
  	--top $(TOP) \
  	--work-dir tapa_workdir \
  	--clock-period 4	\
	--connectivity sptrsv.ini 

tapa_hw_emu : sptrsv.hw.xo
	./sptrsv --bitstream=sptrsv.hw.xo -xosim_work_dir tapa_cosim_debug -xosim_save_waveform

vitis_hw_emu : sptrsv.hw.xo
	v++ -o sptrsv.hw_emu.xclbin \
	  --link \
	  --target hw_emu \
	  --kernel $(TOP) \
	  --platform $(platform) \
	  sptrsv.hw.xo
	./sptrsv --bitstream=sptrsv.hw_emu.xclbin

tapa_full : $(KERNEL_FILE)
	tapac -o sptrsv.hw.xo $(KERNEL_FILE) \
	       --platform xilinx_u280_xdma_201920_3 \
	       --top $(TOP) \
	       --work-dir tapa_workdir \
		   --clock-period 4 \
	       --connectivity sptrsv.ini \
	       --floorplan-output constraint.tcl \
	       --read-only-args "value0" \
		   --read-only-args "value1" \
		   --read-only-args "bval0" \
		   --read-only-args "bval1" \
	       --write-only-args "xSolved" \
	       --enable-floorplan \
	       --enable-hbm-binding-adjustment \
	       --min-area-limit 0.45 \
	       --max-area-limit 0.65 \
		   --enable-synth-util	\
		   

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
	rm -rf /tmp/.frt.*/*
	rm -rf *.log
	rm -rf tapa_cosim_debug
	rm -rf *.jou

hard_clean:
	make clean
	rm -rf .Xil
	rm -rf .ipcache

.PHONY : clean

