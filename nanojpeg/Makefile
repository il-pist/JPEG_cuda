#CC=/usr/local/cuda/bin/nvcc
CC=/opt/cuda/bin/nvcc

nanoex: nanojpeg.cu nanoex.cu
	#$(CC) -D_NJ_EXAMPLE_PROGRAM -o $@ $^
	$(CC) -I . -o $@ $^

all: nanoex

run: nanoex
	./$< saint_nicolas.jpg out.ppm

profile: nanoex
	# "." means source in sh
	sudo sh -c ". ../../setnvidia.sh; nvprof ./$<"

