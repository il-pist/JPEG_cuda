#CC=/usr/local/cuda/bin/nvcc
CC=/opt/cuda/bin/nvcc

nanoex_orig: nanojpeg.c
	$(CC) -D_NJ_EXAMPLE_PROGRAM -o $@ $^

nanoex: nanojpeg.cu nanoex.cu
	#$(CC) -D_NJ_EXAMPLE_PROGRAM -o $@ $^
	$(CC) -lineinfo -I . -o $@ $^

all: nanoex

run: nanoex
	./$< saint_nicolas.jpg out.ppm
run1: nanoex
	/opt/cuda/bin/cuda-memcheck ./nanoex pattern1_444.jpg pattern1_444.ppm | less
run2: nanoex
	/opt/cuda/bin/cuda-memcheck ./nanoex pattern1_422horiz.jpg pattern1_422h.ppm | less
run3: nanoex
	/opt/cuda/bin/cuda-memcheck ./nanoex pattern1_422vert.jpg pattern1_422v.ppm | less

profile: nanoex
	# "." means source in sh
	sudo sh -c ". ../../setnvidia.sh; nvprof ./$<"

