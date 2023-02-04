#CC=/usr/local/cuda/bin/nvcc
#CC=/opt/cuda/bin/nvcc
CC=gcc

nanorig: nanojpeg.c
	$(CC) -D_NJ_EXAMPLE_PROGRAM -o $@ $^
	#$(CC) -I . -o $@ $^

all: nanorig

run: nanorig
	./$< ../saint_nicolas.jpg outorigvscode.ppm

#profile: nanoex
#	# "." means source in sh
#	sudo sh -c ". ../../setnvidia.sh; nvprof ./$<"

