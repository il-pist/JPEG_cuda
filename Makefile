example: example.cpp toojpeg.cpp
	g++ $^ -o $@ -std=c++11
	./example

example_cuda: example_cuda.cu toojpeg_cuda.cu
	nvcc $^ -o $@ 
	./example_cuda

nanojpeg_cuda: example_cuda
	nvcc -O3 -D_NJ_EXAMPLE_PROGRAM -o nanojpeg nanojpeg.cu
	./nanojpeg example_cuda.jpg example_cuda.ppm

clean:			
	rm -f example example.jpg

clean_cuda:	
	rm -f example_cuda example_cuda.jpg nanojpeg example_cuda.ppm
