example:
	g++ example.cpp toojpeg.cpp -o example -std=c++11
	./example

example_cuda:
	nvcc example_cuda.cu toojpeg_cuda.cu -o example_cuda -std=c++11
	./example_cuda

nanojpeg_cuda: example_cuda
	nvcc -O3 -D_NJ_EXAMPLE_PROGRAM -o nanojpeg nanojpeg.cu
	./nanojpeg example_cuda.jpg example_cuda.ppm

clean:			
	rm -f example example.jpg

clean_cuda:	
	rm -f example_cuda example_cuda.jpg nanojpeg example_cuda.ppm
