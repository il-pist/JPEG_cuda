
test_toojpeg: test_toojpeg.cu img.cpp toojpeg_cuda.cu
	nvcc $^ -o $@ 
	./test_toojpeg

test_toojpeg_serial: test_toojpeg_serial.cpp img.cpp toojpeg.cpp
	nvcc $^ -o $@ 
	./test_toojpeg_serial

example: example.cpp toojpeg.cpp
	g++ $^ -o $@ -std=c++11
	./example

example_gray: example_gray.cpp toojpeg.cpp
	g++ $^ -o $@ -std=c++11
	./example_gray

example_cuda: example_cuda.cu toojpeg_cuda.cu
	nvcc $^ -o $@ 
	./example_cuda

example_gray_cuda: example_gray_cuda.cu toojpeg_cuda.cu
	nvcc $^ -o $@ 
	./example_gray_cuda

nanojpeg_cuda: example_cuda
	nvcc -O3 -D_NJ_EXAMPLE_PROGRAM -o nanojpeg nanojpeg.cu
	./nanojpeg example_cuda.jpg example_cuda.ppm

clean:			
	rm -f example example.jpg

clean_cuda:	
	rm -f example_cuda example_cuda.jpg nanojpeg example_cuda.ppm

clean_all_img:
	rm exampl*.jpg test*.jpg 

clean_all:
	make clean_all_img
	rm -f example example_gray example_cuda example_gray_cuda nanjpeg_cuda test_toojpeg test_toojpeg_serial
