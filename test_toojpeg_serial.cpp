// //////////////////////////////////////////////////////////
// how to use TooJpeg: creating a JPEG file
// see https://create.stephan-brumme.com/toojpeg/
// compile: g++ example.cpp toojpeg.cpp -o example -std=c++11

#include "toojpeg.h"
#include "img.h"

// //////////////////////////////////////////////////////////
// use a C++ file stream
#include <fstream>
#include <time.h>

// output file
std::ofstream myFile1("test1_toojpeg_serial.jpg", std::ios_base::out | std::ios_base::binary);

// write a single byte compressed by tooJpeg
void myOutput1(unsigned char byte)
{
  myFile1 << byte;
}

// output file
std::ofstream myFile2("test2_toojpeg_serial.jpg", std::ios_base::out | std::ios_base::binary);

// write a single byte compressed by tooJpeg
void myOutput2(unsigned char byte)
{
  myFile2 << byte;
}

// output file
std::ofstream myFile3("test3_toojpeg_serial.jpg", std::ios_base::out | std::ios_base::binary);

// write a single byte compressed by tooJpeg
void myOutput3(unsigned char byte)
{
  myFile3 << byte;
}

// output file
std::ofstream myFile4("test4_toojpeg_serial.jpg", std::ios_base::out | std::ios_base::binary);

// write a single byte compressed by tooJpeg
void myOutput4(unsigned char byte)
{
  myFile4 << byte;
}

// output file
std::ofstream myFile5("test5_toojpeg_serial.jpg", std::ios_base::out | std::ios_base::binary);

// write a single byte compressed by tooJpeg
void myOutput5(unsigned char byte)
{
  myFile5 << byte;
}

// output file
std::ofstream myFile6("test6_toojpeg_serial.jpg", std::ios_base::out | std::ios_base::binary);

// write a single byte compressed by tooJpeg
void myOutput6(unsigned char byte)
{
  myFile6 << byte;
}



// //////////////////////////////////////////////////////////
int main()
{

	clock_t start, end;
	

// TEST 1

	printf("# test1:\n\n");
	printf("# 8000x6000 RGB image with downsample and 90%% quality image\n");

  // 800x600 image
  const auto width  = 8000;
  const auto height = 6000;
  // RGB: one byte each for red, green, blue
  const auto bytesPerPixel = 3;

  // allocate memory
  auto image = new unsigned char[width * height * bytesPerPixel];

  // create a nice color transition (replace with your code)
  for (auto y = 0; y < height; y++)
    for (auto x = 0; x < width; x++)
    {
      // memory location of current pixel
      auto offset = (y * width + x) * bytesPerPixel;

      // red and green fade from 0 to 255, blue is always 127
      image[offset    ] = 255 * x / width;
      image[offset + 1] = 255 * y / height;
      image[offset + 2] = 127;
    }

  // start JPEG compression
  // note: myOutput is the function defined in line 18, it saves the output in example.jpg
  // optional parameters:
  const bool isRGB      = true;  // true = RGB image, else false = grayscale
  const auto quality    = 90;    // compression quality: 0 = worst, 100 = best, 80 to 90 are most often used
  const bool downsample = true; // false = save as YCbCr444 JPEG (better quality), true = YCbCr420 (smaller file)
  const char* comment = "TooJpeg example image"; // arbitrary JPEG comment

	start=clock();

  auto ok = TooJpeg::writeJpeg(myOutput1, image, width, height, isRGB, quality, downsample, comment);

//  delete[] image;

	end=clock();
	
	printf("time: %f\n\n", double(end-start) / CLOCKS_PER_SEC);
	


// TEST 2

	printf("# test2:\n# 8000x6000 RGB image WITHOUT downsample and 90%% quality image\n");
	start=clock();

 	ok = ok | TooJpeg::writeJpeg(myOutput2, image, width, height, isRGB, quality, false, comment);

	end=clock();
	
	printf("time: %f\n\n", double(end-start) / CLOCKS_PER_SEC);


//TEST 3
	
  printf("# test3:\n# sample_1920x1280.ppm origin image with downsample and 90%% of quality image\n");
	RGBImage* image3=readPPM("sample_1920×1280.ppm");
	start=clock();

  ok = ok | TooJpeg::writeJpeg(myOutput3, image3->data, image3->width, image3->height, isRGB, quality, downsample, comment);

	end=clock();
	
	printf("time: %f\n\n", double(end-start) / CLOCKS_PER_SEC);

	

// TEST 4

  printf("# test4:\n# sample_5184x3456.ppm origin image with downsample and 90%% of quality image\n");
	RGBImage* image4=readPPM("sample_5184×3456.ppm");
	start=clock();
	
  ok = ok | TooJpeg::writeJpeg(myOutput4, image4->data, image4->width, image4->height, isRGB, quality, downsample, comment);

	end=clock();
	
	printf("time: %f\n\n", double(end-start) / CLOCKS_PER_SEC);


// TEST 5

  printf("# test5:\n# sample_5184x3456.ppm origin image WITHOUT downsample and 90%% of quality image\n");
	RGBImage* image5=readPPM("sample_5184×3456.ppm");
	start=clock();
	
  ok = ok | TooJpeg::writeJpeg(myOutput5, image5->data, image5->width, image5->height, isRGB, quality, false, comment);

	end=clock();
	
	printf("time: %f\n\n", double(end-start) / CLOCKS_PER_SEC);

// TEST 6

  printf("# test6:\n# 8000x6000 grayscale image WITHOUT donsampling and 90%% quality\n");
    // 8000x6000 image
//  width  = 8000;
//  height = 6000;
  // Grayscale: one byte per pixel
  int bytesPerPixelGray = 1;

  // allocate memory
  auto image6 = new unsigned char[width * height * bytesPerPixelGray];

  // create a nice color transition (replace with your code)
  for (auto y = 0; y < height; y++)
    for (auto x = 0; x < width; x++)
    {
      // memory location of current pixel
      auto offset = (y * width + x) * bytesPerPixelGray;

      // red and green fade from 0 to 255, blue is always 127
      auto red   = 255 * x / width;
      auto green = 255 * y / height;
      image6[offset] = (red + green) / 2;;
    }
  start=clock();

  ok = ok | TooJpeg::writeJpeg(myOutput6, image6, width, height, !isRGB, quality, false, comment);

  delete[] image6;
  end=clock();

  printf("time: %f\n\n", double(end-start) / CLOCKS_PER_SEC);



  // error => exit code 1
  return ok ? 0 : 1;
}
