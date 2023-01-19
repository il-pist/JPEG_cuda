// //////////////////////////////////////////////////////////
// how to use TooJpeg: creating a JPEG file
// see https://create.stephan-brumme.com/toojpeg/
// compile: g++ example.cpp toojpeg.cpp -o example -std=c++11

#include "toojpeg.h"

// //////////////////////////////////////////////////////////
// use a C++ file stream
#include <fstream>
#include <time.h>

// output file
const char* filename = "example_gray_cuda.jpg";
std::ofstream myFile(filename, std::ios_base::out | std::ios_base::binary);

// write a single byte compressed by TooJpeg
void myOutput(unsigned char byte)
{
  myFile << byte;
}

// //////////////////////////////////////////////////////////
int main()
{
	clock_t start, end;


  // 8000x6000 image
  const auto width  = 8000;
  const auto height = 6000;
  // Grayscale: one byte per pixel
  const auto bytesPerPixel = 1;

  // allocate memory
  auto image = new unsigned char[width * height * bytesPerPixel];

  // create a nice color transition (replace with your code)
  for (auto y = 0; y < height; y++)
    for (auto x = 0; x < width; x++)
    {
      // memory location of current pixel
      auto offset = (y * width + x) * bytesPerPixel;

      // red and green fade from 0 to 255, blue is always 127
      auto red   = 255 * x / width;
      auto green = 255 * y / height;
      image[offset] = (red + green) / 2;;
    }

  // start JPEG compression
  // note: myOutput is the function defined in line 18, it saves the output in example.jpg
  // optional parameters:
  const bool isRGB      = false; // true = RGB image, else false = grayscale
  const auto quality    = 90;    // compression quality: 0 = worst, 100 = best, 80 to 90 are most often used
  const bool downsample = false; // false = save as YCbCr444 JPEG (better quality), true = YCbCr420 (smaller file)
  const char* comment   = "TooJpeg example image"; // arbitrary JPEG comment

  start=clock();

  auto ok = TooJpeg::writeJpeg(myOutput, image, width, height, isRGB, quality, downsample, comment);

  delete[] image;

  end=clock();

  printf("time: %f\n\n", double(end-start) / CLOCKS_PER_SEC);


  // error => exit code 1
  return ok ? 0 : 1;
}
