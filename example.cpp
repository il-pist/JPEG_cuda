// //////////////////////////////////////////////////////////
// how to use TooJpeg: creating a JPEG file
// see https://create.stephan-brumme.com/toojpeg/
// compile: g++ example.cpp toojpeg.cpp -o example -std=c++11

#include "toojpeg.h"

// //////////////////////////////////////////////////////////
// use a C++ file stream
#include <fstream>

// output file
std::ofstream myFile("example.jpg", std::ios_base::out | std::ios_base::binary);

// write a single byte compressed by tooJpeg
void myOutput(unsigned char byte)
{
  myFile << byte;
}

// //////////////////////////////////////////////////////////
int main()
{
  // 800x600 image
  const auto width  = 800;
  const auto height = 600;
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
  const bool downsample = false; // false = save as YCbCr444 JPEG (better quality), true = YCbCr420 (smaller file)
  const char* comment = "TooJpeg example image"; // arbitrary JPEG comment
  auto ok = TooJpeg::writeJpeg(myOutput, image, width, height, isRGB, quality, downsample, comment);

  delete[] image;

  // error => exit code 1
  return ok ? 0 : 1;
}
