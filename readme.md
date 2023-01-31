# A JPEG encoder in a single CUDA file

*This is a CUDA variant of the library hosted at* https://create.stephan-brumme.com/toojpeg/

TooJpeg\_cuda is a compact baseline JPEG/JFIF writer, written in CUDA.  
Its interface has only one function: `writeJpeg()` - and that's it !

The library supports the most common JPEG output color spaces:
- YCbCr444,
- YCbCr420 (=2x2 downsampled) and
- Y (grayscale)

# How to use

1. create an image with any content you like, e.g. 1024x768, RGB (3 bytes per pixel)

```cpp
   auto pixels = new unsigned char[1024*768*3];
```

2. define a callback that receives the compressed data byte-by-byte 

```cpp
// for example, write to disk (could be anything else, too, such as network transfer, in-memory storage, etc.)
void myOutput(unsigned char oneByte) { fputc(oneByte, myFileHandle); }
```

3. start JPEG compression

```cpp
TooJpeg::writeJpeg(myOutput, mypixels, 1024, 768);
// actually there are some optional parameters, too
//bool ok = TooJpeg::writeJpeg(myOutput, pixels, width, height, isRGB, quality, downSample, comment);
```


The project provides some tests that could be run through the makefile provided
