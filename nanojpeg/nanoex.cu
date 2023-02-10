///////////////////////////////////////////////////////////////////////////////
// EXAMPLE PROGRAM                                                           //
// just define _NJ_EXAMPLE_PROGRAM to compile this (requires NJ_USE_LIBC)    //
///////////////////////////////////////////////////////////////////////////////

// EXAMPLE
// =======
//
// A few pages below, you can find an example program that uses NanoJPEG to
// convert JPEG files into PGM or PPM. To compile it, use something like
//     gcc -O3 -D_NJ_EXAMPLE_PROGRAM -o nanojpeg nanojpeg.c
// You may also add -std=c99 -Wall -Wextra -pedantic -Werror, if you want :)
// The only thing you might need is -Wno-shift-negative-value, because this
// code relies on the target machine using two's complement arithmetic, but
// the C standard does not, even though *any* practically useful machine
// nowadays uses two's complement.

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "nanojpeg.h"

int main(int argc, char* argv[]) {
    int size;
    char *buf;
    FILE *f;
	clock_t start, end;

	//printf("sizeof\n  int: %lu\n  unsigned long: %lu\n  char: %lu\n  unsigned char: %lud\n  void * : %lu\n",
	//	sizeof(int), sizeof(unsigned long), sizeof(char), sizeof(unsigned char), sizeof(void*));

	start=clock();
    if (argc < 2) {
        printf("Usage: %s <input.jpg> [<output.ppm>]\n", argv[0]);
        return 2;
    }
    f = fopen(argv[1], "rb");
    if (!f) {
        printf("Error opening the input file.\n");
        return 1;
    }
    fseek(f, 0, SEEK_END);
    size = (int) ftell(f);
    buf = (char*) malloc(size);
    fseek(f, 0, SEEK_SET);
    size = (int) fread(buf, 1, size, f);
    fclose(f);

    njInit();
    if (njDecode(buf, size)) {
        free((void*)buf);
        printf("Error decoding the input file.\n");
        return 1;
    }
    free((void*)buf);

    f = fopen((argc > 2) ? argv[2] : (njIsColor() ? "nanojpeg_out.ppm" : "nanojpeg_out.pgm"), "wb");
    if (!f) {
        printf("Error opening the output file.\n");
        return 1;
    }
    fprintf(f, "P%d\n%d %d\n255\n", njIsColor() ? 6 : 5, njGetWidth(), njGetHeight());
    fwrite(njGetImage(), 1, njGetImageSize(), f);
    fclose(f);
    njDone();

	end=clock();
	printf("\nExecution time: %f\n", double(end-start) / CLOCKS_PER_SEC);
    return 0;
}

