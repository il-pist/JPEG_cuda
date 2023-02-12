#include "nanojpeg.h"

///////////////////////////////////////////////////////////////////////////////
// IMPLEMENTATION SECTION                                                    //
// you may stop reading here                                                 //
///////////////////////////////////////////////////////////////////////////////

#ifdef _MSC_VER
	#define NJ_INLINE static __inline
	#define NJ_FORCE_INLINE static __forceinline
#else
	#define NJ_INLINE static inline
	#define NJ_FORCE_INLINE static inline
#endif

#if NJ_USE_LIBC
	#include <stdlib.h>
	#include <stdio.h>  // added to allow compilation
	#include <string.h>
	#define njAllocMem malloc
	#define njFreeMem  free
	#define njFillMem  memset
	#define njCopyMem  memcpy
#elif NJ_USE_WIN32
	#include <windows.h>
	#define njAllocMem(size) ((void*) LocalAlloc(LMEM_FIXED, (SIZE_T)(size)))
	#define njFreeMem(block) ((void) LocalFree((HLOCAL) block))
	NJ_INLINE void njFillMem(void* block, unsigned char value, int count) { __asm {
		mov edi, block
		mov al, value
		mov ecx, count
		rep stosb
	} }
	NJ_INLINE void njCopyMem(void* dest, const void* src, int count) { __asm {
		mov edi, dest
		mov esi, src
		mov ecx, count
		rep movsb
	} }
#else
	extern void* njAllocMem(int size);
	extern void njFreeMem(void* block);
	extern void njFillMem(void* block, unsigned char byte, int size);
	extern void njCopyMem(void* dest, const void* src, int size);
#endif

#define NSTR 4

typedef struct _nj_code {
	unsigned char bits, code;
} nj_vlc_code_t;

/// One color component (descriptor + pixel data)
typedef struct _nj_cmp {
	int cid;                ///< id del descittore della component nel SoF (1, 2, 3)
	int ssx, ssy;           ///< n blocchi per minimum compressible unit / mb (ad esempio 2x2 per 4:2:0)
	int width, height;
	int stride;             ///< = nj.mbwidth * ssx * 8 : double stride: for chroma subsampling
	int qtsel;              ///< ??? da descittore della component
	int actabsel, dctabsel;
	int dcpred;
	int *intpixels; ///< pixel data for initial file read and IDCT
	int *cuintpixels; ///< pixel data for initial file read and IDCT
	unsigned char *pixels;  ///< pixel data
	unsigned char *cupixels; ///< pixel data on device
} nj_component_t;

typedef struct _nj_ctx {
	nj_result_t error;
	int use_cuda;
	cudaStream_t custreams[NSTR];
	const unsigned char *pos;
	int size;
	int length;
	int width, height;      ///< dimensione immagine in pixel
	int mbwidth, mbheight;  ///< dimensione immagine in unità di minimum coded blocks / mb (8x8, 16x16...)
	int mbsizex, mbsizey;   ///< dimensione in pixel di un Minimum Coded Block / mb: 8x8, 16x16...
	int ncomp;              ///< number of components
	nj_component_t comp[3]; ///< array of components (descriptor + pixel data)
	int qtused, qtavail;
	unsigned char qtab[4][64];
	nj_vlc_code_t vlctab[4][65536];
	int buf, bufbits;
	int block[64];          ///< TEMP un blocco temporaneo usato in njDecodeBlock
	int rstinterval;
	unsigned char *rgb;
	unsigned char *curgb;
} nj_context_t;

static nj_context_t nj; /// Unique static state struct (not multithread-safe in this state)

/// Zig-Zag pattern
static const char njZZ[64] = { 0, 1, 8, 16, 9, 2, 3, 10, 17, 24, 32, 25, 18,
11, 4, 5, 12, 19, 26, 33, 40, 48, 41, 34, 27, 20, 13, 6, 7, 14, 21, 28, 35,
42, 49, 56, 57, 50, 43, 36, 29, 22, 15, 23, 30, 37, 44, 51, 58, 59, 52, 45,
38, 31, 39, 46, 53, 60, 61, 54, 47, 55, 62, 63 };

inline bool failed(cudaError_t error)
{
  if (cudaSuccess == error)
    return false;

  //fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(error));
  printf("[failed] CUDA error: %s\n", cudaGetErrorString(error));
  return true;
}

NJ_FORCE_INLINE unsigned char njClip(const int x) {
	return (x < 0) ? 0 : ((x > 0xFF) ? 0xFF : (unsigned char) x);
}
__device__ __forceinline__ unsigned char njCudaClip(const int x) {
	return (x < 0) ? 0 : ((x > 0xFF) ? 0xFF : (unsigned char) x);
}


#define W1 2841
#define W2 2676
#define W3 2408
#define W5 1609
#define W6 1108
#define W7 565

__global__ void njCudaRowIDCT(int* blk, int stride, int height) {
	int x0, x1, x2, x3, x4, x5, x6, x7, x8;
	
	int x = (blockIdx.x*blockDim.x + threadIdx.x)*8; // original pixel x (8 pixel per thread)
	int y = blockIdx.y*blockDim.y + threadIdx.y; // original pixel y
	blk += stride * y + x;

	if(x < stride && y < height)
	{
		if (!((x1 = blk[4] << 11)
			| (x2 = blk[6])
			| (x3 = blk[2])
			| (x4 = blk[1])
			| (x5 = blk[7])
			| (x6 = blk[5])
			| (x7 = blk[3])))
		{
			blk[0] = blk[1] = blk[2] = blk[3] = blk[4] = blk[5] = blk[6] = blk[7] = blk[0] << 3;
			return;
		}
		x0 = (blk[0] << 11) + 128;
		x8 = W7 * (x4 + x5);
		x4 = x8 + (W1 - W7) * x4;
		x5 = x8 - (W1 + W7) * x5;
		x8 = W3 * (x6 + x7);
		x6 = x8 - (W3 - W5) * x6;
		x7 = x8 - (W3 + W5) * x7;
		x8 = x0 + x1;
		x0 -= x1;
		x1 = W6 * (x3 + x2);
		x2 = x1 - (W2 + W6) * x2;
		x3 = x1 + (W2 - W6) * x3;
		x1 = x4 + x6;
		x4 -= x6;
		x6 = x5 + x7;
		x5 -= x7;
		x7 = x8 + x3;
		x8 -= x3;
		x3 = x0 + x2;
		x0 -= x2;
		x2 = (181 * (x4 + x5) + 128) >> 8;
		x4 = (181 * (x4 - x5) + 128) >> 8;
		blk[0] = (x7 + x1) >> 8;
		blk[1] = (x3 + x2) >> 8;
		blk[2] = (x0 + x4) >> 8;
		blk[3] = (x8 + x6) >> 8;
		blk[4] = (x8 - x6) >> 8;
		blk[5] = (x0 - x4) >> 8;
		blk[6] = (x3 - x2) >> 8;
		blk[7] = (x7 - x1) >> 8;
	}
}

__global__ void njCudaColIDCT(const int* blk, unsigned char *out, int stride, int height) {
	int x0, x1, x2, x3, x4, x5, x6, x7, x8;

	int x = blockIdx.x*blockDim.x + threadIdx.x; // original pixel x
	int y = (blockIdx.y*blockDim.y + threadIdx.y)*8; // original pixel y (8 pixel per thread)
	blk += stride * y + x;
	out += stride * y + x;

	if(x < stride && y < height)
	{
		if (!((x1 = blk[stride*4] << 8)
			| (x2 = blk[stride*6])
			| (x3 = blk[stride*2])
			| (x4 = blk[stride*1])
			| (x5 = blk[stride*7])
			| (x6 = blk[stride*5])
			| (x7 = blk[stride*3])))
		{
			x1 = njCudaClip(((blk[0] + 32) >> 6) + 128);
			for (x0 = 8;  x0;  --x0) {
				*out = (unsigned char) x1;
				out += stride;
			}
			return;
		}
		x0 = (blk[0] << 8) + 8192;
		x8 = W7 * (x4 + x5) + 4;
		x4 = (x8 + (W1 - W7) * x4) >> 3;
		x5 = (x8 - (W1 + W7) * x5) >> 3;
		x8 = W3 * (x6 + x7) + 4;
		x6 = (x8 - (W3 - W5) * x6) >> 3;
		x7 = (x8 - (W3 + W5) * x7) >> 3;
		x8 = x0 + x1;
		x0 -= x1;
		x1 = W6 * (x3 + x2) + 4;
		x2 = (x1 - (W2 + W6) * x2) >> 3;
		x3 = (x1 + (W2 - W6) * x3) >> 3;
		x1 = x4 + x6;
		x4 -= x6;
		x6 = x5 + x7;
		x5 -= x7;
		x7 = x8 + x3;
		x8 -= x3;
		x3 = x0 + x2;
		x0 -= x2;
		x2 = (181 * (x4 + x5) + 128) >> 8;
		x4 = (181 * (x4 - x5) + 128) >> 8;
		*out = njCudaClip(((x7 + x1) >> 14) + 128);  out += stride;
		*out = njCudaClip(((x3 + x2) >> 14) + 128);  out += stride;
		*out = njCudaClip(((x0 + x4) >> 14) + 128);  out += stride;
		*out = njCudaClip(((x8 + x6) >> 14) + 128);  out += stride;
		*out = njCudaClip(((x8 - x6) >> 14) + 128);  out += stride;
		*out = njCudaClip(((x0 - x4) >> 14) + 128);  out += stride;
		*out = njCudaClip(((x3 - x2) >> 14) + 128);  out += stride;
		*out = njCudaClip(((x7 - x1) >> 14) + 128);
	}
}

/// ======= Originale =============
NJ_INLINE void njRowIDCT(int* blk) {
	int x0, x1, x2, x3, x4, x5, x6, x7, x8;
	if (!((x1 = blk[4] << 11)
		| (x2 = blk[6])
		| (x3 = blk[2])
		| (x4 = blk[1])
		| (x5 = blk[7])
		| (x6 = blk[5])
		| (x7 = blk[3])))
	{
		blk[0] = blk[1] = blk[2] = blk[3] = blk[4] = blk[5] = blk[6] = blk[7] = blk[0] << 3;
		return;
	}
	x0 = (blk[0] << 11) + 128;
	x8 = W7 * (x4 + x5);
	x4 = x8 + (W1 - W7) * x4;
	x5 = x8 - (W1 + W7) * x5;
	x8 = W3 * (x6 + x7);
	x6 = x8 - (W3 - W5) * x6;
	x7 = x8 - (W3 + W5) * x7;
	x8 = x0 + x1;
	x0 -= x1;
	x1 = W6 * (x3 + x2);
	x2 = x1 - (W2 + W6) * x2;
	x3 = x1 + (W2 - W6) * x3;
	x1 = x4 + x6;
	x4 -= x6;
	x6 = x5 + x7;
	x5 -= x7;
	x7 = x8 + x3;
	x8 -= x3;
	x3 = x0 + x2;
	x0 -= x2;
	x2 = (181 * (x4 + x5) + 128) >> 8;
	x4 = (181 * (x4 - x5) + 128) >> 8;
	blk[0] = (x7 + x1) >> 8;
	blk[1] = (x3 + x2) >> 8;
	blk[2] = (x0 + x4) >> 8;
	blk[3] = (x8 + x6) >> 8;
	blk[4] = (x8 - x6) >> 8;
	blk[5] = (x0 - x4) >> 8;
	blk[6] = (x3 - x2) >> 8;
	blk[7] = (x7 - x1) >> 8;
}

NJ_INLINE void njColIDCT(const int* blk, unsigned char *out, int stride) {
	int x0, x1, x2, x3, x4, x5, x6, x7, x8;
	if (!((x1 = blk[8*4] << 8)
		| (x2 = blk[8*6])
		| (x3 = blk[8*2])
		| (x4 = blk[8*1])
		| (x5 = blk[8*7])
		| (x6 = blk[8*5])
		| (x7 = blk[8*3])))
	{
		x1 = njClip(((blk[0] + 32) >> 6) + 128);
		for (x0 = 8;  x0;  --x0) {
			*out = (unsigned char) x1;
			out += stride;
		}
		return;
	}
	x0 = (blk[0] << 8) + 8192;
	x8 = W7 * (x4 + x5) + 4;
	x4 = (x8 + (W1 - W7) * x4) >> 3;
	x5 = (x8 - (W1 + W7) * x5) >> 3;
	x8 = W3 * (x6 + x7) + 4;
	x6 = (x8 - (W3 - W5) * x6) >> 3;
	x7 = (x8 - (W3 + W5) * x7) >> 3;
	x8 = x0 + x1;
	x0 -= x1;
	x1 = W6 * (x3 + x2) + 4;
	x2 = (x1 - (W2 + W6) * x2) >> 3;
	x3 = (x1 + (W2 - W6) * x3) >> 3;
	x1 = x4 + x6;
	x4 -= x6;
	x6 = x5 + x7;
	x5 -= x7;
	x7 = x8 + x3;
	x8 -= x3;
	x3 = x0 + x2;
	x0 -= x2;
	x2 = (181 * (x4 + x5) + 128) >> 8;
	x4 = (181 * (x4 - x5) + 128) >> 8;
	*out = njClip(((x7 + x1) >> 14) + 128);  out += stride;
	*out = njClip(((x3 + x2) >> 14) + 128);  out += stride;
	*out = njClip(((x0 + x4) >> 14) + 128);  out += stride;
	*out = njClip(((x8 + x6) >> 14) + 128);  out += stride;
	*out = njClip(((x8 - x6) >> 14) + 128);  out += stride;
	*out = njClip(((x0 - x4) >> 14) + 128);  out += stride;
	*out = njClip(((x3 - x2) >> 14) + 128);  out += stride;
	*out = njClip(((x7 - x1) >> 14) + 128);
}

#define njThrow(e) do { nj.error = e; return; } while (0)
#define njCheckError() do { if (nj.error) return; } while (0)

static int njShowBits(int bits) {
	unsigned char newbyte;
	if (!bits) return 0;
	while (nj.bufbits < bits) {
		if (nj.size <= 0) {
			nj.buf = (nj.buf << 8) | 0xFF;
			nj.bufbits += 8;
			continue;
		}
		newbyte = *nj.pos++;
		nj.size--;
		nj.bufbits += 8;
		nj.buf = (nj.buf << 8) | newbyte;
		if (newbyte == 0xFF) {
			if (nj.size) {
				unsigned char marker = *nj.pos++;
				nj.size--;
				switch (marker) {
					case 0x00:
					case 0xFF:
						break;
					case 0xD9: nj.size = 0; break;
					default:
						if ((marker & 0xF8) != 0xD0)
							nj.error = NJ_SYNTAX_ERROR;
						else {
							nj.buf = (nj.buf << 8) | marker;
							nj.bufbits += 8;
						}
				}
			} else
				nj.error = NJ_SYNTAX_ERROR;
		}
	}
	return (nj.buf >> (nj.bufbits - bits)) & ((1 << bits) - 1);
}

NJ_INLINE void njSkipBits(int bits) {
	if (nj.bufbits < bits)
		(void) njShowBits(bits);
	nj.bufbits -= bits;
}

NJ_INLINE int njGetBits(int bits) {
	int res = njShowBits(bits);
	njSkipBits(bits);
	return res;
}

NJ_INLINE void njByteAlign(void) {
	nj.bufbits &= 0xF8;
}

static void njSkip(int count) {
	nj.pos += count;
	nj.size -= count;
	nj.length -= count;
	if (nj.size < 0) nj.error = NJ_SYNTAX_ERROR;
}

NJ_INLINE unsigned short njDecode16(const unsigned char *pos) {
	return (pos[0] << 8) | pos[1];
}

static void njDecodeLength(void) {
	if (nj.size < 2) njThrow(NJ_SYNTAX_ERROR);
	nj.length = njDecode16(nj.pos);
	if (nj.length > nj.size) njThrow(NJ_SYNTAX_ERROR);
	njSkip(2);
}

NJ_INLINE void njSkipMarker(void) {
	njDecodeLength();
	njSkip(nj.length);
}

NJ_INLINE void njDecodeSOF(void) {
	int i, ssxmax = 0, ssymax = 0; ///< ssxmax, ssymax: massimo numero di blocchi per minimum coded unit / mb (dimensione della grid più larga che si trova nell'immagine, in unità di 8x8)
	nj_component_t* c;
	njDecodeLength();
	njCheckError();
	if (nj.length < 9) njThrow(NJ_SYNTAX_ERROR);
	if (nj.pos[0] != 8) njThrow(NJ_UNSUPPORTED);
	nj.height = njDecode16(nj.pos+1);
	nj.width = njDecode16(nj.pos+3);
	if (!nj.width || !nj.height) njThrow(NJ_SYNTAX_ERROR);
	nj.ncomp = nj.pos[5];
	njSkip(6);
	switch (nj.ncomp) {
		case 1:
		case 3:
			break;
		default:
			njThrow(NJ_UNSUPPORTED);
	}
	if (nj.length < (nj.ncomp * 3)) njThrow(NJ_SYNTAX_ERROR);
	for (i = 0, c = nj.comp;  i < nj.ncomp;  ++i, ++c) { // per ogni descrittore di componente (Y/Cb/Cr/R/G/B)
		c->cid = nj.pos[0]; // pos[1] è un id incrementale del descrittore di componente
		if (!(c->ssx = nj.pos[1] >> 4)) njThrow(NJ_SYNTAX_ERROR); // pos[1] nibble superiore = n blocchi in orizzontale per minimum compressible unit / mb (ad esempio 2x2 per 4:2:0)
		if (c->ssx & (c->ssx - 1)) njThrow(NJ_UNSUPPORTED);  // non-power of two
		if (!(c->ssy = nj.pos[1] & 15)) njThrow(NJ_SYNTAX_ERROR); // pos[1] nibble inferiore = n blocchi in verticale per minimum compressible unit / mb (ad esempio 2x2 per 4:2:0)
		if (c->ssy & (c->ssy - 1)) njThrow(NJ_UNSUPPORTED);  // non-power of two
		if ((c->qtsel = nj.pos[2]) & 0xFC) njThrow(NJ_SYNTAX_ERROR);
		njSkip(3);
		nj.qtused |= 1 << c->qtsel;
		if (c->ssx > ssxmax) ssxmax = c->ssx;
		if (c->ssy > ssymax) ssymax = c->ssy;
	}
	if (nj.ncomp == 1) {
		c = nj.comp;
		c->ssx = c->ssy = ssxmax = ssymax = 1;
	}
	nj.mbsizex = ssxmax << 3;
	nj.mbsizey = ssymax << 3;
	nj.mbwidth = (nj.width + nj.mbsizex - 1) / nj.mbsizex;
	nj.mbheight = (nj.height + nj.mbsizey - 1) / nj.mbsizey;
	for (i = 0, c = nj.comp;  i < nj.ncomp;  ++i, ++c) {
		c->width = (nj.width * c->ssx + ssxmax - 1) / ssxmax;
		c->height = (nj.height * c->ssy + ssymax - 1) / ssymax;
		c->stride = nj.mbwidth * c->ssx << 3;
		if (((c->width < 3) && (c->ssx != ssxmax)) || ((c->height < 3) && (c->ssy != ssymax))) njThrow(NJ_UNSUPPORTED);
		if (!(c->pixels = (unsigned char*) njAllocMem(c->stride * nj.mbheight * c->ssy << 3))) njThrow(NJ_OUT_OF_MEM);
		if (!(c->intpixels = (int*) njAllocMem(sizeof(int) * c->stride * nj.mbheight * c->ssy << 3))) njThrow(NJ_OUT_OF_MEM);
	}
	if (nj.ncomp == 3) {
		nj.rgb = (unsigned char*) njAllocMem(nj.width * nj.height * nj.ncomp);
		if (!nj.rgb) njThrow(NJ_OUT_OF_MEM);
	}
	njSkip(nj.length);
}

/// Decode "Define Huffman Table" marker
NJ_INLINE void njDecodeDHT(void) {
	int codelen, currcnt, remain, spread, i, j;
	nj_vlc_code_t *vlc;
	static unsigned char counts[16];
	njDecodeLength();
	njCheckError();
	while (nj.length >= 17) {
		i = nj.pos[0];
		if (i & 0xEC) njThrow(NJ_SYNTAX_ERROR);
		if (i & 0x02) njThrow(NJ_UNSUPPORTED);
		i = (i | (i >> 3)) & 3;  // combined DC/AC + tableid value
		for (codelen = 1;  codelen <= 16;  ++codelen)
			counts[codelen - 1] = nj.pos[codelen];
		njSkip(17);
		vlc = &nj.vlctab[i][0];
		remain = spread = 65536;
		for (codelen = 1;  codelen <= 16;  ++codelen) {
			spread >>= 1;
			currcnt = counts[codelen - 1];
			if (!currcnt) continue;
			if (nj.length < currcnt) njThrow(NJ_SYNTAX_ERROR);
			remain -= currcnt << (16 - codelen);
			if (remain < 0) njThrow(NJ_SYNTAX_ERROR);
			for (i = 0;  i < currcnt;  ++i) {
				unsigned char code = nj.pos[i];
				for (j = spread;  j;  --j) {
					vlc->bits = (unsigned char) codelen;
					vlc->code = code;
					++vlc;
				}
			}
			njSkip(currcnt);
		}
		while (remain--) {
			vlc->bits = 0;
			++vlc;
		}
	}
	if (nj.length) njThrow(NJ_SYNTAX_ERROR);
}

/// Decode "Define Quantization Table" marker
NJ_INLINE void njDecodeDQT(void) {
	int i;
	unsigned char *t;
	njDecodeLength();
	njCheckError();
	while (nj.length >= 65) {
		i = nj.pos[0];
		if (i & 0xFC) njThrow(NJ_SYNTAX_ERROR);
		nj.qtavail |= 1 << i;
		t = &nj.qtab[i][0];
		for (i = 0;  i < 64;  ++i)
			t[i] = nj.pos[i + 1];
		njSkip(65);
	}
	if (nj.length) njThrow(NJ_SYNTAX_ERROR);
}

/// Decode "Define Restart Interval" marker
NJ_INLINE void njDecodeDRI(void) {
	njDecodeLength();
	njCheckError();
	if (nj.length < 2) njThrow(NJ_SYNTAX_ERROR);
	nj.rstinterval = njDecode16(nj.pos);
	njSkip(nj.length);
}

// Get Variable Length Code (VLC): decodes Huffman compression
static int njGetVLC(nj_vlc_code_t* vlc, unsigned char* code) {
	int value = njShowBits(16);
	int bits = vlc[value].bits;
	if (!bits) { nj.error = NJ_SYNTAX_ERROR; return 0; }
	njSkipBits(bits); // the correct number of bits for the code are consumed, even though 
	value = vlc[value].code;
	if (code) *code = (unsigned char) value;
	bits = value & 15;
	if (!bits) return 0;
	value = njGetBits(bits);
	if (value < (1 << (bits - 1)))
		value += ((-1) << bits) + 1;
	return value;
}

/// Read a block: Huffman decoding and zigzag only, to be followed by CUDA Row/ColIDCT
NJ_INLINE void njReadBlock(nj_component_t* c, int* out) {
	unsigned char code = 0, bx = 0, by = 0;
	int value, coef = 0;
	njFillMem(nj.block, 0, sizeof(nj.block)); // zero 8x8 block (OSS. only values !=0 are written)
	c->dcpred += njGetVLC(&nj.vlctab[c->dctabsel][0], NULL);
	//printf("njGetVLC (init) DC: c->dcpred=%02x\n", c->dcpred);
	nj.block[0] = (c->dcpred) * nj.qtab[c->qtsel][0];
	do {
		value = njGetVLC(&nj.vlctab[c->actabsel][0], &code);
		//printf("njGetVLC: value=%3d code=%3d; ", value, code);
		if (!code) break;  // EOB
		if (!(code & 0x0F) && (code != 0xF0)) njThrow(NJ_SYNTAX_ERROR);
		coef += (code >> 4) + 1;
		//printf("coef=%2d; ", coef);
		if (coef > 63) njThrow(NJ_SYNTAX_ERROR);

		//printf("i_block(zz)=%2d, val dequant=%d\n", njZZ[coef], value * nj.qtab[c->qtsel][coef]);
		nj.block[(int) njZZ[coef]] = value * nj.qtab[c->qtsel][coef]; // to copy directly to the output vector, njZZ (in [0:63]) would need to be njZZ_x and njZZ_y (both in [0:7])
	} while (coef < 63);
	for(coef=0, by=0; by<8; by++) // copy to output vector
	{
		for(bx=0; bx<8; bx++)
		{
			//printf("out copy bx=%d, by=%d: out=%08lx, out[%d] = nj.block[%d]\n", bx, by, (unsigned long) out, (by * c->stride + bx), coef);
			out[by * c->stride + bx] = nj.block[coef]; // [by * 8 + bx];
			coef++;
		}
	}
	// for (coef = 0;  coef < 64;  coef += 8)
	// 	njRowIDCT(&nj.block[coef]);
	// for (coef = 0;  coef < 8;  ++coef)
	// 	njColIDCT(&nj.block[coef], &out[coef], c->stride);
}

/// Decode a block: Huffman decoding, zigzag, de-quantization, iDCT (row, col)
NJ_INLINE void njDecodeBlock(nj_component_t* c, unsigned char* out) {
	unsigned char code = 0;
	int value, coef = 0;
	njFillMem(nj.block, 0, sizeof(nj.block));
	c->dcpred += njGetVLC(&nj.vlctab[c->dctabsel][0], NULL);
	nj.block[0] = (c->dcpred) * nj.qtab[c->qtsel][0];
	do {
		value = njGetVLC(&nj.vlctab[c->actabsel][0], &code);
		if (!code) break;  // EOB
		if (!(code & 0x0F) && (code != 0xF0)) njThrow(NJ_SYNTAX_ERROR);
		coef += (code >> 4) + 1;
		if (coef > 63) njThrow(NJ_SYNTAX_ERROR);
		nj.block[(int) njZZ[coef]] = value * nj.qtab[c->qtsel][coef];
	} while (coef < 63);
	for (coef = 0;  coef < 64;  coef += 8)
		njRowIDCT(&nj.block[coef]);
	for (coef = 0;  coef < 8;  ++coef)
		njColIDCT(&nj.block[coef], &out[coef], c->stride);
}

/// Read and decompress whole image (all blocks)
/// TODO separare in 4 stream, lanciare in parallelo tutti durante lettura (x3 componenti?)
NJ_INLINE void njCudaDecodeScan(void) {
	int i, mbx, mby, sbx, sby, stream_mby, stream_i, stream_n_mcb;
	dim3 dimBlock, dimGrid;
	int rstcount = nj.rstinterval, nextrst = 0;
	nj_component_t* c;
	njDecodeLength();
	njCheckError();
	if (nj.length < (4 + 2 * nj.ncomp)) njThrow(NJ_SYNTAX_ERROR);
	if (nj.pos[0] != nj.ncomp) njThrow(NJ_UNSUPPORTED);
	njSkip(1);
	for (i = 0, c = nj.comp;  i < nj.ncomp;  ++i, ++c) {
		if (nj.pos[0] != c->cid) njThrow(NJ_SYNTAX_ERROR);
		if (nj.pos[1] & 0xEE) njThrow(NJ_SYNTAX_ERROR);
		c->dctabsel = nj.pos[1] >> 4;
		c->actabsel = (nj.pos[1] & 1) | 2;
		njSkip(2);
	}
	if (nj.pos[0] || (nj.pos[1] != 63) || nj.pos[2]) njThrow(NJ_UNSUPPORTED);
	njSkip(nj.length);

	printf("Starting njCudaDecodeScan..........................\n");
	if(failed(cudaDeviceSynchronize())) // ==================================
		printf("sync after ColIDCT component %d failed.\n", i);
	for(i=0; i<nj.ncomp; i++)
	{
		c = &(nj.comp[i]);
		
		printf("(DISABLED) doing cudaHostRegister intpixels component %d ...\n", i);
		printf("size of locked memory: %d bytes\n", (int) (sizeof(int) * c->stride * nj.mbheight * c->ssy << 3));
		//if(failed(cudaHostRegister(c->intpixels, sizeof(int) * c->stride * nj.mbheight * c->ssy << 3, cudaHostRegisterDefault)))
		//	printf("cudaHostRegister intpixels component %d failed\n", i);

		printf("doing malloc cuintpixels component %d ...\n", i);
		if(failed(cudaMalloc((void**)&(c->cuintpixels), sizeof(int) * c->stride * nj.mbheight * c->ssy << 3))) // copy to GPU for IDFT
			printf("malloc cuintpixels component %d failed\n", i);
		
		printf("doing malloc cupixels component %d ...\n", i);
		if(failed(cudaMalloc((void**)&(c->cupixels), c->stride * nj.mbheight * c->ssy << 3))) // alloc cupixels for IDFT results
			printf("malloc cupixels component %d failed\n", i);
		//if(failed(cudaMemcpy( c->cupixels, c->pixels, c->stride * nj.mbheight * c->ssy << 3, cudaMemcpyHostToDevice )))
		//	printf("memcpy iniziale component failed\n");

		//if (!(c->intpixels = (int*) njAllocMem(sizeof(int) * c->stride * nj.mbheight * c->ssy << 3))) njThrow(NJ_OUT_OF_MEM); // allocata altrove prima
	}
	stream_n_mcb = (nj.mbheight + NSTR-1)/NSTR; // vertical MCBs per stream
	stream_i = 0;
	stream_mby = 0;
	for (mbx = mby = 0;;) { // for each block (minimum coded unit, o minimum block: 8x8 o 16x16 o altri)
		for (i = 0, c = nj.comp;  i < nj.ncomp;  ++i, ++c) // for each component in the image (Y,Cb,Cr)
			for (sby = 0;  sby < c->ssy;  ++sby)           // for each block in the minimum coded unit
				for (sbx = 0;  sbx < c->ssx;  ++sbx) {     // es. 1x1 normalmente, o 2x2 per Cb e Cr in 4:2:0
					//printf("readblock mbx=%4d mby=%4d, component %d sbx=%d sby=%d\n", mbx, mby, i, sbx, sby);
					njReadBlock(c, &(c->intpixels[((mby * c->ssy + sby) * c->stride + mbx * c->ssx + sbx) << 3]));
					njCheckError();
				}
		if (++mbx >= nj.mbwidth) {
			mbx = 0;
			mby++;
			stream_mby++;

			if(stream_mby >= stream_n_mcb || mby >= nj.mbheight)
			{
				// start row/col IDCT (on all components) ---- FOR THIS STREAM: 1/NSTR of the whole height ----
				for(i=0; i<nj.ncomp; i++)
				{
					c = &(nj.comp[i]);
					printf("  ==== starting IDCT part %d (made of %d vertical MCBs) component %d ====\n", stream_i, stream_mby, i);

					// TODO async
					printf("component %d: memcpy cuintpix          %08lx intpix          %08lx\n", i, (unsigned long) c->cuintpixels, (unsigned long) c->intpixels);
					printf("component %d: memcpy cuintpix w/offset %08lx intpix w/offset %08lx\n", i,
						(unsigned long) ((c->cuintpixels) + (stream_i * stream_n_mcb * c->stride * c->ssy << 3)),
						(unsigned long) ((c->intpixels) + (stream_i * stream_n_mcb * c->stride * c->ssy << 3)));
					if(failed(cudaMemcpy(      // OSS. advance memory pointers to only pick MCBs belonging to this stream
						(c->cuintpixels) + (stream_i * stream_n_mcb * c->stride * c->ssy << 3), // stream_i * stream_n_mcb == height raggiunta
						(c->intpixels) + (stream_i * stream_n_mcb * c->stride * c->ssy << 3),   // without sizeof(int), already intptr
						sizeof(int) * c->stride * stream_mby * c->ssy << 3, // only copy MCBs of this stream
						cudaMemcpyHostToDevice )))
						printf("memcpy cuintpixels component %d failed\n", i);
					
					//if(failed(cudaDeviceSynchronize())) // ================================== we want async copy
					//	printf("sync after memcpy cuintpixels component %d failed.\n", i);

					
					//if(failed(cudaDeviceSynchronize())) // ==================================
					//	printf("sync after UpsampleH component %d failed.\n", i);
					printf("component %d: row cuintpix          %08lx\n", i, (unsigned long) c->cuintpixels);
					printf("component %d: row cuintpix w/offset %08lx\n", i, (unsigned long) ((c->cuintpixels) + (stream_i * stream_n_mcb * c->stride * c->ssy << 3)));

					dimBlock = dim3 (4, 32);	// thread per grid cell (block): 4x32=128 thread per block (32x32 pixel elaborati)
					dimGrid = dim3 (((c->stride+7)/8 + 3)/4, ((stream_mby * c->ssy << 3) /*c->height*/+31)/32); // grid size (accounting for CUDA block size, and the 8 pixel per thread treated by RowIDCT)
					njCudaRowIDCT<<<dimGrid, dimBlock>>>(
						(c->cuintpixels) + (stream_i * stream_n_mcb * c->stride * c->ssy << 3),
						c->stride,
						(stream_mby * c->ssy << 3) /*c->height*/); // height: only MCBs of this stream

					if (failed(cudaPeekAtLastError()))
						printf("error RowIDCT component %d failed\n", i);
					
					printf("component %d: col cuintpix          %08lx cupix          %08lx\n", i, (unsigned long) c->cuintpixels, (unsigned long) c->cupixels);
					printf("component %d: col cuintpix w/offset %08lx cupix w/offset %08lx\n", i,
						(unsigned long) ((c->cuintpixels) + (stream_i * stream_n_mcb * c->stride * c->ssy << 3)),
						(unsigned long) ((c->cupixels) + (stream_i * stream_n_mcb * c->stride * c->ssy << 3)));
					dimBlock = dim3 (32, 4);	// thread per grid cell (block): 32x4=128 thread per block (32x32 pixel elaborati)
					dimGrid = dim3 ((c->stride + 31)/32, (((stream_mby * c->ssy << 3) /*c->height*/+7)/8 + 3)/4); // grid size (accounting for CUDA block size, and the 8 vertical pixel per thread treated by ColIDCT)
					njCudaColIDCT<<<dimGrid, dimBlock>>>(
						(c->cuintpixels) + (stream_i * stream_n_mcb * c->stride * c->ssy << 3),
						(c->cupixels) + (stream_i * stream_n_mcb * c->stride * c->ssy << 3),
						c->stride,
						(stream_mby * c->ssy << 3) /*c->height*/); // stream_n_mcb (or less for last stream)

					if (failed(cudaPeekAtLastError()))
						printf("error ColIDCT component %d failed\n", i);
					if(failed(cudaDeviceSynchronize())) // ==================================
						printf("sync after ColIDCT component %d failed.\n", i);
					if(failed(cudaFree(c->cuintpixels)))
						printf("free cuintpixels after IDCT component %d failed\n", i);
				}
				

				stream_mby=0;
				stream_i++;
			}
			if (mby >= nj.mbheight) break;
		}
		if (nj.rstinterval && !(--rstcount)) {
			njByteAlign();
			i = njGetBits(16);
			if (((i & 0xFFF8) != 0xFFD0) || ((i & 7) != nextrst)) njThrow(NJ_SYNTAX_ERROR);
			nextrst = (nextrst + 1) & 7;
			rstcount = nj.rstinterval;
			for (i = 0;  i < 3;  ++i)
				nj.comp[i].dcpred = 0;
		}
	}

	// TODO probabilmente CudaDeviceSynchronize?

	if(failed(cudaDeviceSynchronize())) // ================================== we want async copy
		printf("sync after memcpy cuintpixels component %d failed.\n", i);

	for(i=0; i<nj.ncomp; i++)
	{
		c = &(nj.comp[i]);

		if(failed(cudaHostUnregister(c->intpixels)))
			printf("cudaHostUnregister intpixels component %d failed\n", i);
	}
	
	/*
	for (mbx = mby = 0;;) { // for each block (minimum coded unit, o minimum block: 8x8 o 16x16 o altri)
		for (i = 0, c = nj.comp;  i < nj.ncomp;  ++i, ++c) // for each component in the image (Y,Cb,Cr)
			for (sby = 0;  sby < c->ssy;  ++sby)           // for each block in the minimum coded unit
				for (sbx = 0;  sbx < c->ssx;  ++sbx) {     // es. 1x1 normalmente, o 2x2 per Cb e Cr in 4:2:0
					njDecodeBlock(c, &c->pixels[((mby * c->ssy + sby) * c->stride + mbx * c->ssx + sbx) << 3]);
					njCheckError();
				}
		if (++mbx >= nj.mbwidth) {
			mbx = 0;
			if (++mby >= nj.mbheight) break;
		}
		if (nj.rstinterval && !(--rstcount)) {
			njByteAlign();
			i = njGetBits(16);
			if (((i & 0xFFF8) != 0xFFD0) || ((i & 7) != nextrst)) njThrow(NJ_SYNTAX_ERROR);
			nextrst = (nextrst + 1) & 7;
			rstcount = nj.rstinterval;
			for (i = 0;  i < 3;  ++i)
				nj.comp[i].dcpred = 0;
		}
	}
	*/
	nj.error = __NJ_FINISHED;
}

/// Read and decompress whole image (all blocks)
NJ_INLINE void njDecodeScan(void) {
	int i, mbx, mby, sbx, sby;
	int rstcount = nj.rstinterval, nextrst = 0;
	nj_component_t* c;
	njDecodeLength();
	njCheckError();
	if (nj.length < (4 + 2 * nj.ncomp)) njThrow(NJ_SYNTAX_ERROR);
	if (nj.pos[0] != nj.ncomp) njThrow(NJ_UNSUPPORTED);
	njSkip(1);
	for (i = 0, c = nj.comp;  i < nj.ncomp;  ++i, ++c) {
		if (nj.pos[0] != c->cid) njThrow(NJ_SYNTAX_ERROR);
		if (nj.pos[1] & 0xEE) njThrow(NJ_SYNTAX_ERROR);
		c->dctabsel = nj.pos[1] >> 4;
		c->actabsel = (nj.pos[1] & 1) | 2;
		njSkip(2);
	}
	if (nj.pos[0] || (nj.pos[1] != 63) || nj.pos[2]) njThrow(NJ_UNSUPPORTED);
	njSkip(nj.length);
	for (mbx = mby = 0;;) { // for each block (minimum coded unit, o minimum block: 8x8 o 16x16 o altri)
		for (i = 0, c = nj.comp;  i < nj.ncomp;  ++i, ++c) // for each component in the image (Y,Cb,Cr)
			for (sby = 0;  sby < c->ssy;  ++sby)           // for each block in the minimum coded unit
				for (sbx = 0;  sbx < c->ssx;  ++sbx) {     // es. 1x1 normalmente, o 2x2 per Cb e Cr in 4:2:0
					njDecodeBlock(c, &c->pixels[((mby * c->ssy + sby) * c->stride + mbx * c->ssx + sbx) << 3]);
					njCheckError();
				}
		if (++mbx >= nj.mbwidth) {
			mbx = 0;
			if (++mby >= nj.mbheight) break;
		}
		if (nj.rstinterval && !(--rstcount)) {
			njByteAlign();
			i = njGetBits(16);
			if (((i & 0xFFF8) != 0xFFD0) || ((i & 7) != nextrst)) njThrow(NJ_SYNTAX_ERROR);
			nextrst = (nextrst + 1) & 7;
			rstcount = nj.rstinterval;
			for (i = 0;  i < 3;  ++i)
				nj.comp[i].dcpred = 0;
		}
	}
	nj.error = __NJ_FINISHED;
}

#if NJ_CHROMA_FILTER

#define CF4A (-9)
#define CF4B (111)
#define CF4C (29)
#define CF4D (-3)
#define CF3A (28)
#define CF3B (109)
#define CF3C (-9)
#define CF3X (104)
#define CF3Y (27)
#define CF3Z (-3)
#define CF2A (139)
#define CF2B (-11)
#define CUCF(x) njCudaClip(((x) + 64) >> 7) // CUDA version, later this is redefined with the non-CUDA version
#define CF(x) njClip(((x) + 64) >> 7) // non-CUDA version

/// Made to be called one thread every 4 horizontal input pixels;
///   each thread produces 8 horizontal pixels.
/// It is advantageous to avoid divergence: make 2D blocks with x=1 so first & last column
///   fall all in one block. Or at worst 2x16?
/// @param[in]  width  width of input component, not multiple of 8 / possibly halved
/// @param[in]  height height of input component, not multiple of 8 / possibly halved
/// @param[in]  stride real width of input component pixels, multiple of 8 (if applying for the first time to component, else =width)
/// @param[in]  lin    component->cupixels (original size: at least  stride*height)
/// @param[out] lout   component->cupixels (double the width:        width*height*2)
__global__ void njCudaUpsampleH(unsigned char* lin, unsigned char* lout, int width, int height, int stride) {
	//const int xmax = c->width - 3;
	int x = (blockIdx.x*blockDim.x + threadIdx.x)*4; // original pixel x
	int y = blockIdx.y*blockDim.y + threadIdx.y; // original pixel y
	int iin = stride*y+x;
	int iout = (stride*y+x) << 1; // TODO questa è width no? Altrove?
	int i;
	//printf("UpsampleH x=%d y=%d, w=%d, h=%d, str=%d, in %08lx out %08lx\n", x, y, width, height, stride, (unsigned long) lin, (unsigned long) lout);
	if(y < height)
	{
		for(i=0; i<4 && x+i<width; i++, iin+=1, iout+=2) // elaborate (4px in, 8px out) for each thread, stopping at the end of img
		{
			//if(iout+1 >= width*height*2) // TODO rimuovere
			//	printf("UpsampleH iout %d out of bounds x=%d y=%d, w=%d, h=%d, str=%d, in %08lx out %08lx\n", iout, x, y, width, height, stride, (unsigned long) lin, (unsigned long) lout);
			//if(iin >= stride*height)
			//	printf("UpsampleH iin %d out of bounds x=%d y=%d, w=%d, h=%d, str=%d, in %08lx out %08lx\n", iin, x, y, width, height, stride, (unsigned long) lin, (unsigned long) lout);

			if(x+i == 0) // first pixel (*000)
			{
				lout[iout+0] = CUCF(CF2A * lin[iin+0] + CF2B * lin[iin+1]);                         // (offset = -2 ?)
				lout[iout+1] = CUCF(CF3X * lin[iin+0] + CF3Y * lin[iin+1] + CF3Z * lin[iin+2]);     // (offset = -1 ?)
			}
			else if(x+i == 1) // second pixel (0*00)
			{
				lout[iout+0] = CUCF(CF3A * lin[iin-1] + CF3B * lin[iin+0] + CF3C * lin[iin+1]);     // (offset = -1 ?)
				if(x+i == width-2) // second pixel is also second to last (0*0) (image ends right after it started: 3 column wide)
					lout[iout+1] = CUCF(CF3A * lin[iin+1] + CF3B * lin[iin+0] + CF3C * lin[iin-1]); // coeff in reverse order now
				else // normal second pixel (0*00)
					lout[iout+1] = CUCF(CF4A * lin[iin-1] + CF4B * lin[iin+0] + CF4C * lin[iin+1] + CF4D * lin[iin+2]); // offset=iin+1
			}
			else if(x+i == width-2) // second to last pixel (00*0) (3-wide image already handled in if(x+i==1))
			{
				lout[iout+0] = CUCF(CF4D * lin[iin-2] + CF4C * lin[iin-1] + CF4B * lin[iin+0] + CF4A * lin[iin+1]); // offset=iin+1
				lout[iout+1] = CUCF(CF3A * lin[iin+1] + CF3B * lin[iin+0] + CF3C * lin[iin-1]); // coeff in reverse order now
			}
			else if(x+i == width-1) // last pixel (000*)
			{
				lout[iout+0] = CUCF(CF3X * lin[iin-0] + CF3Y * lin[iin-1] + CF3Z * lin[iin-2]);
				lout[iout+1] = CUCF(CF2A * lin[iin-0] + CF2B * lin[iin-1]);
			}
			else // normal middle pixels (...00*00...)
			{
				lout[iout+0] = CUCF(CF4D * lin[iin-2] + CF4C * lin[iin-1] + CF4B * lin[iin+0] + CF4A * lin[iin+1]); // offset=iin+1
				lout[iout+1] = CUCF(CF4A * lin[iin-1] + CF4B * lin[iin+0] + CF4C * lin[iin+1] + CF4D * lin[iin+2]); // offset=iin+1
			}
		}
	}
	// TODO
	//width *= 2;
	//stride = width;
	// lin and lout should be different arrays of memory with the correct dimensions
}

/// Made to be called one thread every 4 vertical input pixels;
///   each thread produces 8 vertical pixels.
/// Note: cache danger!
/// @param[in]  width width of input component, not multiple of 8 / possibly halved
/// @param[in]  height height of input component, not multiple of 8 / possibly halved
/// @param[in]  stride real width of input component pixels, multiple of 8 (if applying for the first time to component, else =width)
/// @param[in]  cin component->pixels
/// @param[out] cout component->pixels double the width
__global__ void njCudaUpsampleV(unsigned char* cin, unsigned char* cout, int width, int height, int stride) {
	const int w = width, s1 = stride, s2 = s1 + s1; // stride, double stride (oss after UpsampleH() stride=width)
	int x = blockIdx.x*blockDim.x + threadIdx.x;       // original pixel x
	int y = (blockIdx.y*blockDim.y + threadIdx.y)*4;   // original pixel y (one thread every 4 pixels in vertical)
	int iin = stride*y+x;
	int iout = stride*y*2 + x; // two output rows for each input row (y), but only one output pixel per input pixel
	int i;
	//out = (unsigned char*) njAllocMem((c->width * c->height) << 1);
	//printf("UpsampleV x=%d y=%d, w=%d, h=%d, str=%d, in %08lx out %08lx\n", x, y, width, height, stride, (unsigned long) cin, (unsigned long) cout);
	if(x < width)
	{
		for(i=0; i<4 && y+i<height; i++, iin+=s1, iout+=2*width) // elaborate (4px in, 8px out) for each thread, stopping at the end of img
		{
			//if(iout+1 >= width*height*2) // TODO rimuovere
			//	printf("UpsampleV iout %d out of bounds x=%d y=%d, w=%d, h=%d, str=%d, in %08lx out %08lx\n", iout, x, y, width, height, stride, (unsigned long) cin, (unsigned long) cout);
			//if(iin >= stride*height)
			//	printf("UpsampleV iin %d out of bounds x=%d y=%d, w=%d, h=%d, str=%d, in %08lx out %08lx\n", iin, x, y, width, height, stride, (unsigned long) cin, (unsigned long) cout);

			if(y+i == 0) // first pixel (*000)
			{
				cout[iout  ] = CUCF(CF2A * cin[iin] + CF2B * cin[iin+s1]);
				cout[iout+w] = CUCF(CF3X * cin[iin] + CF3Y * cin[iin+s1] + CF3Z * cin[iin+s2]);
			}
			else if(y+i == 1) // second pixel (0*00)
			{
				cout[iout  ] = CUCF(CF3A * cin[iin-s1] + CF3B * cin[iin+0] + CF3C * cin[iin+s1]);     // (offset = -1 ?)
				if(x+i == height-2) // second pixel is also second to last (0*0) (image ends right after it started: 3 column wide)
					cout[iout+w] = CUCF(CF3A * cin[iin+s1] + CF3B * cin[iin   ] + CF3C * cin[iin-s1]);
				else // normal second pixel (0*00)
					cout[iout+w] = CUCF(CF4A * cin[iin-s1] + CF4B * cin[iin +0] + CF4C * cin[iin+s1] + CF4D * cin[iin+s2]);
			}
			else if(y+i == height-2) // second to last pixel (00*0) (3-wide image already handled in if(x+i==1))
			{
				cout[iout  ] = CUCF(CF4D * cin[iin-s2] + CF4C * cin[iin-s1] + CF4B * cin[iin +0] + CF4A * cin[iin+s1]);
				cout[iout+w] = CUCF(CF3A * cin[iin+s1] + CF3B * cin[iin +0] + CF3C * cin[iin-s1]);
			}
			else if(y+i == height-1) // last pixel (000*)
			{
				cout[iout  ] = CUCF(CF3X * cin[iin-0] + CF3Y * cin[iin-s1] + CF3Z * cin[iin-s2]);
				cout[iout+w] = CUCF(CF2A * cin[iin-0] + CF2B * cin[iin-s1]);
			}
			else // normal middle pixels (...00*00...)
			{
				cout[iout  ] = CUCF(CF4D * cin[iin-s2] + CF4C * cin[iin-s1] + CF4B * cin[iin +0] + CF4A * cin[iin+s1]);
				cout[iout+w] = CUCF(CF4A * cin[iin-s1] + CF4B * cin[iin +0] + CF4C * cin[iin+s1] + CF4D * cin[iin+s2]);
			}
		}
	}
	//c->height <<= 1;
	//c->stride = c->width;
	//c->pixels = out;
}

NJ_INLINE void njUpsampleH(nj_component_t* c) {
	const int xmax = c->width - 3;
	unsigned char *out, *lin, *lout;
	int x, y;
	out = (unsigned char*) njAllocMem((c->width * c->height) << 1);
	if (!out) njThrow(NJ_OUT_OF_MEM);
	lin = c->pixels;
	lout = out;
	for (y = c->height;  y;  --y) {
		lout[0] = CF(CF2A * lin[0] + CF2B * lin[1]);
		lout[1] = CF(CF3X * lin[0] + CF3Y * lin[1] + CF3Z * lin[2]);
		lout[2] = CF(CF3A * lin[0] + CF3B * lin[1] + CF3C * lin[2]);
		for (x = 0;  x < xmax;  ++x) {
			lout[(x << 1) + 3] = CF(CF4A * lin[x] + CF4B * lin[x + 1] + CF4C * lin[x + 2] + CF4D * lin[x + 3]);
			lout[(x << 1) + 4] = CF(CF4D * lin[x] + CF4C * lin[x + 1] + CF4B * lin[x + 2] + CF4A * lin[x + 3]);
		}
		lin += c->stride;
		lout += c->width << 1;
		lout[-3] = CF(CF3A * lin[-1] + CF3B * lin[-2] + CF3C * lin[-3]);
		lout[-2] = CF(CF3X * lin[-1] + CF3Y * lin[-2] + CF3Z * lin[-3]);
		lout[-1] = CF(CF2A * lin[-1] + CF2B * lin[-2]);
	}
	c->width <<= 1;
	c->stride = c->width;
	njFreeMem((void*)c->pixels);
	c->pixels = out;
}

NJ_INLINE void njUpsampleV(nj_component_t* c) {
	const int w = c->width, s1 = c->stride, s2 = s1 + s1;
	unsigned char *out, *cin, *cout;
	int x, y;
	out = (unsigned char*) njAllocMem((c->width * c->height) << 1);
	if (!out) njThrow(NJ_OUT_OF_MEM);
	for (x = 0;  x < w;  ++x) {
		cin = &c->pixels[x];
		cout = &out[x];
		*cout = CF(CF2A * cin[0] + CF2B * cin[s1]);  cout += w;
		*cout = CF(CF3X * cin[0] + CF3Y * cin[s1] + CF3Z * cin[s2]);  cout += w;
		*cout = CF(CF3A * cin[0] + CF3B * cin[s1] + CF3C * cin[s2]);  cout += w;
		cin += s1;
		for (y = c->height - 3;  y;  --y) {
			*cout = CF(CF4A * cin[-s1] + CF4B * cin[0] + CF4C * cin[s1] + CF4D * cin[s2]);  cout += w;
			*cout = CF(CF4D * cin[-s1] + CF4C * cin[0] + CF4B * cin[s1] + CF4A * cin[s2]);  cout += w;
			cin += s1;
		}
		cin += s1;
		*cout = CF(CF3A * cin[0] + CF3B * cin[-s1] + CF3C * cin[-s2]);  cout += w;
		*cout = CF(CF3X * cin[0] + CF3Y * cin[-s1] + CF3Z * cin[-s2]);  cout += w;
		*cout = CF(CF2A * cin[0] + CF2B * cin[-s1]);
	}
	c->height <<= 1;
	c->stride = c->width;
	njFreeMem((void*) c->pixels);
	c->pixels = out;
}

#else

NJ_INLINE void njUpsample(nj_component_t* c) {
	int x, y, xshift = 0, yshift = 0;
	unsigned char *out, *lin, *lout;
	while (c->width < nj.width) { c->width <<= 1; ++xshift; }
	while (c->height < nj.height) { c->height <<= 1; ++yshift; }
	out = (unsigned char*) njAllocMem(c->width * c->height);
	if (!out) njThrow(NJ_OUT_OF_MEM);
	lin = c->pixels;
	lout = out;
	for (y = 0;  y < c->height;  ++y) {
		lin = &c->pixels[(y >> yshift) * c->stride];
		for (x = 0;  x < c->width;  ++x)
			lout[x] = lin[x >> xshift];
		lout += c->width;
	}
	c->stride = c->width;
	njFreeMem((void*) c->pixels);
	c->pixels = out;
}

#endif

#define PX_PER_THREAD 16

/// Expects to be called one thread for every PX_PER_THREAD horizontal pixels
__global__ void nj_ycbcr_to_rgb(
	unsigned char* py, unsigned char* pcb, unsigned char* pcr,
	int ystride, int cbstride, int crstride,
	unsigned char* rgbout, int width, int height
)
{
	int i;
	//unsigned char *py, *pcb, *pcr;
	int vy, vcb, vcr;
	int x = (blockIdx.x*blockDim.x + threadIdx.x)*PX_PER_THREAD;  // original pixel x (one thread every PX_PER_THREAD pixels in horizontal)
	int y = blockIdx.y*blockDim.y + threadIdx.y;                  // original pixel y
	
	// if(x==0 && y==0)
	// {
	// 	for(i=0; i<PX_PER_THREAD; i++)
	// 		printf("before nj_ycbcr_to_rgb: %3d %3d %3d\n", iny[i], incb[i], incr[i]);
	// }

	if(y < height)
	{
		//if(x==0 && y==0)
		//	printf("nj_ycbcr_to_rgb x=%d y=%d w=%d h=%d\n", x, y, width, height);

		// find starting pointers
		// Aritmetica sui puntatori non era il problema
		//py   = iny  + ystride  * y + x; // single component: one byte each
		//pcb  = incb + cbstride * y + x;
		//pcr  = incr + crstride * y + x;
		
		py   += ystride  * y + x; // single component: one byte each
		pcb  += cbstride * y + x;
		pcr  += crstride * y + x;
		rgbout += (width * y + x) *3; // rgb: 3 byte each


		// convert (up to) PX_PER_THREAD pixels in this thread
		for(i=0; i<PX_PER_THREAD && x < width; i++, x++, py++, pcb++, pcr++, rgbout+=3)
		{
			vy  = *py  << 8;
			vcb = *pcb - 128;
			vcr = *pcr - 128;
			//vy  = iny [ystride  * y + x] << 8;
			//vcb = incb[cbstride * y + x] - 128;
			//vcr = incr[crstride * y + x] - 128;
			// if(x<16 && y==0)
			// 	printf("nj_ycbcr_to_rgb x=%3d y=%3d YCbCr: (%3d %3d %3d) %5d %5d %5d, PTR: %08lx, %08lx, %08lx, rgbout: %08lx\n", x, y, *py, *pcb, *pcr, vy, vcb, vcr, (unsigned long) py, (unsigned long) pcb, (unsigned long) pcr, (unsigned long) rgbout);
			// 	//printf("nj_ycbcr_to_rgb x=%3d y=%3d YCbCr: %5d %5d %5d, PTR: %08lx, %08lx, %08lx, rgbout: %08lx\n", x, y, vy, vcb, vcr, (unsigned long) rgbout);
			rgbout[0] = njCudaClip((vy             + 359 * vcr + 128) >> 8);
			rgbout[1] = njCudaClip((vy -  88 * vcb - 183 * vcr + 128) >> 8);
			rgbout[2] = njCudaClip((vy + 454 * vcb             + 128) >> 8);
			//rgbout[0 + (width * y + x) *3] = njCudaClip((vy             + 359 * vcr + 128) >> 8); // TODO provare senza parentesi come dice Fulvio (width tipo stride)
			//rgbout[1 + (width * y + x) *3] = njCudaClip((vy -  88 * vcb - 183 * vcr + 128) >> 8);
			//rgbout[2 + (width * y + x) *3] = njCudaClip((vy + 454 * vcb             + 128) >> 8);
			//rgbout[0 + (width * y + x) *3] = vy; // TODO rimuovere
			//rgbout[2 + (width * y + x) *3] = vy;
			//rgbout[1 + (width * y + x) *3] = vy;
		}
	}
}

NJ_INLINE void njCudaConvert(void) {
	int i;
	nj_component_t* c;
	unsigned char* newvec;
	//dim3 dimBlock (16, 16);	//roundup 
	//dim3 dimGrid ((n_blocks + 255)/256, 1);
	//dim3 dimGridCbCr ((CbCr_blocks + 255)/256, 1);

	//for(i=0; i<16;i++)
	//	printf("Prima, YCbCr: %3d %3d %3d\n", nj.comp[0].pixels[i], nj.comp[1].pixels[i], nj.comp[2].pixels[i]);
	
	if(failed(cudaMalloc((void**)&(nj.curgb), nj.width * nj.height * 3))) // temporary memcpy to try this CUDA version
		printf("malloc curgb failed\n");
	
	for (i = 0, c = nj.comp;  i < nj.ncomp;  ++i, ++c) {
		//printf("component %d: stride %d, mbheight %d, ssy %d\n", i, c->stride, nj.mbheight, c->ssy);

		// if(failed(cudaMalloc((void**)&(c->cupixels), c->stride * nj.mbheight * c->ssy << 3))) // temporary memcpy to try this CUDA version, moved to njCudaDecodeScan()
		// 	printf("malloc component failed\n");
		// if(failed(cudaMemcpy( c->cupixels, c->pixels, c->stride * nj.mbheight * c->ssy << 3, cudaMemcpyHostToDevice )))
		// 	printf("initial memcpy component failed\n");
		
		// if(failed(cudaDeviceSynchronize())) // ==================================
		// 	printf("sync after UpsampleH component %d failed.\n", i);
		//printf("component %d: pix %08lx cupix %08lx\n", i, (unsigned long) c->pixels, (unsigned long) c->cupixels);

		//#if NJ_CHROMA_FILTER
			while ((c->width < nj.width) || (c->height < nj.height)) {
				if (c->width < nj.width)
				{
					if(failed(cudaMalloc((void**)&newvec, c->width * c->height * 2)))
						printf("malloc newvec component horizontal realloc failed\n");

					dim3 dimBlock (8, 32);	// thread per grid cell: 8x32=256 thread per grid
					dim3 dimGrid (((nj.width+3)/4 + 7)/8, (nj.height+31)/32); // grid size

					//printf("UpsampleH dimGrid %dx%d dimBlock %dx%d\n", dimGrid.x, dimGrid.y, dimBlock.x, dimBlock.y);
					//printf("component %d: pix %08lx cupix %08lx, newvec %08lx 2a print\n", i, (unsigned long) c->pixels, (unsigned long) c->cupixels, (unsigned long) newvec);
					
					njCudaUpsampleH<<<dimGrid, dimBlock>>>(c->cupixels, newvec, c->width, c->height, c->stride); // TODO call it better
					
					if (failed(cudaPeekAtLastError()))
						printf("error UpsampleH component %d failed\n", i);
					if(failed(cudaDeviceSynchronize())) // ==================================
						printf("sync after UpsampleH component %d failed.\n", i);
					if(failed(cudaFree(c->cupixels)))
						printf("free cupixels UpsampleH component %d failed\n", i);
					c->cupixels = newvec;
					c->width *= 2;
					c->stride = c->width;
					//c->pixels = (unsigned char *) realloc(c->pixels, c->stride*c->height); // TODO rimuovere
				}
				njCheckError();
				if (c->height < nj.height)
				{
					if(failed(cudaMalloc((void**)&newvec, c->width * c->height * 2)))
						printf("malloc newvec component vertical realloc failed\n");

					dim3 dimBlock (32, 8);	// thread per grid cell
					dim3 dimGrid ((nj.width + 31)/32, ((nj.height+3)/4 + 7)/8); // grid size

					//printf("UpsampleV dimGrid %dx%d dimBlock %dx%d\n", dimGrid.x, dimGrid.y, dimBlock.x, dimBlock.y);
					//printf("component %d: pix %08lx cupix %08lx, newvec %08lx 3a print\n", i, (unsigned long) c->pixels, (unsigned long) c->cupixels, (unsigned long) newvec);

					njCudaUpsampleV<<<dimGrid, dimBlock>>>(c->cupixels, newvec, c->width, c->height, c->stride); // TODO call it better

					if(failed(cudaPeekAtLastError()))
						printf("error UpsampleV component %d failed\n", i);
					if(failed(cudaDeviceSynchronize())) // ==================================
						printf("sync after UpsampleV component %d failed.\n", i);
					
					if(failed(cudaFree(c->cupixels)))
						printf("free cupixels UpsampleV component %d failed\n", i);
					c->cupixels = newvec;
					c->height *= 2;
					c->stride = c->width;
					//c->pixels = (unsigned char *) realloc(c->pixels, c->stride*c->height); // TODO rimuovere
				}
				njCheckError();
			}
		//#else
		//	if ((c->width < nj.width) || (c->height < nj.height))
		//		njUpsample(c);
		//#endif
		if ((c->width < nj.width) || (c->height < nj.height)) njThrow(NJ_INTERNAL_ERR);

		if (failed(cudaPeekAtLastError()))
        	printf("peek last error failed alla fine del ciclo component %d\n", i);
		//memset(c->pixels, 0, c->stride * c->height); // TODO rimuovere solo diagnostica
		//printf("copy of %d byte component %d.\n", c->stride * c->height, i);
		//if(failed(cudaMemcpy( c->pixels, c->cupixels, c->stride * c->height, cudaMemcpyDeviceToHost ))) // TODO rimuovere
		//	printf("final temporary memcpy component failed, pixels=%08lx, cupix=%08lx\n", (unsigned long) c->pixels, (unsigned long) c->cupixels);
	} // end foreach component

	//for(i=0; i<16;i++)
	//	printf("dopo subsample, YCbCr: %3d %3d %3d\n", nj.comp[0].pixels[i], nj.comp[1].pixels[i], nj.comp[2].pixels[i]);
	
	if (nj.ncomp == 3) {
		// convert to RGB (8-stride may be already removed either horizontally or vertically in Upsample)

		dim3 dimBlock (8, 32);	// thread per grid cell: 8x32=256 thread per grid
		dim3 dimGrid (((nj.width+PX_PER_THREAD-1)/PX_PER_THREAD + 7)/8, (nj.height+31)/32);
		//dim3 dimBlock (1, 8);	// thread per grid cell: 8x32=256 thread per grid
		//dim3 dimGrid (1, 1);

		//printf("nj_ycbcr_to_rgb block %dx%d, dimGrid %dx%d\n", dimBlock.x, dimBlock.y, dimGrid.x, dimGrid.y);
		//printf("  cupixels: %08lx, %08lx, %08lx\n", (unsigned long) nj.comp[0].cupixels, (unsigned long) nj.comp[1].cupixels, (unsigned long) nj.comp[2].cupixels);
		//printf("  strides:  %8d, %8d, %8d\n", nj.comp[0].stride, nj.comp[1].stride, nj.comp[2].stride);
		//printf("  curgb:    %08lx, w=%d h=%d\n", (unsigned long) nj.curgb, nj.width, nj.height);

		nj_ycbcr_to_rgb <<<dimGrid, dimBlock>>>( // TODO chiamare meglio: stream
			nj.comp[0].cupixels, nj.comp[1].cupixels, nj.comp[2].cupixels,
			nj.comp[0].stride, nj.comp[1].stride, nj.comp[2].stride,
			nj.curgb, nj.width, nj.height
		);

		if(failed(cudaPeekAtLastError()))
			printf("error nj_ycbcr_to_rgb failed\n");
		if(failed(cudaDeviceSynchronize())) // ==================================
			printf("sync after nj_ycbcr_to_rgb failed.\n");
		
		if(failed(cudaMemcpy(nj.rgb, nj.curgb, nj.width * nj.height * 3, cudaMemcpyDeviceToHost)))
			printf("memcpy rgb d2host failed\n");
		cudaFree(nj.curgb);
	} else if (nj.comp[0].width != nj.comp[0].stride) {
		// grayscale -> only remove 8-stride
		unsigned char *pin = &nj.comp[0].pixels[nj.comp[0].stride];
		unsigned char *pout = &nj.comp[0].pixels[nj.comp[0].width];
		int y;
		for (y = nj.comp[0].height - 1;  y;  --y) {
			njCopyMem(pout, pin, nj.comp[0].width);
			pin += nj.comp[0].stride;
			pout += nj.comp[0].width;
		}
		nj.comp[0].stride = nj.comp[0].width;
	}
}

NJ_INLINE void njConvert(void) {
	int i;
	nj_component_t* c;
	for (i = 0, c = nj.comp;  i < nj.ncomp;  ++i, ++c) {
		#if NJ_CHROMA_FILTER
			while ((c->width < nj.width) || (c->height < nj.height)) {
				if (c->width < nj.width) njUpsampleH(c);
				njCheckError();
				if (c->height < nj.height) njUpsampleV(c);
				njCheckError();
			}
		#else
			if ((c->width < nj.width) || (c->height < nj.height))
				njUpsample(c);
		#endif
		if ((c->width < nj.width) || (c->height < nj.height)) njThrow(NJ_INTERNAL_ERR);
	}
	if (nj.ncomp == 3) {
		// convert to RGB (8-stride may be already removed either horizontally or vertically in Upsample)
		int x, yy;
		unsigned char *prgb = nj.rgb;
		const unsigned char *py  = nj.comp[0].pixels;
		const unsigned char *pcb = nj.comp[1].pixels;
		const unsigned char *pcr = nj.comp[2].pixels;
		for (yy = nj.height;  yy;  --yy) {
			for (x = 0;  x < nj.width;  ++x) {
				int y = py[x] << 8;
				int cb = pcb[x] - 128;
				int cr = pcr[x] - 128;
				*prgb++ = njClip((y            + 359 * cr + 128) >> 8);
				*prgb++ = njClip((y -  88 * cb - 183 * cr + 128) >> 8);
				*prgb++ = njClip((y + 454 * cb            + 128) >> 8);
			}
			py += nj.comp[0].stride;
			pcb += nj.comp[1].stride;
			pcr += nj.comp[2].stride;
		}
	} else if (nj.comp[0].width != nj.comp[0].stride) {
		// grayscale -> only remove 8-stride
		unsigned char *pin = &nj.comp[0].pixels[nj.comp[0].stride];
		unsigned char *pout = &nj.comp[0].pixels[nj.comp[0].width];
		int y;
		for (y = nj.comp[0].height - 1;  y;  --y) {
			njCopyMem(pout, pin, nj.comp[0].width);
			pin += nj.comp[0].stride;
			pout += nj.comp[0].width;
		}
		nj.comp[0].stride = nj.comp[0].width;
	}
}

void njInit(int use_cuda) {
	int i;

	njFillMem(&nj, 0, sizeof(nj_context_t));
	nj.use_cuda = use_cuda;

	if(nj.use_cuda)
	{
		for(i=0; i<NSTR; i++)
		{
			printf("doing cudaStreamCreate(%016lx) stream %d ... ", (unsigned long) &(nj.custreams[i]), i); // TODO togliere debug
			if(failed(cudaStreamCreate(&(nj.custreams[i]))))
				printf("failed cudaStreamCreate stream %d\n", i);
			printf("done cudaStreamCreate(%016lx) stream %d .\n", (unsigned long) nj.custreams[i], i);
		}
	}
}

void njDone(void) {
	int i;
	if(nj.use_cuda)
	{
		for(i=0; i<NSTR; i++)
		{
			printf("doing cudaStreamDestroy(%016lx) stream %d ... ", (unsigned long) &(nj.custreams[i]), i); // TODO togliere debug
			if(failed(cudaStreamDestroy(nj.custreams[i])))
				printf("failed cudaStreamDestroy stream %d\n", i);
			printf("done cudaStreamCreate(%016lx) stream %d .\n", (unsigned long) nj.custreams[i], i);
		}
	}
	for (i = 0;  i < 3;  ++i) // TODO non dovrebbe essere i < nj.ncomp?
	{
		if (nj.comp[i].pixels) njFreeMem((void*) nj.comp[i].pixels);
		if (nj.comp[i].intpixels) njFreeMem((void*) nj.comp[i].intpixels);
	}
	if (nj.rgb) njFreeMem((void*) nj.rgb);
	njInit(nj.use_cuda);
}

/// Main call to decompress a JPEG
nj_result_t njDecode(const void* jpeg, const int size) {
	njDone();
	if(nj.use_cuda)
		cudaDeviceReset();
	nj.pos = (const unsigned char*) jpeg;
	nj.size = size & 0x7FFFFFFF;
	printf("use_cuda=%d size=%d magic=%02x %02x\n", nj.use_cuda, nj.size, (unsigned) nj.pos[0], (unsigned) nj.pos[1]);
	if (nj.size < 2) return NJ_NO_JPEG;
	if ((nj.pos[0] ^ 0xFF) | (nj.pos[1] ^ 0xD8)) return NJ_NO_JPEG;
	njSkip(2);
	while (!nj.error) {
		if ((nj.size < 2) || (nj.pos[0] != 0xFF)) return NJ_SYNTAX_ERROR;
		njSkip(2);
		switch (nj.pos[-1]) {
			case 0xC0: njDecodeSOF();  break;
			case 0xC4: njDecodeDHT();  break;
			case 0xDB: njDecodeDQT();  break;
			case 0xDD: njDecodeDRI();  break;
			case 0xDA:
				if(nj.use_cuda) njCudaDecodeScan();
				else njDecodeScan();
				break; // CUDA mod
			case 0xFE: njSkipMarker(); break;
			default:
				if ((nj.pos[-1] & 0xF0) == 0xE0)
					njSkipMarker();
				else
					return NJ_UNSUPPORTED;
		}
	}
	if (nj.error != __NJ_FINISHED) return nj.error;
	nj.error = NJ_OK;

	if(nj.use_cuda)
		njCudaConvert();
	else
		njConvert();
	return nj.error;
}

int njGetWidth(void)            { return nj.width; }
int njGetHeight(void)           { return nj.height; }
int njIsColor(void)             { return (nj.ncomp != 1); }
unsigned char* njGetImage(void) { return (nj.ncomp == 1) ? nj.comp[0].pixels : nj.rgb; }
int njGetImageSize(void)        { return nj.width * nj.height * nj.ncomp; }

// Call tree:
//
// njDecode()
//   -> njDecodeSOF/DHT/DQT/DRI()
//   -> njDecodeScan()
//     -> njDecodeBlock()
//       -> njGetVLC()
//       -> njRowIDCT()
//       -> njColIDCT()
//   -> njSkipMarker() chissenefrega
//   -> njConvert()
//     -> njUpsample/H/V()
//     -> njClip()
