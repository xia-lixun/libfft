/*
 
 This file is part of FFTS -- The Fastest Fourier Transform in the South
  
 Copyright (c) 2012, Anthony M. Blake <amb@anthonix.com>
 Copyright (c) 2012, The University of Waikato 
 
 All rights reserved.

 Redistribution and use in source and binary forms, with or without
 modification, are permitted provided that the following conditions are met:
 	* Redistributions of source code must retain the above copyright
 		notice, this list of conditions and the following disclaimer.
 	* Redistributions in binary form must reproduce the above copyright
 		notice, this list of conditions and the following disclaimer in the
 		documentation and/or other materials provided with the distribution.
 	* Neither the name of the organization nor the
	  names of its contributors may be used to endorse or promote products
 		derived from this software without specific prior written permission.

 THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
 ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 DISCLAIMED. IN NO EVENT SHALL ANTHONY M. BLAKE BE LIABLE FOR ANY
 DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
 ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

*/

#ifndef __SSE_FLOAT_H__
#define __SSE_FLOAT_H__

#include <immintrin.h>



#define VADD_ _mm256_add_ps
#define VSUB_ _mm256_sub_ps
#define VMUL_ _mm256_mul_ps
#define VXOR_ _mm256_xor_ps

#define VST_ _mm256_store_ps
#define VLD_ _mm256_load_ps


#define VADD _mm_add_ps
#define VSUB _mm_sub_ps
#define VMUL _mm_mul_ps
#define VXOR _mm_xor_ps

#define VST _mm_store_ps
#define VLD _mm_load_ps

#define VSWAPPAIRS(x) (_mm_shuffle_ps(x,x,_MM_SHUFFLE(2,3,0,1)))
#define VSWAPPAIRS_(x) (_mm256_shuffle_ps(x,x,_MM_SHUFFLE(2,3,0,1)))

#define VUNPACKHI(x,y) (_mm_shuffle_ps(x,y,_MM_SHUFFLE(3,2,3,2)))
#define VUNPACKLO(x,y) (_mm_shuffle_ps(x,y,_MM_SHUFFLE(1,0,1,0)))

#define VBLEND(x,y) (_mm_shuffle_ps(x,y,_MM_SHUFFLE(3,2,1,0)))

#define VLIT4 _mm_set_ps
#define VLIT8 _mm256_set_ps

#define VDUPRE(r) (_mm_shuffle_ps(r,r,_MM_SHUFFLE(2,2,0,0)))
#define VDUPIM(r) (_mm_shuffle_ps(r,r,_MM_SHUFFLE(3,3,1,1)))

#define FFTS_MALLOC(d,a) (_mm_malloc(d,a))
#define FFTS_FREE(d) (_mm_free(d))




// INPUT:
//+------+------+------+------+
//|  im1 |  re1 |  im0 |  re0 |    x   (sign)i  
//+------+------+------+------+
//
// OUTPUT:
//+------+------+------+------+
//| c1 * (sig)i | c0 * (sig)i |  
//+------+------+------+------+

static __inline __m128 IMULI(int inv, __m128 a)
{
	if (inv) return VSWAPPAIRS(VXOR(a, VLIT4(0.0f, -0.0f, 0.0f, -0.0f)));
	else    return VSWAPPAIRS(VXOR(a, VLIT4(-0.0f, 0.0f, -0.0f, 0.0f)));
}

static __inline __m256 IMULI_(int inv, __m256 a)
{
	if (inv)
		return VSWAPPAIRS_(VXOR_(a, VLIT8(0.0f, -0.0f, 0.0f, -0.0f, 0.0f, -0.0f, 0.0f, -0.0f)));
	else
		return VSWAPPAIRS_(VXOR_(a, VLIT8(-0.0f, 0.0f, -0.0f, 0.0f, -0.0f, 0.0f, -0.0f, 0.0f)));
}



// FUNCTION: TWO COMPLEX FLOAT MULTIPLIED BY A COMMON COMPLEX VALUE
//           IMAGINARY MUST BE SET AS VXOR(im, VLIT4(-0.0f, 0.0f, -0.0f, 0.0f))
//
// INPUT:
//+------+------+------+------+
//|  im1 |  re1 |  im0 |  re0 |    x   (re + im*i)  
//+------+------+------+------+
//
// OUTPUT:
//+------+------+------+------+
//|      c1     |      c0     |  
//+------+------+------+------+
static __inline __m128 IMUL(__m128 d, __m128 re, __m128 im) 
{
	re = VMUL(re, d);
	im = VMUL(im, VSWAPPAIRS(d));
	return VSUB(re, im);
}

static __inline __m256 IMUL_(__m256 d, __m256 re, __m256 im)
{
	re = VMUL_(re, d);
	im = VMUL_(im, VSWAPPAIRS_(d));
	return VSUB_(re, im);
}




// FUNCTION: TWO COMPLEX FLOAT MULTIPLIED BY A COMMON (CONJUGATED) COMPLEX VALUE
//           IMAGINARY MUST BE SET AS VXOR(im, VLIT4(-0.0f, 0.0f, -0.0f, 0.0f))
//
// INPUT:
//+------+------+------+------+
//|  im1 |  re1 |  im0 |  re0 |    x   (re - im*i)  
//+------+------+------+------+
//
// OUTPUT:
//+------+------+------+------+
//|      c1     |      c0     |  
//+------+------+------+------+
static __inline __m128 IMULJ(__m128 d, __m128 re, __m128 im) 
{
	re = VMUL(re, d);
	im = VMUL(im, VSWAPPAIRS(d));
	return VADD(re, im);
}

static __inline __m256 IMULJ_(__m256 d, __m256 re, __m256 im)
{
	re = VMUL_(re, d);
	im = VMUL_(im, VSWAPPAIRS_(d));
	return VADD_(re, im);
}





//do 2-complex numbers in one iteration
static __inline void K_N(int inv, __m128 re, __m128 im, __m128 *r0, __m128 *r1, __m128 *r2, __m128 *r3)
{
	__m128 uk, uk2, zk_p, zk_n, zk, zk_d;
	
	uk = *r0; 
	uk2 = *r1;
	
	zk_p = IMUL(*r2, re, im);   //Wnk*r2
	zk_n = IMULJ(*r3, re, im);  //conj(Wnk)*r3

	zk = VADD(zk_p, zk_n);
	zk_d = IMULI(inv, VSUB(zk_p, zk_n));

	*r2 = VSUB(uk, zk);
	*r0 = VADD(uk, zk);
	*r3 = VADD(uk2, zk_d);
	*r1 = VSUB(uk2, zk_d);
}


// do 4-complex numbers in one iteration of complex conjugate pair algorithm
static __inline void K_N_(int inv, __m256 re, __m256 im, __m256 *r0, __m256 *r1, __m256 *r2, __m256 *r3)
{
	__m256 uk, uk2, zk_p, zk_n, zk, zk_d;

	uk = *r0; 
	uk2 = *r1;

	zk_p = IMUL_(*r2, re, im);   //Wnk*r2
	zk_n = IMULJ_(*r3, re, im);  //conj(Wnk)*r3

	zk = VADD_(zk_p, zk_n);
	zk_d = IMULI_(inv, VSUB_(zk_p, zk_n));

	*r2 = VSUB_(uk, zk);
	*r0 = VADD_(uk, zk);
	*r3 = VADD_(uk2, zk_d);
	*r1 = VSUB_(uk2, zk_d);
}


static __inline void S_4(
	const __m128 *r0, const __m128 *r1, const __m128 *r2, const __m128 *r3, 
	float * __restrict o0, float * __restrict o1,
	float * __restrict o2, float * __restrict o3)
{
	VST(o0, *r0); VST(o1, *r1); VST(o2, *r2); VST(o3, *r3);
}

static __inline void S_8(
	const __m256 *r0, const __m256 *r1, const __m256 *r2, const __m256 *r3,
	float * __restrict o0, float * __restrict o1,
	float * __restrict o2, float * __restrict o3)
{
	VST_(o0, *r0); VST_(o1, *r1); VST_(o2, *r2); VST_(o3, *r3);
}


static __inline void L_4(
	__m128 *r0, __m128 *r1, __m128 *r2, __m128 *r3,
	const float *i0, const float *i1,
	const float *i2, const float *i3)
{
	*r0 = VLD(i0); *r1 = VLD(i1); *r2 = VLD(i2); *r3 = VLD(i3);
}

static __inline void L_8(
	__m256 *r0, __m256 *r1, __m256 *r2, __m256 *r3,
	const float *i0, const float *i1,
	const float *i2, const float *i3)
{
	*r0 = VLD_(i0); *r1 = VLD_(i1); *r2 = VLD_(i2); *r3 = VLD_(i3);
}














// junk yard

static __inline void fft_1(float * __restrict a, const float *m, const float *r)
{
	//*a++ = *m++;
	//*a++ = *m++;
	//*a++ = *r++;
	//*a++ = *r++;
	__m128 fm = _mm_load_ps(m);
	fm = _mm_loadh_pi(fm, (const __m64*)r);
	_mm_store_ps(a, fm);
}



//a[0] + a[1] * i
//b[0] + b[1] * i 
static __inline void fft_2(const float *a, const float *b, float * __restrict c)
{
	//c[0] = a[0] + b[0]; c[1] = a[1] + b[1];
	//c[2] = a[0] - b[0]; c[3] = a[1] - b[1];

	//*c++ = (*a++) + (*b++);
	//*c++ = (*a--) + (*b--);
	//*c++ = (*a++) - (*b++);
	//*c++ = (*a) - (*b);
	__m128 mma = _mm_setzero_ps();
	__m128 mmb = _mm_setzero_ps();
	mma = _mm_loadl_pi(mma, (const __m64*)a); //a[0] a[1] 0 0
	mmb = _mm_loadl_pi(mmb, (const __m64*)b); //b[0] b[1] 0 0

	__m128 mmc = _mm_add_ps(mma, mmb);        //a[0]+b[0]  a[1]+b[1]  0  0
	__m128 mmd = _mm_sub_ps(mma, mmb);        //a[0]-b[0]  a[1]-b[1]  0  0
	_mm_store_ps(c, _mm_shuffle_ps(mmc, mmd, _MM_SHUFFLE(1, 0, 1, 0)));
}


//Obs! Equal signs below are simultaneous update assignments!
//     Not individual update assignements!
//
//x0 = x0 + (x2 + x3) -> x0r = x0r + (x2r + x3r)
//                    -> x0i = x0i + (x2i + x3i)
//x2 = x0 - (x2 + x3) 
//                    -> x2r = x0r - (x2r + x3r)
//                    -> x2i = x0i - (x2i + x3i)
//x1 = x1 - (x2 - x3)I
//                    -> x1r = x1r + (x2i - x3i)
//                    -> x1i = x1i - (x2r - x3r)
//x3 = x1 + (x2 - x3)I
//                    -> x3r = x1r - (x2i - x3i)
//                    -> x3i = x1i + (x2r - x3r)
//
static __inline void fft_4(const float *in, float *out)
{
	//float t0[2], t1[2], t2[2], t3[2], t4[2], t5[2];

	//t0[0] = in[0]; t0[1] = in[1]; //x0r x0i
	//t2[0] = in[4]; t2[1] = in[5]; //x2r x2i
	//t1[0] = in[2]; t1[1] = in[3]; //x1r x1i
	//t3[0] = in[6]; t3[1] = in[7]; //x3r x3i

	//t4[0] = t2[0] + t3[0]; //(x2r + x3r)
	//t4[1] = t2[1] + t3[1]; //(x2i + x3i)

	//t5[0] = t2[1] - t3[1]; //(x2i - x3i) 
	//t5[1] = t2[0] - t3[0]; //(x2r - x3r)

	//out[0] = t0[0] + t4[0]; 
	//out[1] = t0[1] + t4[1];
	//out[2] = t1[0] + t5[0]; 
	//out[3] = t1[1] - t5[1];

	//out[4] = t0[0] - t4[0]; 
	//out[5] = t0[1] - t4[1];
	//out[6] = t1[0] - t5[0]; 
	//out[7] = t1[1] + t5[1];

	__m128 mm0 = _mm_setzero_ps();
	__m128 mm1 = _mm_setzero_ps();
	mm0 = _mm_loadl_pi(mm0, (const __m64*)&in[4]); //a[0] a[1] 0 0
	mm1 = _mm_loadl_pi(mm1, (const __m64*)&in[6]); //b[0] b[1] 0 0
	__m128 mm2 = _mm_add_ps(mm0, mm1);             //a[0]+b[0]  a[1]+b[1]  0  0
	__m128 mm3 = _mm_sub_ps(mm0, mm1);             //a[0]-b[0]  a[1]-b[1]  0  0

	mm0 = _mm_shuffle_ps(mm2, mm3, _MM_SHUFFLE(0, 1, 1, 0)); //t4[0] t4[1] t5[0] t5[1]
	mm1 = VXOR(mm0, VLIT4(-0.0f, 0.0f, 0.0f, 0.0f));
	mm0 = _mm_load_ps(in);

	_mm_store_ps(out, _mm_add_ps(mm0, mm1));
	_mm_store_ps(&out[4], _mm_sub_ps(mm0, mm1));
}

//x0 = x0 + (x2 + x3) -> x0r = x0r + (x2r + x3r)
//                    -> x0i = x0i + (x2i + x3i)
//x2 = x0 - (x2 + x3) 
//                    -> x2r = x0r - (x2r + x3r)
//                    -> x2i = x0i - (x2i + x3i)
//x1 = x1 + (x2 - x3)I
//                    -> x1r = x1r - (x2i - x3i)
//                    -> x1i = x1i + (x2r - x3r)
//x3 = x1 - (x2 - x3)I
//                    -> x3r = x1r + (x2i - x3i)
//                    -> x3i = x1i - (x2r - x3r)
//
static __inline void ifft_4(const float *in, float *out)
{
	//float t0[2], t1[2], t2[2], t3[2], t4[2], t5[2];

	//t0[0] = in[0]; t0[1] = in[1]; //x0r x0i
	//t2[0] = in[4]; t2[1] = in[5]; //x2r x2i
	//t1[0] = in[2]; t1[1] = in[3]; //x1r x1i
	//t3[0] = in[6]; t3[1] = in[7]; //x3r x3i

	//t4[0] = t2[0] + t3[0]; //(x2r + x3r)
	//t4[1] = t2[1] + t3[1]; //(x2i + x3i)

	//t5[0] = t2[1] - t3[1]; //(x2i - x3i) 
	//t5[1] = t2[0] - t3[0]; //(x2r - x3r)

	//out[0] = t0[0] + t4[0]; 
	//out[1] = t0[1] + t4[1];
	//out[2] = t1[0] - t5[0]; 
	//out[3] = t1[1] + t5[1];

	//out[4] = t0[0] - t4[0]; 
	//out[5] = t0[1] - t4[1];
	//out[6] = t1[0] + t5[0]; 
	//out[7] = t1[1] - t5[1];

	__m128 mm0 = _mm_setzero_ps();
	__m128 mm1 = _mm_setzero_ps();
	mm0 = _mm_loadl_pi(mm0, (const __m64*)&in[4]); //a[0] a[1] 0 0
	mm1 = _mm_loadl_pi(mm1, (const __m64*)&in[6]); //b[0] b[1] 0 0
	__m128 mm2 = _mm_add_ps(mm0, mm1);             //a[0]+b[0]  a[1]+b[1]  0  0
	__m128 mm3 = _mm_sub_ps(mm0, mm1);             //a[0]-b[0]  a[1]-b[1]  0  0

	__m128 mm4 = _mm_shuffle_ps(mm2, mm3, _MM_SHUFFLE(0, 1, 1, 0)); //t4[0] t4[1] t5[0] t5[1]
	__m128 mm5 = VXOR(mm4, VLIT4(0.0f, -0.0f, 0.0f, 0.0f));
	__m128 mm6 = _mm_load_ps(in);

	_mm_store_ps(out, _mm_add_ps(mm6, mm5));
	_mm_store_ps(&out[4], _mm_sub_ps(mm6, mm5));
}




static __inline void fft_8(float *pt0, float *pt1, float *pt2, float *pt3, const float *wre, const float *wim)
{
	__m128 R2 = VLD(pt2);
	__m128 R3 = VLD(pt3);
	__m128 re = VLD(wre);
	__m128 im = VLD(wim);

	__m128 zk_p = IMUL(R2, re, im);   //Wnk*r2
	__m128 zk_n = IMULJ(R3, re, im);  //conj(Wnk)*r3
	__m128 zk = VADD(zk_p, zk_n);
	__m128 zk_d = IMULI(0, VSUB(zk_p, zk_n));

	__m128 R0 = VLD(pt0);
	R2 = VSUB(R0, zk);
	R0 = VADD(R0, zk);

	__m128 R1 = VLD(pt1);
	R3 = VADD(R1, zk_d);
	R1 = VSUB(R1, zk_d);

	S_4(&R0, &R1, &R2, &R3, pt0, pt1, pt2, pt3);
}




//n_4 = n / 4
//pt0 = &output[re(input_offset)]
//pt1 = &output[re(input_offset + n / 4)]
//pt2 = &output[re(input_offset + n / 2)]
//pt3 = &output[re(input_offset + n / 2 + n / 4)]
//Wre = &Wre[k * stride * 2]
//Wim = &Wim[k * stride * 2]
static __inline void fft_n(const size_t stride, const size_t n_4, float * pt0, float * pt1, float * pt2, float * pt3, const float * Wre, const float * Wim)
{
	__m128 R0, R1, R2, R3;
	float wre[4];
	float wim[4];
	for (size_t k = 0; k < n_4; k += 2) {
		R0 = VLD(pt0 + k * 2); R1 = VLD(pt1 + k * 2); R2 = VLD(pt2 + k * 2); R3 = VLD(pt3 + k * 2);

		wre[0] = Wre[k*stride * 2];
		wre[1] = Wre[k*stride * 2 + 1];
		wre[2] = Wre[(k + 1)*stride * 2];
		wre[3] = Wre[(k + 1)*stride * 2 + 1];

		wim[0] = Wim[k*stride * 2];
		wim[1] = Wim[k*stride * 2 + 1];
		wim[2] = Wim[(k + 1)*stride * 2];
		wim[3] = Wim[(k + 1)*stride * 2 + 1];

		K_N(0, VLD(wre), VLD(wim), &R0, &R1, &R2, &R3);
		VST(pt0 + k * 2, R0); VST(pt1 + k * 2, R1); VST(pt2 + k * 2, R2); VST(pt3 + k * 2, R3);
	}
}


//AVX extension, process 8 floats(4 complex) in parallel
static __inline void fft_n_(const size_t stride, const size_t n_4, float * pt0, float * pt1, float * pt2, float * pt3, const float ** wre_tab, const float ** wim_tab)
{
	__m256 R0, R1, R2, R3;
	//size_t index = (int)log2((double)n_4) - 1;
	unsigned long index; _BitScanForward(&index, (unsigned long)n_4); index -= 1;
	const float *wre = wre_tab[index];
	const float *wim = wim_tab[index];

	ptrdiff_t k2;
	for (size_t k = 0; k < n_4; k += 4) {
		k2 = k << 1;
		R0 = VLD_(pt0 + k2); R1 = VLD_(pt1 + k2); R2 = VLD_(pt2 + k2); R3 = VLD_(pt3 + k2);
		K_N_(0, VLD_(wre), VLD_(wim), &R0, &R1, &R2, &R3);
		wre += 8; wim += 8;
		VST_(pt0 + k2, R0); VST_(pt1 + k2, R1); VST_(pt2 + k2, R2); VST_(pt3 + k2, R3);
	}
}


#endif
// vim: set autoindent noexpandtab tabstop=3 shiftwidth=3:
