/*!
 \file
 \brief fft building blocks
  
 Copyright (c) 2016, Lixun Xia <lixun.xia@outlook.com> 
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

!*/

#ifndef __SSE_FLOAT_H__
#define __SSE_FLOAT_H__

#include <immintrin.h>
#define AVX_ENABLE 1




#if(0)
typedef struct plan {
	
	int direct;
	size_t n;  //fft size

 	float * x;  //input buffer, complex number interleaved format
	float * y;  //output buffer, complex number interleaved format

	//n_4 counts from 8/4 to N/4, which is 2..N/4
	float ** wre_table;
	float ** wim_table;
} plan_t;
#endif



#define PI 3.1415926535897932384626433832795028841971693993751058209
#define re(x) (x*2)
#define im(x) (x*2+1)



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



///
/// INPUT:
///+------+------+------+------+
///|  im1 |  re1 |  im0 |  re0 |    x   (sign)i  
///+------+------+------+------+
///
/// OUTPUT:
///+------+------+------+------+
///| c1 * (sig)i | c0 * (sig)i |  
///+------+------+------+------+

__inline __m128 IMULI(int inv, __m128 a)
{
	if (inv) return VSWAPPAIRS(VXOR(a, VLIT4(0.0f, -0.0f, 0.0f, -0.0f)));
	else    return VSWAPPAIRS(VXOR(a, VLIT4(-0.0f, 0.0f, -0.0f, 0.0f)));
}

#ifdef AVX_ENABLE
__inline __m256 IMULI_(int inv, __m256 a)
{
	if (inv)
		return VSWAPPAIRS_(VXOR_(a, VLIT8(0.0f, -0.0f, 0.0f, -0.0f, 0.0f, -0.0f, 0.0f, -0.0f)));
	else
		return VSWAPPAIRS_(VXOR_(a, VLIT8(-0.0f, 0.0f, -0.0f, 0.0f, -0.0f, 0.0f, -0.0f, 0.0f)));
}
#endif


// FUNCTION: TWO COMPLEX FLOAT MULTICATIONS IN PARALLEL
//           IMAGINARY MUST BE SET AS VXOR(im, VLIT4(-0.0f, 0.0f, -0.0f, 0.0f))
//                                                                        ^
//                                                                       [0] index of XMM register
//
//           IF a,b,w,k are complex numbers, then this function calculates
//           a*w and b*k in one call:
//           [re(a) + im(a) * i] x [re(w) + im(w) * i] = [re(a)re(w) - im(a)im(w)] + [im(a)re(w) + re(a)im(w)]*i
//           [re(b) + im(b) * i] x [re(k) + im(k) * i] = [re(b)re(k) - im(b)im(k)] + [im(b)re(k) + re(b)im(k)]*i
// STEPS:
//           re(a)  im(a)  re(b)  im(b)    x    re(w)  re(w)  re(k)  re(k)    =    re(a)re(w)  im(a)re(w)  re(b)re(k)  im(b)re(k)
//           im(a)  re(a)  im(b)  re(b)    x    im(w) -im(w)  im(k) -im(k)    =    im(a)im(w) -re(a)im(w)  im(b)im(k) -re(b)im(k)
//
// NOTE: 
//           Imaginary parts of w and k starts with positive sign!
//
__inline __m128 IMUL(__m128 d, __m128 re, __m128 im) 
{
	re = VMUL(re, d);
	im = VMUL(im, VSWAPPAIRS(d));
	return VSUB(re, im);
}

#ifdef AVX_ENABLE
__inline __m256 IMUL_(__m256 d, __m256 re, __m256 im)
{
	re = VMUL_(re, d);
	im = VMUL_(im, VSWAPPAIRS_(d));
	return VSUB_(re, im);
}
#endif



/// FUNCTION: TWO COMPLEX FLOAT MULTIPLIED BY A COMMON (CONJUGATED) COMPLEX VALUE
///           IMAGINARY MUST BE SET AS VXOR(im, VLIT4(-0.0f, 0.0f, -0.0f, 0.0f))
///
/// INPUT:
///+------+------+------+------+
///|  im1 |  re1 |  im0 |  re0 |    x   (re - im*i)  
///+------+------+------+------+
///
/// OUTPUT:
///+------+------+------+------+
///|      c1     |      c0     |  
///+------+------+------+------+
__inline __m128 IMULJ(__m128 d, __m128 re, __m128 im) 
{
	re = VMUL(re, d);
	im = VMUL(im, VSWAPPAIRS(d));
	return VADD(re, im);
}

#ifdef AVX_ENABLE
__inline __m256 IMULJ_(__m256 d, __m256 re, __m256 im)
{
	re = VMUL_(re, d);
	im = VMUL_(im, VSWAPPAIRS_(d));
	return VADD_(re, im);
}
#endif




__inline void K_N(int inv, __m128 re, __m128 im, __m128 *r0, __m128 *r1, __m128 *r2, __m128 *r3)
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


#ifdef AVX_ENABLE
/// do 4-complex numbers in one iteration of complex conjugate pair algorithm
__inline void K_N_(int inv, __m256 re, __m256 im, __m256 *r0, __m256 *r1, __m256 *r2, __m256 *r3)
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
#endif


#if(0)
__inline void S_4(__m128 r0, __m128 r1, __m128 r2, __m128 r3, 
	float * restrict o0, float * restrict o1, 
	float * restrict o2, float * restrict o3)
{
	VST(o0, r0); VST(o1, r1); VST(o2, r2); VST(o3, r3);
}
#endif

__inline void S_4(
	const __m128 *r0, const __m128 *r1, const __m128 *r2, const __m128 *r3,
	float * __restrict o0, float * __restrict o1,
	float * __restrict o2, float * __restrict o3)
{
	VST(o0, *r0); VST(o1, *r1); VST(o2, *r2); VST(o3, *r3);
}

#ifdef AVX_ENABLE
__inline void S_8(
	const __m256 *r0, const __m256 *r1, const __m256 *r2, const __m256 *r3,
	float * __restrict o0, float * __restrict o1,
	float * __restrict o2, float * __restrict o3)
{
	VST_(o0, *r0); VST_(o1, *r1); VST_(o2, *r2); VST_(o3, *r3);
}
#endif

__inline void L_4(
	__m128 *r0, __m128 *r1, __m128 *r2, __m128 *r3,
	const float *i0, const float *i1,
	const float *i2, const float *i3)
{
	*r0 = VLD(i0); *r1 = VLD(i1); *r2 = VLD(i2); *r3 = VLD(i3);
}

#ifdef AVX_ENABLE
__inline void L_8(
	__m256 *r0, __m256 *r1, __m256 *r2, __m256 *r3,
	const float *i0, const float *i1,
	const float *i2, const float *i3)
{
	*r0 = VLD_(i0); *r1 = VLD_(i1); *r2 = VLD_(i2); *r3 = VLD_(i3);
}
#endif


__inline void MERGE(float * __restrict a, const float *b, const float *c)
{
	__m128 fm = _mm_load_ps(b);
	fm = _mm_loadh_pi(fm, (const __m64*)c);
	_mm_store_ps(a, fm);
}






///
/// Here I merge pair of size-4 base cases in parallel (leaf)
///
__inline void fft_4_2_leaf(
	const float * xk,
	const float * xk4n8,
	const float * xk2n8,
	const float * xk6n8,
	const float * xkn8,
	const float * xk5n8,
	const float * xk7n8,
	const float * xk3n8,
	float * __restrict out)
{
	__m128 mm0, mm1, mm2, mm3, mm4, mm5;

	//fft2 portion
	mm0 = _mm_loadu_ps(xk);                 // mm0 == x(k).re       x(k).im        ?               ?
	mm1 = _mm_loadu_ps(xk4n8);              // mm1 == x(k+4n/8).re  x(k+4n/8).im   ?               ?
	mm4 = _mm_loadu_ps(xkn8);                 
	mm5 = _mm_loadu_ps(xk5n8);              
	
	mm0 = _mm_loadh_pi(mm0, (const __m64*)xk2n8);  // mm0 == x(k).re       x(k).im        x(k+2n/8).re    x(k+2n/8).im
	mm1 = _mm_loadh_pi(mm1, (const __m64*)xk6n8);  // mm1 == x(k+4n/8).re  x(k+4n/8).im   x(k+6n/8).re    x(k+6n/8).im
	mm4 = _mm_loadh_pi(mm4, (const __m64*)xk7n8);
	mm5 = _mm_loadh_pi(mm5, (const __m64*)xk3n8);


	mm2 = VADD(mm0, mm1);
	mm3 = VSUB(mm0, mm1);
	mm0 = _mm_shuffle_ps(mm2, mm3, _MM_SHUFFLE(1, 0, 1, 0));

	//fft4 portion, multiply by (-i)
	mm1 = _mm_shuffle_ps(mm2, mm3, _MM_SHUFFLE(2, 3, 3, 2));
	mm1 = VXOR(mm1, VLIT4(-0.0f, 0.0f, 0.0f, 0.0f));


	//parallel portion
	mm2 = VADD(mm4, mm5);
	mm3 = VSUB(mm4, mm5);
	mm4 = _mm_shuffle_ps(mm2, mm3, _MM_SHUFFLE(1, 0, 1, 0));
	mm5 = _mm_shuffle_ps(mm2, mm3, _MM_SHUFFLE(2, 3, 3, 2));
	mm5 = VXOR(mm5, VLIT4(-0.0f, 0.0f, 0.0f, 0.0f));


	//save to linear memory
	_mm_store_ps(out, _mm_add_ps(mm0, mm1));        out += 4;
	_mm_store_ps(out, _mm_sub_ps(mm0, mm1));        out += 4;
	_mm_store_ps(out, _mm_add_ps(mm4, mm5));        out += 4;
	_mm_store_ps(out, _mm_sub_ps(mm4, mm5));
}


__inline void ifft_4_2_leaf(
	const float * xk,
	const float * xk4n8,
	const float * xk2n8,
	const float * xk6n8,
	const float * xkn8,
	const float * xk5n8,
	const float * xk7n8,
	const float * xk3n8,
	float * __restrict out)
{
	__m128 mm0, mm1, mm2, mm3, mm4, mm5;

	//fft2 portion
	mm0 = _mm_loadu_ps(xk);                 // mm0 == x(k).re       x(k).im        ?               ?
	mm1 = _mm_loadu_ps(xk4n8);              // mm1 == x(k+4n/8).re  x(k+4n/8).im   ?               ?
	mm4 = _mm_loadu_ps(xkn8);
	mm5 = _mm_loadu_ps(xk5n8);

	mm0 = _mm_loadh_pi(mm0, (const __m64*)xk2n8);  // mm0 == x(k).re       x(k).im        x(k+2n/8).re    x(k+2n/8).im
	mm1 = _mm_loadh_pi(mm1, (const __m64*)xk6n8);  // mm1 == x(k+4n/8).re  x(k+4n/8).im   x(k+6n/8).re    x(k+6n/8).im
	mm4 = _mm_loadh_pi(mm4, (const __m64*)xk7n8);
	mm5 = _mm_loadh_pi(mm5, (const __m64*)xk3n8);


	mm2 = VADD(mm0, mm1);
	mm3 = VSUB(mm0, mm1);
	mm0 = _mm_shuffle_ps(mm2, mm3, _MM_SHUFFLE(1, 0, 1, 0));

	//fft4 portion, multiply by (-i)
	mm1 = _mm_shuffle_ps(mm2, mm3, _MM_SHUFFLE(2, 3, 3, 2));
	mm1 = VXOR(mm1, VLIT4(0.0f, -0.0f, 0.0f, 0.0f));


	//parallel portion
	mm2 = VADD(mm4, mm5);
	mm3 = VSUB(mm4, mm5);
	mm4 = _mm_shuffle_ps(mm2, mm3, _MM_SHUFFLE(1, 0, 1, 0));
	mm5 = _mm_shuffle_ps(mm2, mm3, _MM_SHUFFLE(2, 3, 3, 2));
	mm5 = VXOR(mm5, VLIT4(0.0f, -0.0f, 0.0f, 0.0f));


	//save to linear memory
	_mm_store_ps(out, _mm_add_ps(mm0, mm1));        out += 4;
	_mm_store_ps(out, _mm_sub_ps(mm0, mm1));        out += 4;
	_mm_store_ps(out, _mm_add_ps(mm4, mm5));        out += 4;
	_mm_store_ps(out, _mm_sub_ps(mm4, mm5));
}



__inline void fft_8_1_leaf(
	const float * xk,
	const float * xk4n8,
	const float * xk2n8,
	const float * xk6n8,
	const float * xkn8,
	const float * xk5n8,
	const float * xk7n8,
	const float * xk3n8,
	float * __restrict out)
	//const float * wtan,
	//const float * wcos)
{
	__m128 mm0, mm1, mm2, mm3, mm4, mm5, mm6;

	//fft2 portion
	mm0 = _mm_loadu_ps(xk);                 // mm0 == x(k).re       x(k).im        ?               ?
	mm1 = _mm_loadu_ps(xk4n8);              // mm1 == x(k+4n/8).re  x(k+4n/8).im   ?               ?
	mm4 = _mm_loadu_ps(xkn8);
	mm5 = _mm_loadu_ps(xk5n8);

	mm0 = _mm_loadh_pi(mm0, (const __m64*)xk2n8);  // mm0 == x(k).re       x(k).im        x(k+2n/8).re    x(k+2n/8).im
	mm1 = _mm_loadh_pi(mm1, (const __m64*)xk6n8);  // mm1 == x(k+4n/8).re  x(k+4n/8).im   x(k+6n/8).re    x(k+6n/8).im
	mm4 = _mm_loadh_pi(mm4, (const __m64*)xk7n8);
	mm5 = _mm_loadh_pi(mm5, (const __m64*)xk3n8);


	mm2 = VADD(mm0, mm1);
	mm3 = VSUB(mm0, mm1);
	mm0 = _mm_shuffle_ps(mm2, mm3, _MM_SHUFFLE(1, 0, 1, 0));

	//fft4 portion, multiply by (-i)
	mm1 = _mm_shuffle_ps(mm2, mm3, _MM_SHUFFLE(2, 3, 3, 2));
	mm1 = VXOR(mm1, VLIT4(-0.0f, 0.0f, 0.0f, 0.0f));

	//_mm_add_ps(mm0, mm1); //interface to fft_8
	//_mm_sub_ps(mm0, mm1); //interface to fft_8


	mm2 = VADD(mm4, mm5);
	mm3 = VSUB(mm4, mm5);
	mm4 = _mm_shuffle_ps(mm2, mm3, _MM_SHUFFLE(1, 0, 1, 0));
	mm5 = _mm_shuffle_ps(mm2, mm3, _MM_SHUFFLE(3, 2, 3, 2));



	//__m128 R0 = VLD(pt0);  mm2
	//__m128 R1 = VLD(pt1);  mm3
	//__m128 R2 = VLD(pt2);  mm4
	//__m128 R3 = VLD(pt3);  mm5
	mm2 = _mm_add_ps(mm0, mm1); //interface to fft_8
	mm3 = _mm_sub_ps(mm0, mm1); //interface to fft_8


	mm0 = VLIT4(-1.0f, 1.0f, 0.0f, -0.0f);																	//VLD(wtan);  
	mm1 = VLIT4(0.70710678118654752440084436210485f, 0.70710678118654752440084436210485f, 1.0f, 1.0f);		//VLD(wcos);
	mm6 = VXOR(mm1, VLIT4(0.0f, -0.0f, 0.0f, -0.0f));


	mm4 = VADD(mm4, VMUL(mm0, VSWAPPAIRS(mm4))); //sigma
	mm5 = VSUB(mm5, VMUL(mm0, VSWAPPAIRS(mm5))); //delta

	mm1 = VMUL(mm1, VADD(mm4, mm5));
	mm6 = VMUL(mm6, VSWAPPAIRS(VSUB(mm4, mm5)));


	mm4 = VSUB(mm2, mm1);
	mm2 = VADD(mm2, mm1);
	mm5 = VADD(mm3, mm6);
	mm3 = VSUB(mm3, mm6);

	_mm_store_ps(out, mm2);        out += 4;
	_mm_store_ps(out, mm3);        out += 4;
	_mm_store_ps(out, mm4);        out += 4;
	_mm_store_ps(out, mm5);
}


__inline void ifft_8_1_leaf(
	const float * xk,
	const float * xk4n8,
	const float * xk2n8,
	const float * xk6n8,
	const float * xkn8,
	const float * xk5n8,
	const float * xk7n8,
	const float * xk3n8,
	float * __restrict out)
	//const float * wtan,
	//const float * wcos)
{
	__m128 mm0, mm1, mm2, mm3, mm4, mm5, mm6;

	//fft2 portion
	mm0 = _mm_loadu_ps(xk);                 // mm0 == x(k).re       x(k).im        ?               ?
	mm1 = _mm_loadu_ps(xk4n8);              // mm1 == x(k+4n/8).re  x(k+4n/8).im   ?               ?
	mm4 = _mm_loadu_ps(xkn8);
	mm5 = _mm_loadu_ps(xk5n8);

	mm0 = _mm_loadh_pi(mm0, (const __m64*)xk2n8);  // mm0 == x(k).re       x(k).im        x(k+2n/8).re    x(k+2n/8).im
	mm1 = _mm_loadh_pi(mm1, (const __m64*)xk6n8);  // mm1 == x(k+4n/8).re  x(k+4n/8).im   x(k+6n/8).re    x(k+6n/8).im
	mm4 = _mm_loadh_pi(mm4, (const __m64*)xk7n8);
	mm5 = _mm_loadh_pi(mm5, (const __m64*)xk3n8);


	mm2 = VADD(mm0, mm1);
	mm3 = VSUB(mm0, mm1);
	mm0 = _mm_shuffle_ps(mm2, mm3, _MM_SHUFFLE(1, 0, 1, 0));

	//fft4 portion, multiply by (-i)
	mm1 = _mm_shuffle_ps(mm2, mm3, _MM_SHUFFLE(2, 3, 3, 2));
	mm1 = VXOR(mm1, VLIT4(0.0f, -0.0f, 0.0f, 0.0f));

	//_mm_add_ps(mm0, mm1); //interface to fft_8
	//_mm_sub_ps(mm0, mm1); //interface to fft_8


	mm2 = VADD(mm4, mm5);
	mm3 = VSUB(mm4, mm5);
	mm4 = _mm_shuffle_ps(mm2, mm3, _MM_SHUFFLE(1, 0, 1, 0));
	mm5 = _mm_shuffle_ps(mm2, mm3, _MM_SHUFFLE(3, 2, 3, 2));



	//__m128 R0 = VLD(pt0);  mm2
	//__m128 R1 = VLD(pt1);  mm3
	//__m128 R2 = VLD(pt2);  mm4
	//__m128 R3 = VLD(pt3);  mm5
	mm2 = _mm_add_ps(mm0, mm1); //interface to fft_8
	mm3 = _mm_sub_ps(mm0, mm1); //interface to fft_8


	mm0 = VLIT4(1.0f, -1.0f, -0.0f, 0.0f);																	//VLD(wtan);  
	mm1 = VLIT4(0.70710678118654752440084436210485f, 0.70710678118654752440084436210485f, 1.0f, 1.0f);		//VLD(wcos);
	mm6 = VXOR(mm1, VLIT4(-0.0f, 0.0f, -0.0f, 0.0f));


	mm4 = VADD(mm4, VMUL(mm0, VSWAPPAIRS(mm4))); //sigma
	mm5 = VSUB(mm5, VMUL(mm0, VSWAPPAIRS(mm5))); //delta

	mm1 = VMUL(mm1, VADD(mm4, mm5));
	mm6 = VMUL(mm6, VSWAPPAIRS(VSUB(mm4, mm5)));


	mm4 = VSUB(mm2, mm1);
	mm2 = VADD(mm2, mm1);
	mm5 = VADD(mm3, mm6);
	mm3 = VSUB(mm3, mm6);

	_mm_store_ps(out, mm2);        out += 4;
	_mm_store_ps(out, mm3);        out += 4;
	_mm_store_ps(out, mm4);        out += 4;
	_mm_store_ps(out, mm5);
}








///
/// I moved building blocks of fft here
///

__inline void fft_1(float * a, const float *m, const float *r)
{
	__m128 fm = _mm_loadu_ps(m);
    fm = _mm_loadh_pi(fm, (const __m64*)r);
    _mm_store_ps(a, fm);
}



__inline void fft_2(const float *a, const float *b, float *c)
{
	__m128 mma = _mm_setzero_ps();
    __m128 mmb = _mm_setzero_ps();
    mma = _mm_loadl_pi(mma, (const __m64*)a); //a[0] a[1] 0 0
    mmb = _mm_loadl_pi(mmb, (const __m64*)b); //b[0] b[1] 0 0

    //__m128 mmc = _mm_add_ps(mma, mmb);        //a[0]+b[0]  a[1]+b[1]  0  0
    //__m128 mmd = _mm_sub_ps(mma, mmb);        //a[0]-b[0]  a[1]-b[1]  0  0
    _mm_store_ps(c, _mm_shuffle_ps(VADD(mma, mmb), VSUB(mma, mmb), _MM_SHUFFLE(1, 0, 1, 0)));
}




__inline void ifft_4(const float *in, float *out)
{
   //const float *din = (const float *)in;
   //float *dout = (float *)out;
   //float t0[2], t1[2], t2[2], t3[2], t4[2], t5[2];

   //t0[0] = in[0]; t0[1] = in[1]; //x0r x0i
   //t2[0] = in[4]; t2[1] = in[5]; //x2r x2i
   //t1[0] = in[2]; t1[1] = in[3]; //x1r x1i
   //t3[0] = in[6]; t3[1] = in[7]; //x3r x3i

   //t4[0] = t2[0] + t3[0]; //(x2r + x3r)
   //t4[1] = t2[1] + t3[1]; //(x2i + x3i)

   //t5[0] = t2[1] - t3[1]; //(x2i - x3i)
   //t5[1] = t2[0] - t3[0]; //(x2r - x3r)

   //out[0] = t0[0] + t4[0]; out[1] = t0[1] + t4[1];
   //out[4] = t0[0] - t4[0]; out[5] = t0[1] - t4[1];

   //out[2] = t1[0] - t5[0]; out[3] = t1[1] + t5[1];
   //out[6] = t1[0] + t5[0]; out[7] = t1[1] - t5[1];
	__m128 mm0 = _mm_setzero_ps();
	__m128 mm1 = _mm_setzero_ps();
	mm0 = _mm_loadl_pi(mm0, (const __m64*)&in[4]); //a[0] a[1] 0 0
	mm1 = _mm_loadl_pi(mm1, (const __m64*)&in[6]); //b[0] b[1] 0 0
	__m128 mm2 = _mm_add_ps(mm0, mm1);             //a[0]+b[0]  a[1]+b[1]  0  0
	__m128 mm3 = _mm_sub_ps(mm0, mm1);             //a[0]-b[0]  a[1]-b[1]  0  0\n
	
	mm0 = _mm_shuffle_ps(mm2, mm3, _MM_SHUFFLE(0, 1, 1, 0)); //t4[0] t4[1] t5[0] t5[1]
	mm1 = VXOR(mm0, VLIT4(0.0f, -0.0f, 0.0f, 0.0f));
	
	mm0 = _mm_load_ps(in);
	_mm_store_ps(out, _mm_add_ps(mm0, mm1));
	_mm_store_ps(&out[4], _mm_sub_ps(mm0, mm1));
}



__inline void fft_4(const float *in, float *out)
{
   //float t0[2], t1[2], t2[2], t3[2], t4[2], t5[2];\n
   //t0[0] = in[0]; t0[1] = in[1]; //x0r x0i
   //t2[0] = in[4]; t2[1] = in[5]; //x2r x2i
   //t1[0] = in[2]; t1[1] = in[3]; //x1r x1i
   //t3[0] = in[6]; t3[1] = in[7]; //x3r x3i\n
   //t4[0] = t2[0] + t3[0]; //(x2r + x3r)
   //t4[1] = t2[1] + t3[1]; //(x2i + x3i)\n
   //t5[0] = t2[1] - t3[1]; //(x2i - x3i)
   //t5[1] = t2[0] - t3[0]; //(x2r - x3r)\n
   //out[0] = t0[0] + t4[0]; out[1] = t0[1] + t4[1];
   //out[4] = t0[0] - t4[0]; out[5] = t0[1] - t4[1];\n
   //out[2] = t1[0] + t5[0]; out[3] = t1[1] - t5[1];
   //out[6] = t1[0] - t5[0]; out[7] = t1[1] + t5[1];
	__m128 mm0 = _mm_setzero_ps();
	__m128 mm1 = _mm_setzero_ps();
	mm0 = _mm_loadl_pi(mm0, (const __m64*)&in[4]); //a[0] a[1] 0 0
	mm1 = _mm_loadl_pi(mm1, (const __m64*)&in[6]); //b[0] b[1] 0 0
	__m128 mm2 = _mm_add_ps(mm0, mm1);             //a[0]+b[0]  a[1]+b[1]  0  0
	__m128 mm3 = _mm_sub_ps(mm0, mm1);             //a[0]-b[0]  a[1]-b[1]  0  0\n
	
	mm0 = _mm_shuffle_ps(mm2, mm3, _MM_SHUFFLE(0, 1, 1, 0)); //t4[0] t4[1] t5[0] t5[1]
	mm1 = VXOR(mm0, VLIT4(-0.0f, 0.0f, 0.0f, 0.0f));
	
	mm0 = _mm_load_ps(in);
	_mm_store_ps(out, _mm_add_ps(mm0, mm1));
	_mm_store_ps(&out[4], _mm_sub_ps(mm0, mm1));
}






// FFTS reference implementation, equivalent to current solution
#if(0)
__inline void ifft_8(float *pt0, float *pt1, float *pt2, float *pt3, const float *wre, const float *wim)
{
	__m128 R2 = VLD(pt2);
	__m128 R3 = VLD(pt3);
	__m128 re = VLD(wre);
	__m128 im = VLD(wim);

	__m128 zk_p = IMUL(R2, re, im);   //Wnk*r2
	__m128 zk_n = IMULJ(R3, re, im);  //conj(Wnk)*r3
	__m128 zk = VADD(zk_p, zk_n);
	__m128 zk_d = IMULI(1, VSUB(zk_p, zk_n));

	__m128 R0 = VLD(pt0);
	R2 = VSUB(R0, zk);
	R0 = VADD(R0, zk);

	__m128 R1 = VLD(pt1);
	R3 = VADD(R1, zk_d);
	R1 = VSUB(R1, zk_d);

	S_4(&R0, &R1, &R2, &R3, pt0, pt1, pt2, pt3);
}


// pt2 layout:    pt2[0] + pt2[1] * i    pt2[2] + pt2[3] * i
// pt3 layout:    pt3[0] + pt3[1] * i    pt3[2] + pt3[3] * i
//
// load to R2:    R2 = pt2[0], pt2[1], pt2[2], pt2[3]
// load to R3:    R3 = pt3[0], pt3[1], pt3[2], pt3[3]
// load to re:    re = cos(0), cos(0), cos(pi/4), cos(pi/4)
// load to im:    im = sin(0), -sin(0), sin(pi/4), -sin(pi/4)
// 
// zk_p is:       (pt2[0] + pt2[1] * i) x (cos(0) + sin(0) * i)   (pt2[2] + pt2[3] * i) x (cos(pi/4) + sin(pi/4) * i)  
// zk_n is:       (pt3[0] + pt3[1] * i) x (cos(0) - sin(0) * i)   (pt3[2] + pt3[3] * i) x (cos(pi/4) - sin(pi/4) * i) 
//
// zk is:         zk_p + zk_n
// zk_d is:       (zk_p - zk_n) * i
//
//                R0 = pt0[0], pt0[1], pt0[2], pt0[3]
//                R1 = pt1[0], pt1[1], pt1[2], pt1[3]
//
//                R2 = R0 - zk
//                R0 = R0 + zk
//                R3 = R1 + zk_d
//                R1 = R1 - zk_d
//
__inline void fft_8(float *pt0, float *pt1, float *pt2, float *pt3, const float *wre, const float *wim)
{
	__m128 R2 = VLD(pt2);
	__m128 R3 = VLD(pt3);
	__m128 re = VLD(wre);
	__m128 im = VLD(wim);

	__m128 R0 = IMUL(R2, re, im);   //zk_p := Wnk*r2, renaming to R0
	__m128 R1 = IMULJ(R3, re, im);  //zk_n := conj(Wnk)*r3, renaming to R1

	__m128 zk = VADD(R0, R1);
	__m128 zk_d = IMULI(0, VSUB(R0, R1));

	R0 = VLD(pt0);
	R1 = VLD(pt1);

	R2 = VSUB(R0, zk);
	R0 = VADD(R0, zk);

	R3 = VADD(R1, zk_d);
	R1 = VSUB(R1, zk_d);

	S_4(&R0, &R1, &R2, &R3, pt0, pt1, pt2, pt3);
}
#endif


#if(1)
// pt2 layout:    pt2[0] + pt2[1] * i    pt2[2] + pt2[3] * i
// pt3 layout:    pt3[0] + pt3[1] * i    pt3[2] + pt3[3] * i
//
// load to R2:    R2 = pt2[0], pt2[1], pt2[2], pt2[3]
// load to R3:    R3 = pt3[0], pt3[1], pt3[2], pt3[3]
// load to re:    re = cos(0), cos(0), cos(pi/4), cos(pi/4)
// load to im:    im = -tan(0), tan(0), -tan(-pi/4), tan(-pi/4)  :: divided by cos(phi), change sign order!!
// 
// zk_p is:       cos(0) x (pt2[0] + pt2[1] * i) x (1 + tan(0) * i)     cos(pi/4) x (pt2[2] + pt2[3] * i) x (1 + tan(pi/4) * i)  
// zk_n is:       cos(0) x (pt3[0] + pt3[1] * i) x (1 - tan(0) * i)     cos(pi/4) x (pt3[2] + pt3[3] * i) x (1 - tan(pi/4) * i) 
//
// zk_p is:       cos(0) x [(pt2[0] + pt2[1] * i) + (pt2[0] + pt2[1] * i) * tan(0) * i]    cos(pi/4) x [(pt2[2] + pt2[3] * i) + (pt2[2] + pt2[3] * i) * tan(pi/4) * i]
// zk_p is:       cos(0) x [(pt2[0] + pt2[1] * i) + (-pt2[1] + pt2[0] * i) * tan(0)]    cos(pi/4) x [(pt2[2] + pt2[3] * i) + (-pt2[3] + pt2[2] * i) * tan(pi/4)]
//                := cos() x sigma
//
// zk_n is:       cos(0) x [(pt3[0] + pt3[1] * i) - (pt3[0] + pt3[1] * i) * tan(0) * i]     cos(pi/4) x [(pt3[2] + pt3[3] * i) - (pt3[2] + pt3[3] * i) * tan(pi/4) * i]
// zk_n is:       cos(0) x [(pt3[0] + pt3[1] * i) - (-pt3[1] + pt3[0] * i) * tan(0)]     cos(pi/4) x [(pt3[2] + pt3[3] * i) - (-pt3[3] + pt3[2] * i) * tan(pi/4)]
//                := cos() x delta
//
//
// zk is:         zk_p + zk_n          =   (sigma + delta) x cos()
// zk_d is:       (zk_p - zk_n) * i    =   (sigma - delta) x cos() x i
//
//                R0 = pt0[0], pt0[1], pt0[2], pt0[3]
//                R1 = pt1[0], pt1[1], pt1[2], pt1[3]
//
//                R2 = R0 - zk
//                R0 = R0 + zk
//                R3 = R1 + zk_d
//                R1 = R1 - zk_d
__inline void fft_8(
	float *pt0, 
	float *pt1, 
	float *pt2, 
	float *pt3,
	const float * wtan,
	const float * wcos)
{
	__m128 R0 = VLD(pt0);
	__m128 R1 = VLD(pt1);
	__m128 R2 = VLD(pt2);
	__m128 R3 = VLD(pt3);

	__m128 tan_theta = VLD(wtan);   //VLIT4(-1.0f, 1.0f, 0.0f, -0.0f);  
	__m128 cos_theta = VLD(wcos);   //VLIT4(0.70710678118654752440084436210485f, 0.70710678118654752440084436210485f, 1.0f, 1.0f);  
	__m128 cos_theta_ = VXOR(cos_theta, VLIT4(0.0f,-0.0f,0.0f,-0.0f));


	R2 = VADD(R2, VMUL(tan_theta, VSWAPPAIRS(R2))); //sigma
	R3 = VSUB(R3, VMUL(tan_theta, VSWAPPAIRS(R3))); //delta

	cos_theta = VMUL(cos_theta, VADD(R2, R3));
	cos_theta_ = VMUL(cos_theta_, VSWAPPAIRS(VSUB(R2, R3)));


	R2 = VSUB(R0, cos_theta);
	R0 = VADD(R0, cos_theta);
	R3 = VADD(R1, cos_theta_);
	R1 = VSUB(R1, cos_theta_);

	S_4(&R0, &R1, &R2, &R3, pt0, pt1, pt2, pt3);
}


__inline void ifft_8(
	float *pt0,
	float *pt1,
	float *pt2,
	float *pt3,
	const float * wtan,
	const float * wcos)
{
	__m128 R0 = VLD(pt0);
	__m128 R1 = VLD(pt1);
	__m128 R2 = VLD(pt2);
	__m128 R3 = VLD(pt3);

	__m128 tan_theta = VLD(wtan);    
	__m128 cos_theta = VLD(wcos);   
	__m128 cos_theta_ = VXOR(cos_theta, VLIT4(-0.0f, 0.0f, -0.0f, 0.0f));


	R2 = VADD(R2, VMUL(tan_theta, VSWAPPAIRS(R2))); //sigma
	R3 = VSUB(R3, VMUL(tan_theta, VSWAPPAIRS(R3))); //delta

	cos_theta = VMUL(cos_theta, VADD(R2, R3));
	cos_theta_ = VMUL(cos_theta_, VSWAPPAIRS(VSUB(R2, R3)));


	R2 = VSUB(R0, cos_theta);
	R0 = VADD(R0, cos_theta);
	R3 = VADD(R1, cos_theta_);
	R1 = VSUB(R1, cos_theta_);

	S_4(&R0, &R1, &R2, &R3, pt0, pt1, pt2, pt3);
}
#endif



///
/// sub fft with sse support
///
#if(0)
__inline void ifft_n(const size_t n_4, float *pt0, float *pt1, float *pt2, float *pt3, const float *wre, const float *wim)
{
    __m128 R0, R1, R2, R3;
    //size_t index = (int)log2((double)n_4) - 1;
    //unsigned long index; _BitScanForward(&index, (unsigned long)n_4); index -= 1;
    //const float *wre = wre_tab[index];
    //const float *wim = wim_tab[index];

    //ptrdiff_t k2;
    for (size_t k = 0; k < n_4; k += 2) {
        //k2 = k << 1;
        //R0 = VLD(pt0 + k2); R1 = VLD(pt1 + k2); R2 = VLD(pt2 + k2); R3 = VLD(pt3 + k2);
        L_4(&R0, &R1, &R2, &R3, pt0, pt1, pt2, pt3);
        K_N(1, VLD(wre), VLD(wim), &R0, &R1, &R2, &R3);
        //VST(pt0 + k2, R0); VST(pt1 + k2, R1); VST(pt2 + k2, R2); VST(pt3 + k2, R3);
        S_4(&R0, &R1, &R2, &R3, pt0, pt1, pt2, pt3);
        wre += 4; wim += 4;
        pt0 += 4; pt1 += 4; pt2 += 4; pt3 += 4;
    }
}


__inline void fft_n(const size_t n_4, float *pt0, float *pt1, float *pt2, float *pt3, const float *wre, const float *wim)
{
    __m128 R0, R1, R2, R3;
    for (size_t k = 0; k < n_4; k += 2) {
        L_4(&R0, &R1, &R2, &R3, pt0, pt1, pt2, pt3);
        K_N(0, VLD(wre), VLD(wim), &R0, &R1, &R2, &R3);
        S_4(&R0, &R1, &R2, &R3, pt0, pt1, pt2, pt3);
        wre += 4; wim += 4;
        pt0 += 4; pt1 += 4; pt2 += 4; pt3 += 4;
    }
}
#endif









__inline void fft_n(
	const size_t n_4, 
	float *pt0, 
	float *pt1, 
	float *pt2, 
	float *pt3, 
	const float *wre, 
	const float *wim)
{
	for (size_t k = 0; k < n_4; k += 2) {

		__m128 R0 = VLD(pt0);
		__m128 R1 = VLD(pt1);
		__m128 R2 = VLD(pt2);
		__m128 R3 = VLD(pt3);

		__m128 tan_theta = VLD(wre);    // VLIT4(-1.0f, 1.0f, 0.0f, -0.0f);
		__m128 cos_theta = VLD(wim);    // VLIT4(0.70710678118654752440084436210485f, 0.70710678118654752440084436210485f, 1.0f, 1.0f);
		__m128 cos_theta_ = VXOR(cos_theta, VLIT4(0.0f, -0.0f, 0.0f, -0.0f));


		R2 = VADD(R2, VMUL(tan_theta, VSWAPPAIRS(R2))); //sigma
		R3 = VSUB(R3, VMUL(tan_theta, VSWAPPAIRS(R3))); //delta

		cos_theta = VMUL(cos_theta, VADD(R2, R3));
		cos_theta_ = VMUL(cos_theta_, VSWAPPAIRS(VSUB(R2, R3)));


		R2 = VSUB(R0, cos_theta);
		R0 = VADD(R0, cos_theta);
		R3 = VADD(R1, cos_theta_);
		R1 = VSUB(R1, cos_theta_);

		S_4(&R0, &R1, &R2, &R3, pt0, pt1, pt2, pt3);

		wre += 4; wim += 4;
		pt0 += 4; pt1 += 4; pt2 += 4; pt3 += 4;
	}
}


__inline void ifft_n(
	const size_t n_4,
	float *pt0,
	float *pt1,
	float *pt2,
	float *pt3,
	const float *wre,
	const float *wim)
{
	for (size_t k = 0; k < n_4; k += 2) {

		__m128 R0 = VLD(pt0);
		__m128 R1 = VLD(pt1);
		__m128 R2 = VLD(pt2);
		__m128 R3 = VLD(pt3);

		__m128 tan_theta = VLD(wre);    // VLIT4(-1.0f, 1.0f, 0.0f, -0.0f);
		__m128 cos_theta = VLD(wim);    // VLIT4(0.70710678118654752440084436210485f, 0.70710678118654752440084436210485f, 1.0f, 1.0f);
		__m128 cos_theta_ = VXOR(cos_theta, VLIT4(-0.0f, 0.0f, -0.0f, 0.0f));


		R2 = VADD(R2, VMUL(tan_theta, VSWAPPAIRS(R2))); //sigma
		R3 = VSUB(R3, VMUL(tan_theta, VSWAPPAIRS(R3))); //delta

		cos_theta = VMUL(cos_theta, VADD(R2, R3));
		cos_theta_ = VMUL(cos_theta_, VSWAPPAIRS(VSUB(R2, R3)));


		R2 = VSUB(R0, cos_theta);
		R0 = VADD(R0, cos_theta);
		R3 = VADD(R1, cos_theta_);
		R1 = VSUB(R1, cos_theta_);

		S_4(&R0, &R1, &R2, &R3, pt0, pt1, pt2, pt3);

		wre += 4; wim += 4;
		pt0 += 4; pt1 += 4; pt2 += 4; pt3 += 4;
	}
}





///
/// FFTS reference implementation
/// sub fft with avx support
///
#if(0)
__inline void ifft_n_(const size_t n_4, float *pt0, float *pt1, float *pt2, float *pt3, const float *wre, const float *wim)
{
	__m256 R0, R1, R2, R3;
	for (size_t k = 0; k < n_4; k += 4) {
        L_8(&R0, &R1, &R2, &R3, pt0, pt1, pt2, pt3);
		K_N_(1, VLD_(wre), VLD_(wim), &R0, &R1, &R2, &R3);
        S_8(&R0, &R1, &R2, &R3, pt0, pt1, pt2, pt3);
		wre += 8; wim += 8;
        pt0 += 8; pt1 += 8; pt2 += 8; pt3 += 8;
	}
}


__inline void fft_n_(const size_t n_4, float *pt0, float *pt1, float *pt2, float *pt3, const float *wre, const float *wim)
{
	__m256 R0, R1, R2, R3;
	for (size_t k = 0; k < n_4; k += 4) {

        L_8(&R0, &R1, &R2, &R3, pt0, pt1, pt2, pt3);
		K_N_(0, VLD_(wre), VLD_(wim), &R0, &R1, &R2, &R3);
		wre += 8; wim += 8;
        S_8(&R0, &R1, &R2, &R3, pt0, pt1, pt2, pt3);
        pt0 += 8; pt1 += 8; pt2 += 8; pt3 += 8;
	}
}
#endif





#ifdef AVX_ENABLE
__inline void fft_n_(
	const size_t n_4,
	float *pt0,
	float *pt1,
	float *pt2,
	float *pt3,
	const float *wre,
	const float *wim)
{
	for (size_t k = 0; k < n_4; k += 4) {

		__m256 R0 = VLD_(pt0);
		__m256 R1 = VLD_(pt1);
		__m256 R2 = VLD_(pt2);
		__m256 R3 = VLD_(pt3);

		__m256 tan_theta = VLD_(wre);    // VLIT4(-1.0f, 1.0f, 0.0f, -0.0f);
		__m256 cos_theta = VLD_(wim);    // VLIT4(0.70710678118654752440084436210485f, 0.70710678118654752440084436210485f, 1.0f, 1.0f);
		__m256 cos_theta_ = VXOR_(cos_theta, VLIT8(0.0f, -0.0f, 0.0f, -0.0f, 0.0f, -0.0f, 0.0f, -0.0f));


		R2 = VADD_(R2, VMUL_(tan_theta, VSWAPPAIRS_(R2))); //sigma
		R3 = VSUB_(R3, VMUL_(tan_theta, VSWAPPAIRS_(R3))); //delta

		cos_theta = VMUL_(cos_theta, VADD_(R2, R3));
		cos_theta_ = VMUL_(cos_theta_, VSWAPPAIRS_(VSUB_(R2, R3)));


		R2 = VSUB_(R0, cos_theta);
		R0 = VADD_(R0, cos_theta);
		R3 = VADD_(R1, cos_theta_);
		R1 = VSUB_(R1, cos_theta_);

		S_8(&R0, &R1, &R2, &R3, pt0, pt1, pt2, pt3);

		wre += 8; wim += 8;
		pt0 += 8; pt1 += 8; pt2 += 8; pt3 += 8;
	}
}


__inline void ifft_n_(
	const size_t n_4,
	float *pt0,
	float *pt1,
	float *pt2,
	float *pt3,
	const float *wre,
	const float *wim)
{
	for (size_t k = 0; k < n_4; k += 4) {

		__m256 R0 = VLD_(pt0);
		__m256 R1 = VLD_(pt1);
		__m256 R2 = VLD_(pt2);
		__m256 R3 = VLD_(pt3);

		__m256 tan_theta = VLD_(wre);    // VLIT4(-1.0f, 1.0f, 0.0f, -0.0f);
		__m256 cos_theta = VLD_(wim);    // VLIT4(0.70710678118654752440084436210485f, 0.70710678118654752440084436210485f, 1.0f, 1.0f);
		__m256 cos_theta_ = VXOR_(cos_theta, VLIT8(-0.0f, 0.0f, -0.0f, 0.0f, -0.0f, 0.0f, -0.0f, 0.0f));


		R2 = VADD_(R2, VMUL_(tan_theta, VSWAPPAIRS_(R2))); //sigma
		R3 = VSUB_(R3, VMUL_(tan_theta, VSWAPPAIRS_(R3))); //delta

		cos_theta = VMUL_(cos_theta, VADD_(R2, R3));
		cos_theta_ = VMUL_(cos_theta_, VSWAPPAIRS_(VSUB_(R2, R3)));


		R2 = VSUB_(R0, cos_theta);
		R0 = VADD_(R0, cos_theta);
		R3 = VADD_(R1, cos_theta_);
		R1 = VSUB_(R1, cos_theta_);

		S_8(&R0, &R1, &R2, &R3, pt0, pt1, pt2, pt3);

		wre += 8; wim += 8;
		pt0 += 8; pt1 += 8; pt2 += 8; pt3 += 8;
	}
}
#endif


#endif
// vim: set autoindent noexpandtab tabstop=3 shiftwidth=3:
