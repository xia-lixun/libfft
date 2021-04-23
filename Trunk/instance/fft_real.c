#include <time.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <immintrin.h>
#include "fftw3.h"
#include "macros-sse.h"

#ifdef WIN32
#include <Windows.h>
#else
#include <malloc.h>
#endif


#include "libfft.h"


plan_t * fft_init(size_t n, int direct)
{
    plan_t* p = (plan_t*)malloc(sizeof(plan_t));

	p->n = n;
    float * W_REAL = (float *) malloc(n / 4 * 2 * sizeof(float));
    float * W_IMAG = (float *) malloc(n / 4 * 2 * sizeof(float));


    //W_REAL[0] = 1.0f;  W_REAL[1] = 1.0f;
    //W_IMAG[0] = 0.0f;  W_IMAG[1] = -0.0f;

    for (size_t k = 0; k < n / 4; ++k) {
        W_REAL[2*k+0] = -(float)tan((double)(direct) * PI2 * (double)k / (double)n);
        W_REAL[2*k+1] = -W_REAL[2*k+0];
        W_IMAG[2*k+0] = (float)cos((double)(direct) * PI2 * (double)k / (double)n);
        W_IMAG[2*k+1] = W_IMAG[2*k+0];
    }

    p->wre_table = (float **)malloc((int)log2(n / 4) * sizeof(float *));
    p->wim_table = (float **)malloc((int)log2(n / 4) * sizeof(float *));

   for (size_t n_4 = 2; n_4 < n / 2; n_4 = n_4 * 2) {

       //stride is constant for fixed n_4 value
       size_t stride = (n / 4) / n_4;
       size_t index = (int)log2((double)n_4) - 1;

#ifdef WIN32
       p->wre_table[index] = (float *)_mm_malloc(n_4 / 2 * 4 * sizeof(float), 32);
       p->wim_table[index] = (float *)_mm_malloc(n_4 / 2 * 4 * sizeof(float), 32);
#else
		p->wre_table[index] = (float *)memalign(32, n_4 / 2 * 4 * sizeof(float));
		p->wim_table[index] = (float *)memalign(32, n_4 / 2 * 4 * sizeof(float));
#endif

       float * pre = p->wre_table[index];
       float * pim = p->wim_table[index];

       size_t i = 0;
       size_t j = 0;
       for (size_t k = 0; k < n_4; k += 2) {

	        pre[i++] = W_REAL[k*stride * 2];
	        pre[i++] = W_REAL[k*stride * 2 + 1];
	        pre[i++] = W_REAL[(k + 1)*stride * 2];
	        pre[i++] = W_REAL[(k + 1)*stride * 2 + 1];

	        pim[j++] = W_IMAG[k*stride * 2];
	        pim[j++] = W_IMAG[k*stride * 2 + 1];
	        pim[j++] = W_IMAG[(k + 1)*stride * 2];
	        pim[j++] = W_IMAG[(k + 1)*stride * 2 + 1];
        }

       printf("------------%d------------------\n", n_4);
       for (size_t k = 0; k < n_4 / 2 * 4; ++k) {
           printf("(%2.16f) + (%2.16f) I\n", pre[k], pim[k]);
       }
    }

   p->wre_ropt = (float*)malloc((n + 1) * sizeof(float));
   p->wim_ropt = (float*)malloc((n + 1) * sizeof(float));

    if (direct == -1) {
		// code below prepares constant for fft_8() and ifft_8() conversions. wre is actual tan() and wim is actual cos()
		p->wre_table[0][0] = -0.0f; p->wre_table[0][1] = 0.0f; p->wre_table[0][2] = 1.0f; p->wre_table[0][3] = -1.0f; 
		p->wim_table[0][0] = 1.0f; p->wim_table[0][1] = 1.0f; p->wim_table[0][2] = (float)(sqrt(2.0)*0.5); p->wim_table[0][3] = (float)(sqrt(2.0)*0.5);

        // coefficients for real valued merge via butter-fly
        for (size_t k = 0; k < n + 1; ++k) {
            p->wre_ropt[k] = (float)cos((double)(direct) * PI * (double)k / (double)(n));
            p->wim_ropt[k] = (float)sin((double)(direct) * PI * (double)k / (double)(n));
        }
        
    } else {
        p->wre_table[0][0] = 0.0f; p->wre_table[0][1] = -0.0f; p->wre_table[0][2] = -1.0f; p->wre_table[0][3] = 1.0f;
        p->wim_table[0][0] = 1.0f; p->wim_table[0][1] = 1.0f; p->wim_table[0][2] = (float)(sqrt(2.0)*0.5); p->wim_table[0][3] = (float)(sqrt(2.0)*0.5);
    }

    free(W_REAL);
    free(W_IMAG);

#ifdef WIN32
	p->x = (float *)_mm_malloc(n * 2 * sizeof(float), 32);
	p->y = (float *)_mm_malloc(n * 2 * sizeof(float), 32);
    p->ze = (float*)_mm_malloc((n+1) * 2 * sizeof(float), 32);
    p->zo = (float*)_mm_malloc((n+1) * 2 * sizeof(float), 32);
#else
	p->x = (float *)memalign(32, n * 2 * sizeof(float));
	p->y = (float *)memalign(32, n * 2 * sizeof(float));
#endif

	_MM_SET_FLUSH_ZERO_MODE(_MM_FLUSH_ZERO_ON);
    _MM_SET_DENORMALS_ZERO_MODE(_MM_DENORMALS_ZERO_ON);
    
    return p;
}





void fft_clean(plan_t * p)
{
   for (size_t n_4 = 2; n_4 < p->n / 2; n_4 = n_4 * 2) {
       size_t index = (int)log2((double)n_4) - 1;
#ifdef WIN32
       _mm_free(p->wre_table[index]);
       _mm_free(p->wim_table[index]);
#else
       free(p->wre_table[index]);
       free(p->wim_table[index]);
#endif
   }
   free(p->wim_ropt);
   free(p->wre_ropt);
   free(p->wim_table);
   free(p->wre_table);

#ifdef WIN32
   _mm_free(p->zo);
   _mm_free(p->ze);
   _mm_free(p->y);
   _mm_free(p->x);
#else
	free(p->x);
	free(p->y);
#endif
    free(p);
}




// This routine calculates 128-point complex DFT
void fft(plan_t * p)
{

fft_8_1_leaf(p->x + re(0), p->x + re(64), p->x + re(32), p->x + re(96), p->x + re(16), p->x + re(80), p->x + re(112), p->x + re(48), p->y + re(0));
fft_4_2_leaf(p->x + re(8), p->x + re(72), p->x + re(40), p->x + re(104), p->x + re(120), p->x + re(56), p->x + re(24), p->x + re(88), p->y + re(8));
fft_8_1_leaf(p->x + re(4), p->x + re(68), p->x + re(36), p->x + re(100), p->x + re(20), p->x + re(84), p->x + re(116), p->x + re(52), p->y + re(16));
fft_8_1_leaf(p->x + re(124), p->x + re(60), p->x + re(28), p->x + re(92), p->x + re(12), p->x + re(76), p->x + re(108), p->x + re(44), p->y + re(24));
fft_8_1_leaf(p->x + re(2), p->x + re(66), p->x + re(34), p->x + re(98), p->x + re(18), p->x + re(82), p->x + re(114), p->x + re(50), p->y + re(32));
fft_4_2_leaf(p->x + re(10), p->x + re(74), p->x + re(42), p->x + re(106), p->x + re(122), p->x + re(58), p->x + re(26), p->x + re(90), p->y + re(40));
fft_8_1_leaf(p->x + re(126), p->x + re(62), p->x + re(30), p->x + re(94), p->x + re(14), p->x + re(78), p->x + re(110), p->x + re(46), p->y + re(48));
fft_4_2_leaf(p->x + re(6), p->x + re(70), p->x + re(38), p->x + re(102), p->x + re(118), p->x + re(54), p->x + re(22), p->x + re(86), p->y + re(56));
fft_8_1_leaf(p->x + re(1), p->x + re(65), p->x + re(33), p->x + re(97), p->x + re(17), p->x + re(81), p->x + re(113), p->x + re(49), p->y + re(64));
fft_4_2_leaf(p->x + re(9), p->x + re(73), p->x + re(41), p->x + re(105), p->x + re(121), p->x + re(57), p->x + re(25), p->x + re(89), p->y + re(72));
fft_8_1_leaf(p->x + re(5), p->x + re(69), p->x + re(37), p->x + re(101), p->x + re(21), p->x + re(85), p->x + re(117), p->x + re(53), p->y + re(80));
fft_8_1_leaf(p->x + re(125), p->x + re(61), p->x + re(29), p->x + re(93), p->x + re(13), p->x + re(77), p->x + re(109), p->x + re(45), p->y + re(88));
fft_8_1_leaf(p->x + re(127), p->x + re(63), p->x + re(31), p->x + re(95), p->x + re(15), p->x + re(79), p->x + re(111), p->x + re(47), p->y + re(96));
fft_4_2_leaf(p->x + re(7), p->x + re(71), p->x + re(39), p->x + re(103), p->x + re(119), p->x + re(55), p->x + re(23), p->x + re(87), p->y + re(104));
fft_8_1_leaf(p->x + re(3), p->x + re(67), p->x + re(35), p->x + re(99), p->x + re(19), p->x + re(83), p->x + re(115), p->x + re(51), p->y + re(112));
fft_8_1_leaf(p->x + re(123), p->x + re(59), p->x + re(27), p->x + re(91), p->x + re(11), p->x + re(75), p->x + re(107), p->x + re(43), p->y + re(120));
fft_n_(4, p->y + re(0), p->y + re(4), p->y + re(8), p->y + re(12), p->wre_table[1], p->wim_table[1]);
fft_n_(8, p->y + re(0), p->y + re(8), p->y + re(16), p->y + re(24), p->wre_table[2], p->wim_table[2]);
fft_n_(4, p->y + re(32), p->y + re(36), p->y + re(40), p->y + re(44), p->wre_table[1], p->wim_table[1]);
fft_n_(4, p->y + re(48), p->y + re(52), p->y + re(56), p->y + re(60), p->wre_table[1], p->wim_table[1]);
fft_n_(16, p->y + re(0), p->y + re(16), p->y + re(32), p->y + re(48), p->wre_table[3], p->wim_table[3]);
fft_n_(4, p->y + re(64), p->y + re(68), p->y + re(72), p->y + re(76), p->wre_table[1], p->wim_table[1]);
fft_n_(8, p->y + re(64), p->y + re(72), p->y + re(80), p->y + re(88), p->wre_table[2], p->wim_table[2]);
fft_n_(4, p->y + re(96), p->y + re(100), p->y + re(104), p->y + re(108), p->wre_table[1], p->wim_table[1]);
fft_n_(8, p->y + re(96), p->y + re(104), p->y + re(112), p->y + re(120), p->wre_table[2], p->wim_table[2]);
fft_n_(32, p->y + re(0), p->y + re(32), p->y + re(64), p->y + re(96), p->wre_table[4], p->wim_table[4]);
}



// This routine does real-value optimization to convert to results of 256-point real-valued DFT
// We assume the 128-point DFT results have already been calculated and stored in p->y.
void ropt(plan_t* p, float * output)
{
    size_t n = p->n;
    size_t i_re = re(n);
    size_t i_im = im(n);
    size_t ni_re;
    size_t ni_im;
    size_t i;

    // 1) compute DFT results of the even part
    // 2) compute DFT results of the odd part
    p->ze[re(0)] = p->y[re(0)];
    p->ze[im(0)] = 0.0;
    p->zo[re(0)] = p->y[im(0)];
    p->zo[im(0)] = 0.0;

    p->ze[i_re] = p->ze[re(0)];
    p->ze[i_im] = 0.0;
    p->zo[i_re] = p->zo[re(0)];
    p->zo[i_im] = 0.0;

    for (i = 1; i < n; ++i) {
        i_re = re(i);
        i_im = im(i);
        ni_re = re(n - i);
        ni_im = im(n - i);

        p->ze[i_re] = 0.5f * (p->y[i_re] + p->y[ni_re]);
        p->ze[i_im] = 0.5f * (p->y[i_im] - p->y[ni_im]);
        p->zo[i_re] = 0.5f * (p->y[i_im] + p->y[ni_im]);
        p->zo[i_im] = -0.5f * (p->y[i_re] - p->y[ni_re]);
    }

    // 3) do butter-fly merge of the two
    for (i = 0; i < n + 1; ++i) {
        i_re = re(i);
        i_im = im(i);
        output[i_re] = p->ze[i_re] + p->wre_ropt[i] * p->zo[i_re] - p->wim_ropt[i] * p->zo[i_im];
        output[i_im] = p->ze[i_im] + p->wre_ropt[i] * p->zo[i_im] + p->wim_ropt[i] * p->zo[i_re];
    }

    // 4) the result can be found at p->ze with real-imag interleaved (packed) format
}






#ifdef WIN32
LARGE_INTEGER StartingTime, EndingTime, ElapsedMicroseconds;
LARGE_INTEGER Frequency;

//__declspec(align(32)) float x[N*2];
//__declspec(align(32)) float y[N*2];
#else
struct timespec t0, t1;

//float __attribute__((aligned(32))) x[N * 2];
//float __attribute__((aligned(32))) y[N * 2];
#endif





int main(int argc, char * argv[])
{

    float xk[2] = { 0.1, 0.2 };
    float xk4n8[2] = { 0.3, 0.4 };
    float xk2n8[2] = { 0.5, 0.6 };
    float xk6n8[2] = { 0.7, 0.8 };
    float xkn8[2] = { 0.9, 1.0 };
    float xk5n8[2] = { 1.1, 1.2 };
    float xk7n8[2] = { 1.3, 1.4 };
    float xk3n8[2] = { 1.5, 1.6 };
    float y[16];
    fft_4_2_leaf(&xk, &xk4n8, &xk2n8, &xk6n8, &xkn8, &xk5n8, &xk7n8, &xk3n8, &y);
    for (int i = 0; i < 16; ++i)
        printf("%3.12f\n", y[i]);
    return 0;




	plan_t * p = fft_init(128, -1);

    srand((unsigned int)time(0));
    for (size_t i = 0; i < p->n; i++) {
        p->x[re(i)] = (float)((double)rand() / (double)RAND_MAX);
        p->x[im(i)] = (float)((double)rand() / (double)RAND_MAX);
    }
    //FILE *fid = fopen("./input1024.dat", "w");
    //for (int i = 0; i < N; i++)
    //    fprintf(fid, "%3.16f, %3.16f\n", x[re(i)], x[im(i)]);
    //fclose(fid);


















	fftwf_complex * fftIn = (fftwf_complex*)fftwf_malloc(sizeof(fftwf_complex) * 128);
    fftwf_complex * fftOut = (fftwf_complex*)fftwf_malloc(sizeof(fftwf_complex) * 128);

    float * cast = (float *)fftIn;
	for (size_t i = 0; i < 128; ++i) {
	    cast[re(i)] = p->x[re(i)];
	    cast[im(i)] = p->x[im(i)];
    }

    fftwf_plan plan = fftwf_plan_dft_1d(128, fftIn, fftOut, FFTW_FORWARD, FFTW_ESTIMATE);


#ifdef WIN32
	QueryPerformanceFrequency(&Frequency);
    QueryPerformanceCounter(&StartingTime);

#else
	clock_gettime(CLOCK_MONOTONIC, &t0);
#endif

    for (int i = 0; i < 10000; ++i)
        fftwf_execute(plan);

#ifdef WIN32
	QueryPerformanceCounter(&EndingTime);
    ElapsedMicroseconds.QuadPart = EndingTime.QuadPart - StartingTime.QuadPart;
    ElapsedMicroseconds.QuadPart *= 1000000;
    ElapsedMicroseconds.QuadPart /= Frequency.QuadPart;
    printf("fftw : %lld usec \n", ElapsedMicroseconds.QuadPart);


#else
	clock_gettime(CLOCK_MONOTONIC, &t1);
	{struct timespec temp;
	if ((t1.tv_nsec - t0.tv_nsec)<0) {
		temp.tv_sec = t1.tv_sec - t0.tv_sec - 1;
		temp.tv_nsec = 1000000000 + t1.tv_nsec - t0.tv_nsec;
	}
	else {
		temp.tv_sec = t1.tv_sec - t0.tv_sec;
		temp.tv_nsec = t1.tv_nsec - t0.tv_nsec;
	}
	printf("fftw : %ld s, %ld ns\n", temp.tv_sec, temp.tv_nsec);}
#endif
    fftwf_destroy_plan(plan);



















#ifdef WIN32
	QueryPerformanceFrequency(&Frequency);
    QueryPerformanceCounter(&StartingTime);

#else
	clock_gettime(CLOCK_MONOTONIC, &t0);
#endif
    for (int i = 0; i < 10000; ++i)
        fft(p);


#ifdef WIN32
	QueryPerformanceCounter(&EndingTime);
    ElapsedMicroseconds.QuadPart = EndingTime.QuadPart - StartingTime.QuadPart;
    ElapsedMicroseconds.QuadPart *= 1000000;
    ElapsedMicroseconds.QuadPart /= Frequency.QuadPart;
    printf("fft : %lld usec \n", ElapsedMicroseconds.QuadPart);


#else
	clock_gettime(CLOCK_MONOTONIC, &t1);
	{struct timespec temp;
	if ((t1.tv_nsec - t0.tv_nsec)<0) {
		temp.tv_sec = t1.tv_sec - t0.tv_sec - 1;
		temp.tv_nsec = 1000000000 + t1.tv_nsec - t0.tv_nsec;
	}
	else {
		temp.tv_sec = t1.tv_sec - t0.tv_sec;
		temp.tv_nsec = t1.tv_nsec - t0.tv_nsec;
	}
	printf("fft : %ld s, %ld ns\n", temp.tv_sec, temp.tv_nsec);}
#endif

    fft_clean(p);



















    size_t maxreidx, maximidx;
    float err_max_re = 0.0f;
    float err_max_im = 0.0f;
	cast = (float *)fftOut;
	for (int i = 0; i < 128; ++i) {
	    if ( fabsf(p->y[re(i)] - cast[re(i)]) / fabsf(cast[re(i)]) > err_max_re ) {
            err_max_re = fabsf(p->y[re(i)] - cast[re(i)]) / fabsf(cast[re(i)]);
            maxreidx = i; }
        if ( fabsf(p->y[im(i)] - cast[im(i)]) / fabsf(cast[im(i)]) > err_max_im ) {
            err_max_im = fabsf(p->y[im(i)] - cast[im(i)]) / fabsf(cast[im(i)]);
            maximidx = i; }
    }
    printf("max error (real, imag) : (%3.16f, %3.16f)\n", err_max_re, err_max_im);
    printf("real fft vs fftw: (%3.16f, %3.16f * I) (%3.16f, %3.16f * I)\n", p->y[re(maxreidx)], p->y[im(maxreidx)], cast[re(maxreidx)], cast[im(maxreidx)]);
    printf("imag fft vs fftw: (%3.16f, %3.16f * I) (%3.16f, %3.16f * I)\n", p->y[re(maximidx)], p->y[im(maximidx)], cast[re(maximidx)], cast[im(maximidx)]);







    //[2020-08-05] Add real value optimization
    plan_t * r = fft_init(128, -1);

    for (size_t i = 0; i < 2 * r->n; i++) {
        r->x[i] = (float)((double)i);
    }

    float * result = (float *) malloc(sizeof(float) * (256 / 2 + 1) * 2);
    
    fft(r);
    ropt(r, result);
    
    //for (size_t i = 0; i < (r->n + 1) * 2; i+=2) {
        //printf("(%3.16f, %3.16f)\n", result[i], result[i+1]);
    //}

    fft_clean(r);
    free(result);
    return 0;
}




