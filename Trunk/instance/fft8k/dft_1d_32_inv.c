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





void fft_init(plan_t * p, size_t n, int direct)
{
	p->n = n;
    float * W_REAL = (float *) malloc(n / 4 * 2 * sizeof(float));
    float * W_IMAG = (float *) malloc(n / 4 * 2 * sizeof(float));


    //W_REAL[0] = 1.0f;  W_REAL[1] = 1.0f;
    //W_IMAG[0] = 0.0f;  W_IMAG[1] = -0.0f;

    for (size_t k = 0; k < n / 4; ++k) {
        W_REAL[2*k+0] = -(float)tan((double)(direct * 2) * PI * (double)k / (double)n);
        W_REAL[2*k+1] = -W_REAL[2*k+0];
        W_IMAG[2*k+0] = (float)cos((double)(direct * 2) * PI * (double)k / (double)n);
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
    }

    if (direct == -1) {
		// code below prepares constant for fft_8() and ifft_8() conversions. wre is actual tan() and wim is actual cos()
		p->wre_table[0][0] = -0.0f; p->wre_table[0][1] = 0.0f; p->wre_table[0][2] = 1.0f; p->wre_table[0][3] = -1.0f; 
		p->wim_table[0][0] = 1.0f; p->wim_table[0][1] = 1.0f; p->wim_table[0][2] = (float)(sqrt(2.0)*0.5); p->wim_table[0][3] = (float)(sqrt(2.0)*0.5); 
    } else {
        p->wre_table[0][0] = 0.0f; p->wre_table[0][1] = -0.0f; p->wre_table[0][2] = -1.0f; p->wre_table[0][3] = 1.0f;
        p->wim_table[0][0] = 1.0f; p->wim_table[0][1] = 1.0f; p->wim_table[0][2] = (float)(sqrt(2.0)*0.5); p->wim_table[0][3] = (float)(sqrt(2.0)*0.5);
    }

    free(W_REAL);
    free(W_IMAG);

#ifdef WIN32
	p->x = (float *)_mm_malloc(n * 2 * sizeof(float), 32);
	p->y = (float *)_mm_malloc(n * 2 * sizeof(float), 32);
#else
	p->x = (float *)memalign(32, n * 2 * sizeof(float));
	p->y = (float *)memalign(32, n * 2 * sizeof(float));
#endif

	_MM_SET_FLUSH_ZERO_MODE(_MM_FLUSH_ZERO_ON);
    _MM_SET_DENORMALS_ZERO_MODE(_MM_DENORMALS_ZERO_ON);

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
   free(p->wre_table);
   free(p->wim_table);

#ifdef WIN32
	_mm_free(p->x);
	_mm_free(p->y);
#else
	free(p->x);
	free(p->y);
#endif
}




void fft(plan_t * p)
{

ifft_8_1_leaf(p->x + re(0), p->x + re(16), p->x + re(8), p->x + re(24), p->x + re(4), p->x + re(20), p->x + re(28), p->x + re(12), p->y + re(0));
ifft_4_2_leaf(p->x + re(2), p->x + re(18), p->x + re(10), p->x + re(26), p->x + re(30), p->x + re(14), p->x + re(6), p->x + re(22), p->y + re(8));
ifft_8_1_leaf(p->x + re(1), p->x + re(17), p->x + re(9), p->x + re(25), p->x + re(5), p->x + re(21), p->x + re(29), p->x + re(13), p->y + re(16));
ifft_8_1_leaf(p->x + re(31), p->x + re(15), p->x + re(7), p->x + re(23), p->x + re(3), p->x + re(19), p->x + re(27), p->x + re(11), p->y + re(24));
ifft_n_(4, p->y + re(0), p->y + re(4), p->y + re(8), p->y + re(12), p->wre_table[1], p->wim_table[1]);
ifft_n_(8, p->y + re(0), p->y + re(8), p->y + re(16), p->y + re(24), p->wre_table[2], p->wim_table[2]);
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
	plan_t p;
	fft_init(&p, 32, 1);

    srand((unsigned int)time(0));
    for (size_t i = 0; i < p.n; i++) {
        p.x[re(i)] = (float)((double)rand() / (double)RAND_MAX);
        p.x[im(i)] = (float)((double)rand() / (double)RAND_MAX);
    }
    //FILE *fid = fopen("./input1024.dat", "w");
    //for (int i = 0; i < N; i++)
    //    fprintf(fid, "%3.16f, %3.16f\n", x[re(i)], x[im(i)]);
    //fclose(fid);


















	fftwf_complex * fftIn = (fftwf_complex*)fftwf_malloc(sizeof(fftwf_complex) * 32);
    fftwf_complex * fftOut = (fftwf_complex*)fftwf_malloc(sizeof(fftwf_complex) * 32);

    float * cast = (float *)fftIn;
	for (size_t i = 0; i < 32; ++i) {
	    cast[re(i)] = p.x[re(i)];
	    cast[im(i)] = p.x[im(i)];
    }

    fftwf_plan plan = fftwf_plan_dft_1d(32, fftIn, fftOut, FFTW_BACKWARD, FFTW_ESTIMATE);


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
        fft(&p);


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

    fft_clean(&p);



















    size_t maxreidx, maximidx;
    double err_max_re = 0.0f;
    double err_max_im = 0.0f;
	cast = (float *)fftOut;
	for (int i = 0; i < 32; ++i) {
	    if ( fabs(p.y[re(i)] - cast[re(i)]) / fabs(cast[re(i)]) > err_max_re ) {
            err_max_re = fabs(p.y[re(i)] - cast[re(i)]) / fabs(cast[re(i)]);
            maxreidx = i; }
        if ( fabs(p.y[im(i)] - cast[im(i)]) / fabs(cast[im(i)]) > err_max_im ) {
            err_max_im = fabs(p.y[im(i)] - cast[im(i)]) / fabs(cast[im(i)]);
            maximidx = i; }
    }
    printf("max error (real, imag) : (%3.16f, %3.16f)\n", err_max_re, err_max_im);
    printf("real fft vs fftw: (%3.16f, %3.16f * I) (%3.16f, %3.16f * I)\n", p.y[re(maxreidx)], p.y[im(maxreidx)], cast[re(maxreidx)], cast[im(maxreidx)]);
    printf("imag fft vs fftw: (%3.16f, %3.16f * I) (%3.16f, %3.16f * I)\n", p.y[re(maximidx)], p.y[im(maximidx)], cast[re(maximidx)], cast[im(maximidx)]);
    return 0;
}




