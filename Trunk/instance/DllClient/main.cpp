

#include <time.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <immintrin.h>
#include "../fftw3.h"


#include <Windows.h>
#include "../libfft.h"





#define re(x) (x*2)
#define im(x) (x*2+1)







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

	printf("Hello!\n"); fflush(stdout); fflush(stderr);

#if 1
	plan_t p;
	fft_init(&p, 512, -1);
#endif



	srand((unsigned int)time(0));
	for (size_t i = 0; i < p.n; i++) {
		p.x[re(i)] = (float)((double)rand() / (double)RAND_MAX);
		p.x[im(i)] = (float)((double)rand() / (double)RAND_MAX);
	}
	//FILE *fid = fopen("./input1024.dat", "w");
	//for (int i = 0; i < N; i++)
	//    fprintf(fid, "%3.16f, %3.16f\n", x[re(i)], x[im(i)]);
	//fclose(fid);


















	fftwf_complex * fftIn = (fftwf_complex*)fftwf_malloc(sizeof(fftwf_complex) * 512);
	fftwf_complex * fftOut = (fftwf_complex*)fftwf_malloc(sizeof(fftwf_complex) * 512);

	float * cast = (float *)fftIn;
	for (size_t i = 0; i < 512; ++i) {
		cast[re(i)] = p.x[re(i)];
		cast[im(i)] = p.x[im(i)];
	}

	fftwf_plan plan = fftwf_plan_dft_1d(512, fftIn, fftOut, FFTW_FORWARD, FFTW_ESTIMATE);


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
	printf("fftw : %ld s, %ld ns\n", temp.tv_sec, temp.tv_nsec); }
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
	printf("fft : %ld s, %ld ns\n", temp.tv_sec, temp.tv_nsec); }
#endif

	fft_clean(&p);


















	size_t maxreidx, maximidx;
	double err_max_re = 0.0f;
	double err_max_im = 0.0f;
	cast = (float *)fftOut;
	for (int i = 0; i < 512; ++i) {
		if (fabs(p.y[re(i)] - cast[re(i)]) / fabs(cast[re(i)]) > err_max_re) {
			err_max_re = fabs(p.y[re(i)] - cast[re(i)]) / fabs(cast[re(i)]);
			maxreidx = i;
		}
		if (fabs(p.y[im(i)] - cast[im(i)]) / fabs(cast[im(i)]) > err_max_im) {
			err_max_im = fabs(p.y[im(i)] - cast[im(i)]) / fabs(cast[im(i)]);
			maximidx = i;
		}
	}
	printf("max error (real, imag) : (%3.16f, %3.16f)\n", err_max_re, err_max_im);
	printf("real fft vs fftw: (%3.16f, %3.16f * I) (%3.16f, %3.16f * I)\n", p.y[re(maxreidx)], p.y[im(maxreidx)], cast[re(maxreidx)], cast[im(maxreidx)]);
	printf("imag fft vs fftw: (%3.16f, %3.16f * I) (%3.16f, %3.16f * I)\n", p.y[re(maximidx)], p.y[im(maximidx)], cast[re(maximidx)], cast[im(maximidx)]);



	return 0;
}
