#include <time.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <immintrin.h>
#include "../instance/fftw3.h"
#include <Windows.h>
#include <malloc.h>
#include "deliverable/fft.h"
#include "deliverable/macros-sse.h"




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
    printf("porcess starts...\n");

    for (int nfft = 16; nfft < 16384; nfft = nfft * 2) {


        float* fft_in = (float *)_mm_malloc(nfft * 2 * sizeof(float), 32);
        float* fft_out = (float *)_mm_malloc(nfft * 2 * sizeof(float), 32);

        plan_t p;
        fft_init(&p, nfft, 1);

        //prepare test input vector
        srand((unsigned int)time(0));
        for (size_t i = 0; i < nfft; i++) {
            fft_in[re(i)] = (float)((double)rand() / (double)RAND_MAX);
            fft_in[im(i)] = (float)((double)rand() / (double)RAND_MAX);
        }

        fftwf_complex* fftIn = (fftwf_complex*)fftwf_malloc(sizeof(fftwf_complex) * nfft);
        fftwf_complex* fftOut = (fftwf_complex*)fftwf_malloc(sizeof(fftwf_complex) * nfft);
        float* cast = (float*)fftIn;
        for (size_t i = 0; i < nfft; ++i) {
            cast[re(i)] = fft_in[re(i)];
            cast[im(i)] = fft_in[im(i)];
        }
        fftwf_plan plan = fftwf_plan_dft_1d(nfft, fftIn, fftOut, FFTW_BACKWARD, FFTW_ESTIMATE);

        //the ground truth
        fftw_complex* fftInd = (fftw_complex*)fftw_malloc(sizeof(fftw_complex) * nfft);
        fftw_complex* fftOutd = (fftw_complex*)fftw_malloc(sizeof(fftw_complex) * nfft);
        double* castd = (double*)fftInd;
        for (size_t i = 0; i < nfft; ++i) {
            castd[re(i)] = (double)(fft_in[re(i)]);
            castd[im(i)] = (double)(fft_in[im(i)]);
        }
        fftw_plan pland = fftw_plan_dft_1d(nfft, fftInd, fftOutd, FFTW_BACKWARD, FFTW_ESTIMATE);
        fftw_execute(pland);


#ifdef WIN32
        QueryPerformanceFrequency(&Frequency);
        QueryPerformanceCounter(&StartingTime);
#else
        clock_gettime(CLOCK_MONOTONIC, &t0);
#endif
        for (int i = 0; i < 10000; ++i) {
            fftwf_execute(plan);
        }
#ifdef WIN32
        QueryPerformanceCounter(&EndingTime);
        ElapsedMicroseconds.QuadPart = EndingTime.QuadPart - StartingTime.QuadPart;
        ElapsedMicroseconds.QuadPart *= 1000000;
        ElapsedMicroseconds.QuadPart /= Frequency.QuadPart;
        printf("fftw : %lld usec \n", ElapsedMicroseconds.QuadPart);
#else
        clock_gettime(CLOCK_MONOTONIC, &t1);
        {struct timespec temp;
        if ((t1.tv_nsec - t0.tv_nsec) < 0) {
            temp.tv_sec = t1.tv_sec - t0.tv_sec - 1;
            temp.tv_nsec = 1000000000 + t1.tv_nsec - t0.tv_nsec;
        }
        else {
            temp.tv_sec = t1.tv_sec - t0.tv_sec;
            temp.tv_nsec = t1.tv_nsec - t0.tv_nsec;
        }
        printf("fftw : %ld s, %ld ns\n", temp.tv_sec, temp.tv_nsec); }
#endif


#ifdef WIN32
        QueryPerformanceFrequency(&Frequency);
        QueryPerformanceCounter(&StartingTime);
#else
        clock_gettime(CLOCK_MONOTONIC, &t0);
#endif
        for (int i = 0; i < 10000; ++i) {
            fft_exec(&p, fft_in, fft_out);
        }
#ifdef WIN32
        QueryPerformanceCounter(&EndingTime);
        ElapsedMicroseconds.QuadPart = EndingTime.QuadPart - StartingTime.QuadPart;
        ElapsedMicroseconds.QuadPart *= 1000000;
        ElapsedMicroseconds.QuadPart /= Frequency.QuadPart;
        printf("fft : %lld usec \n", ElapsedMicroseconds.QuadPart);
#else
        clock_gettime(CLOCK_MONOTONIC, &t1);
        {struct timespec temp;
        if ((t1.tv_nsec - t0.tv_nsec) < 0) {
            temp.tv_sec = t1.tv_sec - t0.tv_sec - 1;
            temp.tv_nsec = 1000000000 + t1.tv_nsec - t0.tv_nsec;
        }
        else {
            temp.tv_sec = t1.tv_sec - t0.tv_sec;
            temp.tv_nsec = t1.tv_nsec - t0.tv_nsec;
        }
        printf("fft : %ld s, %ld ns\n", temp.tv_sec, temp.tv_nsec); }
#endif


        {
            size_t maxreidx, maximidx;
            double err_max_re = 0.0;
            double err_max_im = 0.0;
            double err_rms = 0.0;

            double delta_re;
            double delta_im;

            castd = (double*)fftOutd;
            for (int i = 0; i < nfft; ++i) {
                if (fabs(fft_out[re(i)] - castd[re(i)]) / fabs(castd[re(i)]) > err_max_re) {
                    err_max_re = fabs(fft_out[re(i)] - castd[re(i)]) / fabs(castd[re(i)]);
                    maxreidx = i;
                }
                if (fabs(fft_out[im(i)] - castd[im(i)]) / fabs(castd[im(i)]) > err_max_im) {
                    err_max_im = fabs(fft_out[im(i)] - castd[im(i)]) / fabs(castd[im(i)]);
                    maximidx = i;
                }

                delta_re = fft_out[re(i)] - castd[re(i)];
                delta_im = fft_out[im(i)] - castd[im(i)];
                err_rms += delta_re * delta_re + delta_im * delta_im;
            }
            printf("=======[FFT]=======\n");
            printf("error rms: %1.16f\n", sqrt(err_rms / (double)nfft));
            printf("max error real: %3.16f\n", err_max_re);
            printf("max error imag: %3.16f\n", err_max_im);
            //printf("fft - fftw, real: (%3.16f, %3.16f * I) (%3.16f, %3.16f * I)\n", p.y[re(maxreidx)], p.y[im(maxreidx)], castd[re(maxreidx)], castd[im(maxreidx)]);
            //printf("fft - fftw, imag: (%3.16f, %3.16f * I) (%3.16f, %3.16f * I)\n", p.y[re(maximidx)], p.y[im(maximidx)], castd[re(maximidx)], castd[im(maximidx)]);
        }


        {
            size_t maxreidx, maximidx;
            double err_max_re = 0.0;
            double err_max_im = 0.0;
            double err_rms = 0.0;

            double delta_re;
            double delta_im;

            castd = (double*)fftOutd;
            cast = (float*)fftOut;
            for (int i = 0; i < nfft; ++i) {
                if (fabs(cast[re(i)] - castd[re(i)]) / fabs(castd[re(i)]) > err_max_re) {
                    err_max_re = fabs(cast[re(i)] - castd[re(i)]) / fabs(castd[re(i)]);
                    maxreidx = i;
                }
                if (fabs(cast[im(i)] - castd[im(i)]) / fabs(castd[im(i)]) > err_max_im) {
                    err_max_im = fabs(cast[im(i)] - castd[im(i)]) / fabs(castd[im(i)]);
                    maximidx = i;
                }

                delta_re = cast[re(i)] - castd[re(i)];
                delta_im = cast[im(i)] - castd[im(i)];
                err_rms += delta_re * delta_re + delta_im * delta_im;
            }
            printf("=======[FFTW]========\n");
            printf("error rms = %1.16f\n", sqrt(err_rms / (double)nfft));
            printf("max error real: %3.16f\n", err_max_re);
            printf("max error imag: %3.16f\n", err_max_im);
            //printf("fftwf - fftw, real: (%3.16f, %3.16f * I) (%3.16f, %3.16f * I)\n", cast[re(maxreidx)], cast[im(maxreidx)], castd[re(maxreidx)], castd[im(maxreidx)]);
            //printf("fftwf - fftw, imag: (%3.16f, %3.16f * I) (%3.16f, %3.16f * I)\n", cast[re(maximidx)], cast[im(maximidx)], castd[re(maximidx)], castd[im(maximidx)]);
        }

        fftwf_destroy_plan(plan);
        fftw_destroy_plan(pland);

        fftwf_free(fftIn);
        fftwf_free(fftOut);
        fftw_free(fftInd);
        fftw_free(fftOutd);

        fft_clean(&p);
        _mm_free(fft_in);
        _mm_free(fft_out);
    }




    return 0;
}




