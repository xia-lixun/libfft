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

fft_8_1_leaf(p->x + re(0), p->x + re(128), p->x + re(64), p->x + re(192), p->x + re(32), p->x + re(160), p->x + re(224), p->x + re(96), p->y + re(0));
fft_4_2_leaf(p->x + re(16), p->x + re(144), p->x + re(80), p->x + re(208), p->x + re(240), p->x + re(112), p->x + re(48), p->x + re(176), p->y + re(8));
fft_8_1_leaf(p->x + re(8), p->x + re(136), p->x + re(72), p->x + re(200), p->x + re(40), p->x + re(168), p->x + re(232), p->x + re(104), p->y + re(16));
fft_8_1_leaf(p->x + re(248), p->x + re(120), p->x + re(56), p->x + re(184), p->x + re(24), p->x + re(152), p->x + re(216), p->x + re(88), p->y + re(24));
fft_8_1_leaf(p->x + re(4), p->x + re(132), p->x + re(68), p->x + re(196), p->x + re(36), p->x + re(164), p->x + re(228), p->x + re(100), p->y + re(32));
fft_4_2_leaf(p->x + re(20), p->x + re(148), p->x + re(84), p->x + re(212), p->x + re(244), p->x + re(116), p->x + re(52), p->x + re(180), p->y + re(40));
fft_8_1_leaf(p->x + re(252), p->x + re(124), p->x + re(60), p->x + re(188), p->x + re(28), p->x + re(156), p->x + re(220), p->x + re(92), p->y + re(48));
fft_4_2_leaf(p->x + re(12), p->x + re(140), p->x + re(76), p->x + re(204), p->x + re(236), p->x + re(108), p->x + re(44), p->x + re(172), p->y + re(56));
fft_8_1_leaf(p->x + re(2), p->x + re(130), p->x + re(66), p->x + re(194), p->x + re(34), p->x + re(162), p->x + re(226), p->x + re(98), p->y + re(64));
fft_4_2_leaf(p->x + re(18), p->x + re(146), p->x + re(82), p->x + re(210), p->x + re(242), p->x + re(114), p->x + re(50), p->x + re(178), p->y + re(72));
fft_8_1_leaf(p->x + re(10), p->x + re(138), p->x + re(74), p->x + re(202), p->x + re(42), p->x + re(170), p->x + re(234), p->x + re(106), p->y + re(80));
fft_8_1_leaf(p->x + re(250), p->x + re(122), p->x + re(58), p->x + re(186), p->x + re(26), p->x + re(154), p->x + re(218), p->x + re(90), p->y + re(88));
fft_8_1_leaf(p->x + re(254), p->x + re(126), p->x + re(62), p->x + re(190), p->x + re(30), p->x + re(158), p->x + re(222), p->x + re(94), p->y + re(96));
fft_4_2_leaf(p->x + re(14), p->x + re(142), p->x + re(78), p->x + re(206), p->x + re(238), p->x + re(110), p->x + re(46), p->x + re(174), p->y + re(104));
fft_8_1_leaf(p->x + re(6), p->x + re(134), p->x + re(70), p->x + re(198), p->x + re(38), p->x + re(166), p->x + re(230), p->x + re(102), p->y + re(112));
fft_8_1_leaf(p->x + re(246), p->x + re(118), p->x + re(54), p->x + re(182), p->x + re(22), p->x + re(150), p->x + re(214), p->x + re(86), p->y + re(120));
fft_8_1_leaf(p->x + re(1), p->x + re(129), p->x + re(65), p->x + re(193), p->x + re(33), p->x + re(161), p->x + re(225), p->x + re(97), p->y + re(128));
fft_4_2_leaf(p->x + re(17), p->x + re(145), p->x + re(81), p->x + re(209), p->x + re(241), p->x + re(113), p->x + re(49), p->x + re(177), p->y + re(136));
fft_8_1_leaf(p->x + re(9), p->x + re(137), p->x + re(73), p->x + re(201), p->x + re(41), p->x + re(169), p->x + re(233), p->x + re(105), p->y + re(144));
fft_8_1_leaf(p->x + re(249), p->x + re(121), p->x + re(57), p->x + re(185), p->x + re(25), p->x + re(153), p->x + re(217), p->x + re(89), p->y + re(152));
fft_8_1_leaf(p->x + re(5), p->x + re(133), p->x + re(69), p->x + re(197), p->x + re(37), p->x + re(165), p->x + re(229), p->x + re(101), p->y + re(160));
fft_4_2_leaf(p->x + re(21), p->x + re(149), p->x + re(85), p->x + re(213), p->x + re(245), p->x + re(117), p->x + re(53), p->x + re(181), p->y + re(168));
fft_8_1_leaf(p->x + re(253), p->x + re(125), p->x + re(61), p->x + re(189), p->x + re(29), p->x + re(157), p->x + re(221), p->x + re(93), p->y + re(176));
fft_4_2_leaf(p->x + re(13), p->x + re(141), p->x + re(77), p->x + re(205), p->x + re(237), p->x + re(109), p->x + re(45), p->x + re(173), p->y + re(184));
fft_8_1_leaf(p->x + re(255), p->x + re(127), p->x + re(63), p->x + re(191), p->x + re(31), p->x + re(159), p->x + re(223), p->x + re(95), p->y + re(192));
fft_4_2_leaf(p->x + re(15), p->x + re(143), p->x + re(79), p->x + re(207), p->x + re(239), p->x + re(111), p->x + re(47), p->x + re(175), p->y + re(200));
fft_8_1_leaf(p->x + re(7), p->x + re(135), p->x + re(71), p->x + re(199), p->x + re(39), p->x + re(167), p->x + re(231), p->x + re(103), p->y + re(208));
fft_8_1_leaf(p->x + re(247), p->x + re(119), p->x + re(55), p->x + re(183), p->x + re(23), p->x + re(151), p->x + re(215), p->x + re(87), p->y + re(216));
fft_8_1_leaf(p->x + re(3), p->x + re(131), p->x + re(67), p->x + re(195), p->x + re(35), p->x + re(163), p->x + re(227), p->x + re(99), p->y + re(224));
fft_4_2_leaf(p->x + re(19), p->x + re(147), p->x + re(83), p->x + re(211), p->x + re(243), p->x + re(115), p->x + re(51), p->x + re(179), p->y + re(232));
fft_8_1_leaf(p->x + re(251), p->x + re(123), p->x + re(59), p->x + re(187), p->x + re(27), p->x + re(155), p->x + re(219), p->x + re(91), p->y + re(240));
fft_4_2_leaf(p->x + re(11), p->x + re(139), p->x + re(75), p->x + re(203), p->x + re(235), p->x + re(107), p->x + re(43), p->x + re(171), p->y + re(248));
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
fft_n_(4, p->y + re(128), p->y + re(132), p->y + re(136), p->y + re(140), p->wre_table[1], p->wim_table[1]);
fft_n_(8, p->y + re(128), p->y + re(136), p->y + re(144), p->y + re(152), p->wre_table[2], p->wim_table[2]);
fft_n_(4, p->y + re(160), p->y + re(164), p->y + re(168), p->y + re(172), p->wre_table[1], p->wim_table[1]);
fft_n_(4, p->y + re(176), p->y + re(180), p->y + re(184), p->y + re(188), p->wre_table[1], p->wim_table[1]);
fft_n_(16, p->y + re(128), p->y + re(144), p->y + re(160), p->y + re(176), p->wre_table[3], p->wim_table[3]);
fft_n_(4, p->y + re(192), p->y + re(196), p->y + re(200), p->y + re(204), p->wre_table[1], p->wim_table[1]);
fft_n_(8, p->y + re(192), p->y + re(200), p->y + re(208), p->y + re(216), p->wre_table[2], p->wim_table[2]);
fft_n_(4, p->y + re(224), p->y + re(228), p->y + re(232), p->y + re(236), p->wre_table[1], p->wim_table[1]);
fft_n_(4, p->y + re(240), p->y + re(244), p->y + re(248), p->y + re(252), p->wre_table[1], p->wim_table[1]);
fft_n_(16, p->y + re(192), p->y + re(208), p->y + re(224), p->y + re(240), p->wre_table[3], p->wim_table[3]);
fft_n_(64, p->y + re(0), p->y + re(64), p->y + re(128), p->y + re(192), p->wre_table[5], p->wim_table[5]);
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
	fft_init(&p, 256, -1);

    srand((unsigned int)time(0));
    for (size_t i = 0; i < p.n; i++) {
        p.x[re(i)] = (float)((double)rand() / (double)RAND_MAX);
        p.x[im(i)] = (float)((double)rand() / (double)RAND_MAX);
    }
    //FILE *fid = fopen("./input1024.dat", "w");
    //for (int i = 0; i < N; i++)
    //    fprintf(fid, "%3.16f, %3.16f\n", x[re(i)], x[im(i)]);
    //fclose(fid);


















	fftwf_complex * fftIn = (fftwf_complex*)fftwf_malloc(sizeof(fftwf_complex) * 256);
    fftwf_complex * fftOut = (fftwf_complex*)fftwf_malloc(sizeof(fftwf_complex) * 256);

    float * cast = (float *)fftIn;
	for (size_t i = 0; i < 256; ++i) {
	    cast[re(i)] = p.x[re(i)];
	    cast[im(i)] = p.x[im(i)];
    }

    fftwf_plan plan = fftwf_plan_dft_1d(256, fftIn, fftOut, FFTW_FORWARD, FFTW_ESTIMATE);


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
	for (int i = 0; i < 256; ++i) {
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



