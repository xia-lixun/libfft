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
		p->wre_table[0][0] = -0.0f; 
        p->wre_table[0][1] = 0.0f; 
        p->wre_table[0][2] = 1.0f; 
        p->wre_table[0][3] = -1.0f; 
		p->wim_table[0][0] = 1.0f; 
        p->wim_table[0][1] = 1.0f; 
        p->wim_table[0][2] = (float)(sqrt(2.0)*0.5); 
        p->wim_table[0][3] = (float)(sqrt(2.0)*0.5); 
    } else {
        p->wre_table[0][0] = 0.0f; 
        p->wre_table[0][1] = -0.0f; 
        p->wre_table[0][2] = -1.0f; 
        p->wre_table[0][3] = 1.0f;
        p->wim_table[0][0] = 1.0f; 
        p->wim_table[0][1] = 1.0f; 
        p->wim_table[0][2] = (float)(sqrt(2.0)*0.5); 
        p->wim_table[0][3] = (float)(sqrt(2.0)*0.5);
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

fft_8_1_leaf(p->x + re(0), p->x + re(4096), p->x + re(2048), p->x + re(6144), p->x + re(1024), p->x + re(5120), p->x + re(7168), p->x + re(3072), p->y + re(0));
fft_4_2_leaf(p->x + re(512), p->x + re(4608), p->x + re(2560), p->x + re(6656), p->x + re(7680), p->x + re(3584), p->x + re(1536), p->x + re(5632), p->y + re(8));
fft_8_1_leaf(p->x + re(256), p->x + re(4352), p->x + re(2304), p->x + re(6400), p->x + re(1280), p->x + re(5376), p->x + re(7424), p->x + re(3328), p->y + re(16));
fft_8_1_leaf(p->x + re(7936), p->x + re(3840), p->x + re(1792), p->x + re(5888), p->x + re(768), p->x + re(4864), p->x + re(6912), p->x + re(2816), p->y + re(24));
fft_8_1_leaf(p->x + re(128), p->x + re(4224), p->x + re(2176), p->x + re(6272), p->x + re(1152), p->x + re(5248), p->x + re(7296), p->x + re(3200), p->y + re(32));
fft_4_2_leaf(p->x + re(640), p->x + re(4736), p->x + re(2688), p->x + re(6784), p->x + re(7808), p->x + re(3712), p->x + re(1664), p->x + re(5760), p->y + re(40));
fft_8_1_leaf(p->x + re(8064), p->x + re(3968), p->x + re(1920), p->x + re(6016), p->x + re(896), p->x + re(4992), p->x + re(7040), p->x + re(2944), p->y + re(48));
fft_4_2_leaf(p->x + re(384), p->x + re(4480), p->x + re(2432), p->x + re(6528), p->x + re(7552), p->x + re(3456), p->x + re(1408), p->x + re(5504), p->y + re(56));
fft_8_1_leaf(p->x + re(64), p->x + re(4160), p->x + re(2112), p->x + re(6208), p->x + re(1088), p->x + re(5184), p->x + re(7232), p->x + re(3136), p->y + re(64));
fft_4_2_leaf(p->x + re(576), p->x + re(4672), p->x + re(2624), p->x + re(6720), p->x + re(7744), p->x + re(3648), p->x + re(1600), p->x + re(5696), p->y + re(72));
fft_8_1_leaf(p->x + re(320), p->x + re(4416), p->x + re(2368), p->x + re(6464), p->x + re(1344), p->x + re(5440), p->x + re(7488), p->x + re(3392), p->y + re(80));
fft_8_1_leaf(p->x + re(8000), p->x + re(3904), p->x + re(1856), p->x + re(5952), p->x + re(832), p->x + re(4928), p->x + re(6976), p->x + re(2880), p->y + re(88));
fft_8_1_leaf(p->x + re(8128), p->x + re(4032), p->x + re(1984), p->x + re(6080), p->x + re(960), p->x + re(5056), p->x + re(7104), p->x + re(3008), p->y + re(96));
fft_4_2_leaf(p->x + re(448), p->x + re(4544), p->x + re(2496), p->x + re(6592), p->x + re(7616), p->x + re(3520), p->x + re(1472), p->x + re(5568), p->y + re(104));
fft_8_1_leaf(p->x + re(192), p->x + re(4288), p->x + re(2240), p->x + re(6336), p->x + re(1216), p->x + re(5312), p->x + re(7360), p->x + re(3264), p->y + re(112));
fft_8_1_leaf(p->x + re(7872), p->x + re(3776), p->x + re(1728), p->x + re(5824), p->x + re(704), p->x + re(4800), p->x + re(6848), p->x + re(2752), p->y + re(120));
fft_8_1_leaf(p->x + re(32), p->x + re(4128), p->x + re(2080), p->x + re(6176), p->x + re(1056), p->x + re(5152), p->x + re(7200), p->x + re(3104), p->y + re(128));
fft_4_2_leaf(p->x + re(544), p->x + re(4640), p->x + re(2592), p->x + re(6688), p->x + re(7712), p->x + re(3616), p->x + re(1568), p->x + re(5664), p->y + re(136));
fft_8_1_leaf(p->x + re(288), p->x + re(4384), p->x + re(2336), p->x + re(6432), p->x + re(1312), p->x + re(5408), p->x + re(7456), p->x + re(3360), p->y + re(144));
fft_8_1_leaf(p->x + re(7968), p->x + re(3872), p->x + re(1824), p->x + re(5920), p->x + re(800), p->x + re(4896), p->x + re(6944), p->x + re(2848), p->y + re(152));
fft_8_1_leaf(p->x + re(160), p->x + re(4256), p->x + re(2208), p->x + re(6304), p->x + re(1184), p->x + re(5280), p->x + re(7328), p->x + re(3232), p->y + re(160));
fft_4_2_leaf(p->x + re(672), p->x + re(4768), p->x + re(2720), p->x + re(6816), p->x + re(7840), p->x + re(3744), p->x + re(1696), p->x + re(5792), p->y + re(168));
fft_8_1_leaf(p->x + re(8096), p->x + re(4000), p->x + re(1952), p->x + re(6048), p->x + re(928), p->x + re(5024), p->x + re(7072), p->x + re(2976), p->y + re(176));
fft_4_2_leaf(p->x + re(416), p->x + re(4512), p->x + re(2464), p->x + re(6560), p->x + re(7584), p->x + re(3488), p->x + re(1440), p->x + re(5536), p->y + re(184));
fft_8_1_leaf(p->x + re(8160), p->x + re(4064), p->x + re(2016), p->x + re(6112), p->x + re(992), p->x + re(5088), p->x + re(7136), p->x + re(3040), p->y + re(192));
fft_4_2_leaf(p->x + re(480), p->x + re(4576), p->x + re(2528), p->x + re(6624), p->x + re(7648), p->x + re(3552), p->x + re(1504), p->x + re(5600), p->y + re(200));
fft_8_1_leaf(p->x + re(224), p->x + re(4320), p->x + re(2272), p->x + re(6368), p->x + re(1248), p->x + re(5344), p->x + re(7392), p->x + re(3296), p->y + re(208));
fft_8_1_leaf(p->x + re(7904), p->x + re(3808), p->x + re(1760), p->x + re(5856), p->x + re(736), p->x + re(4832), p->x + re(6880), p->x + re(2784), p->y + re(216));
fft_8_1_leaf(p->x + re(96), p->x + re(4192), p->x + re(2144), p->x + re(6240), p->x + re(1120), p->x + re(5216), p->x + re(7264), p->x + re(3168), p->y + re(224));
fft_4_2_leaf(p->x + re(608), p->x + re(4704), p->x + re(2656), p->x + re(6752), p->x + re(7776), p->x + re(3680), p->x + re(1632), p->x + re(5728), p->y + re(232));
fft_8_1_leaf(p->x + re(8032), p->x + re(3936), p->x + re(1888), p->x + re(5984), p->x + re(864), p->x + re(4960), p->x + re(7008), p->x + re(2912), p->y + re(240));
fft_4_2_leaf(p->x + re(352), p->x + re(4448), p->x + re(2400), p->x + re(6496), p->x + re(7520), p->x + re(3424), p->x + re(1376), p->x + re(5472), p->y + re(248));
fft_8_1_leaf(p->x + re(16), p->x + re(4112), p->x + re(2064), p->x + re(6160), p->x + re(1040), p->x + re(5136), p->x + re(7184), p->x + re(3088), p->y + re(256));
fft_4_2_leaf(p->x + re(528), p->x + re(4624), p->x + re(2576), p->x + re(6672), p->x + re(7696), p->x + re(3600), p->x + re(1552), p->x + re(5648), p->y + re(264));
fft_8_1_leaf(p->x + re(272), p->x + re(4368), p->x + re(2320), p->x + re(6416), p->x + re(1296), p->x + re(5392), p->x + re(7440), p->x + re(3344), p->y + re(272));
fft_8_1_leaf(p->x + re(7952), p->x + re(3856), p->x + re(1808), p->x + re(5904), p->x + re(784), p->x + re(4880), p->x + re(6928), p->x + re(2832), p->y + re(280));
fft_8_1_leaf(p->x + re(144), p->x + re(4240), p->x + re(2192), p->x + re(6288), p->x + re(1168), p->x + re(5264), p->x + re(7312), p->x + re(3216), p->y + re(288));
fft_4_2_leaf(p->x + re(656), p->x + re(4752), p->x + re(2704), p->x + re(6800), p->x + re(7824), p->x + re(3728), p->x + re(1680), p->x + re(5776), p->y + re(296));
fft_8_1_leaf(p->x + re(8080), p->x + re(3984), p->x + re(1936), p->x + re(6032), p->x + re(912), p->x + re(5008), p->x + re(7056), p->x + re(2960), p->y + re(304));
fft_4_2_leaf(p->x + re(400), p->x + re(4496), p->x + re(2448), p->x + re(6544), p->x + re(7568), p->x + re(3472), p->x + re(1424), p->x + re(5520), p->y + re(312));
fft_8_1_leaf(p->x + re(80), p->x + re(4176), p->x + re(2128), p->x + re(6224), p->x + re(1104), p->x + re(5200), p->x + re(7248), p->x + re(3152), p->y + re(320));
fft_4_2_leaf(p->x + re(592), p->x + re(4688), p->x + re(2640), p->x + re(6736), p->x + re(7760), p->x + re(3664), p->x + re(1616), p->x + re(5712), p->y + re(328));
fft_8_1_leaf(p->x + re(336), p->x + re(4432), p->x + re(2384), p->x + re(6480), p->x + re(1360), p->x + re(5456), p->x + re(7504), p->x + re(3408), p->y + re(336));
fft_8_1_leaf(p->x + re(8016), p->x + re(3920), p->x + re(1872), p->x + re(5968), p->x + re(848), p->x + re(4944), p->x + re(6992), p->x + re(2896), p->y + re(344));
fft_8_1_leaf(p->x + re(8144), p->x + re(4048), p->x + re(2000), p->x + re(6096), p->x + re(976), p->x + re(5072), p->x + re(7120), p->x + re(3024), p->y + re(352));
fft_4_2_leaf(p->x + re(464), p->x + re(4560), p->x + re(2512), p->x + re(6608), p->x + re(7632), p->x + re(3536), p->x + re(1488), p->x + re(5584), p->y + re(360));
fft_8_1_leaf(p->x + re(208), p->x + re(4304), p->x + re(2256), p->x + re(6352), p->x + re(1232), p->x + re(5328), p->x + re(7376), p->x + re(3280), p->y + re(368));
fft_8_1_leaf(p->x + re(7888), p->x + re(3792), p->x + re(1744), p->x + re(5840), p->x + re(720), p->x + re(4816), p->x + re(6864), p->x + re(2768), p->y + re(376));
fft_8_1_leaf(p->x + re(8176), p->x + re(4080), p->x + re(2032), p->x + re(6128), p->x + re(1008), p->x + re(5104), p->x + re(7152), p->x + re(3056), p->y + re(384));
fft_4_2_leaf(p->x + re(496), p->x + re(4592), p->x + re(2544), p->x + re(6640), p->x + re(7664), p->x + re(3568), p->x + re(1520), p->x + re(5616), p->y + re(392));
fft_8_1_leaf(p->x + re(240), p->x + re(4336), p->x + re(2288), p->x + re(6384), p->x + re(1264), p->x + re(5360), p->x + re(7408), p->x + re(3312), p->y + re(400));
fft_8_1_leaf(p->x + re(7920), p->x + re(3824), p->x + re(1776), p->x + re(5872), p->x + re(752), p->x + re(4848), p->x + re(6896), p->x + re(2800), p->y + re(408));
fft_8_1_leaf(p->x + re(112), p->x + re(4208), p->x + re(2160), p->x + re(6256), p->x + re(1136), p->x + re(5232), p->x + re(7280), p->x + re(3184), p->y + re(416));
fft_4_2_leaf(p->x + re(624), p->x + re(4720), p->x + re(2672), p->x + re(6768), p->x + re(7792), p->x + re(3696), p->x + re(1648), p->x + re(5744), p->y + re(424));
fft_8_1_leaf(p->x + re(8048), p->x + re(3952), p->x + re(1904), p->x + re(6000), p->x + re(880), p->x + re(4976), p->x + re(7024), p->x + re(2928), p->y + re(432));
fft_4_2_leaf(p->x + re(368), p->x + re(4464), p->x + re(2416), p->x + re(6512), p->x + re(7536), p->x + re(3440), p->x + re(1392), p->x + re(5488), p->y + re(440));
fft_8_1_leaf(p->x + re(48), p->x + re(4144), p->x + re(2096), p->x + re(6192), p->x + re(1072), p->x + re(5168), p->x + re(7216), p->x + re(3120), p->y + re(448));
fft_4_2_leaf(p->x + re(560), p->x + re(4656), p->x + re(2608), p->x + re(6704), p->x + re(7728), p->x + re(3632), p->x + re(1584), p->x + re(5680), p->y + re(456));
fft_8_1_leaf(p->x + re(304), p->x + re(4400), p->x + re(2352), p->x + re(6448), p->x + re(1328), p->x + re(5424), p->x + re(7472), p->x + re(3376), p->y + re(464));
fft_8_1_leaf(p->x + re(7984), p->x + re(3888), p->x + re(1840), p->x + re(5936), p->x + re(816), p->x + re(4912), p->x + re(6960), p->x + re(2864), p->y + re(472));
fft_8_1_leaf(p->x + re(8112), p->x + re(4016), p->x + re(1968), p->x + re(6064), p->x + re(944), p->x + re(5040), p->x + re(7088), p->x + re(2992), p->y + re(480));
fft_4_2_leaf(p->x + re(432), p->x + re(4528), p->x + re(2480), p->x + re(6576), p->x + re(7600), p->x + re(3504), p->x + re(1456), p->x + re(5552), p->y + re(488));
fft_8_1_leaf(p->x + re(176), p->x + re(4272), p->x + re(2224), p->x + re(6320), p->x + re(1200), p->x + re(5296), p->x + re(7344), p->x + re(3248), p->y + re(496));
fft_8_1_leaf(p->x + re(7856), p->x + re(3760), p->x + re(1712), p->x + re(5808), p->x + re(688), p->x + re(4784), p->x + re(6832), p->x + re(2736), p->y + re(504));
fft_8_1_leaf(p->x + re(8), p->x + re(4104), p->x + re(2056), p->x + re(6152), p->x + re(1032), p->x + re(5128), p->x + re(7176), p->x + re(3080), p->y + re(512));
fft_4_2_leaf(p->x + re(520), p->x + re(4616), p->x + re(2568), p->x + re(6664), p->x + re(7688), p->x + re(3592), p->x + re(1544), p->x + re(5640), p->y + re(520));
fft_8_1_leaf(p->x + re(264), p->x + re(4360), p->x + re(2312), p->x + re(6408), p->x + re(1288), p->x + re(5384), p->x + re(7432), p->x + re(3336), p->y + re(528));
fft_8_1_leaf(p->x + re(7944), p->x + re(3848), p->x + re(1800), p->x + re(5896), p->x + re(776), p->x + re(4872), p->x + re(6920), p->x + re(2824), p->y + re(536));
fft_8_1_leaf(p->x + re(136), p->x + re(4232), p->x + re(2184), p->x + re(6280), p->x + re(1160), p->x + re(5256), p->x + re(7304), p->x + re(3208), p->y + re(544));
fft_4_2_leaf(p->x + re(648), p->x + re(4744), p->x + re(2696), p->x + re(6792), p->x + re(7816), p->x + re(3720), p->x + re(1672), p->x + re(5768), p->y + re(552));
fft_8_1_leaf(p->x + re(8072), p->x + re(3976), p->x + re(1928), p->x + re(6024), p->x + re(904), p->x + re(5000), p->x + re(7048), p->x + re(2952), p->y + re(560));
fft_4_2_leaf(p->x + re(392), p->x + re(4488), p->x + re(2440), p->x + re(6536), p->x + re(7560), p->x + re(3464), p->x + re(1416), p->x + re(5512), p->y + re(568));
fft_8_1_leaf(p->x + re(72), p->x + re(4168), p->x + re(2120), p->x + re(6216), p->x + re(1096), p->x + re(5192), p->x + re(7240), p->x + re(3144), p->y + re(576));
fft_4_2_leaf(p->x + re(584), p->x + re(4680), p->x + re(2632), p->x + re(6728), p->x + re(7752), p->x + re(3656), p->x + re(1608), p->x + re(5704), p->y + re(584));
fft_8_1_leaf(p->x + re(328), p->x + re(4424), p->x + re(2376), p->x + re(6472), p->x + re(1352), p->x + re(5448), p->x + re(7496), p->x + re(3400), p->y + re(592));
fft_8_1_leaf(p->x + re(8008), p->x + re(3912), p->x + re(1864), p->x + re(5960), p->x + re(840), p->x + re(4936), p->x + re(6984), p->x + re(2888), p->y + re(600));
fft_8_1_leaf(p->x + re(8136), p->x + re(4040), p->x + re(1992), p->x + re(6088), p->x + re(968), p->x + re(5064), p->x + re(7112), p->x + re(3016), p->y + re(608));
fft_4_2_leaf(p->x + re(456), p->x + re(4552), p->x + re(2504), p->x + re(6600), p->x + re(7624), p->x + re(3528), p->x + re(1480), p->x + re(5576), p->y + re(616));
fft_8_1_leaf(p->x + re(200), p->x + re(4296), p->x + re(2248), p->x + re(6344), p->x + re(1224), p->x + re(5320), p->x + re(7368), p->x + re(3272), p->y + re(624));
fft_8_1_leaf(p->x + re(7880), p->x + re(3784), p->x + re(1736), p->x + re(5832), p->x + re(712), p->x + re(4808), p->x + re(6856), p->x + re(2760), p->y + re(632));
fft_8_1_leaf(p->x + re(40), p->x + re(4136), p->x + re(2088), p->x + re(6184), p->x + re(1064), p->x + re(5160), p->x + re(7208), p->x + re(3112), p->y + re(640));
fft_4_2_leaf(p->x + re(552), p->x + re(4648), p->x + re(2600), p->x + re(6696), p->x + re(7720), p->x + re(3624), p->x + re(1576), p->x + re(5672), p->y + re(648));
fft_8_1_leaf(p->x + re(296), p->x + re(4392), p->x + re(2344), p->x + re(6440), p->x + re(1320), p->x + re(5416), p->x + re(7464), p->x + re(3368), p->y + re(656));
fft_8_1_leaf(p->x + re(7976), p->x + re(3880), p->x + re(1832), p->x + re(5928), p->x + re(808), p->x + re(4904), p->x + re(6952), p->x + re(2856), p->y + re(664));
fft_8_1_leaf(p->x + re(168), p->x + re(4264), p->x + re(2216), p->x + re(6312), p->x + re(1192), p->x + re(5288), p->x + re(7336), p->x + re(3240), p->y + re(672));
fft_4_2_leaf(p->x + re(680), p->x + re(4776), p->x + re(2728), p->x + re(6824), p->x + re(7848), p->x + re(3752), p->x + re(1704), p->x + re(5800), p->y + re(680));
fft_8_1_leaf(p->x + re(8104), p->x + re(4008), p->x + re(1960), p->x + re(6056), p->x + re(936), p->x + re(5032), p->x + re(7080), p->x + re(2984), p->y + re(688));
fft_4_2_leaf(p->x + re(424), p->x + re(4520), p->x + re(2472), p->x + re(6568), p->x + re(7592), p->x + re(3496), p->x + re(1448), p->x + re(5544), p->y + re(696));
fft_8_1_leaf(p->x + re(8168), p->x + re(4072), p->x + re(2024), p->x + re(6120), p->x + re(1000), p->x + re(5096), p->x + re(7144), p->x + re(3048), p->y + re(704));
fft_4_2_leaf(p->x + re(488), p->x + re(4584), p->x + re(2536), p->x + re(6632), p->x + re(7656), p->x + re(3560), p->x + re(1512), p->x + re(5608), p->y + re(712));
fft_8_1_leaf(p->x + re(232), p->x + re(4328), p->x + re(2280), p->x + re(6376), p->x + re(1256), p->x + re(5352), p->x + re(7400), p->x + re(3304), p->y + re(720));
fft_8_1_leaf(p->x + re(7912), p->x + re(3816), p->x + re(1768), p->x + re(5864), p->x + re(744), p->x + re(4840), p->x + re(6888), p->x + re(2792), p->y + re(728));
fft_8_1_leaf(p->x + re(104), p->x + re(4200), p->x + re(2152), p->x + re(6248), p->x + re(1128), p->x + re(5224), p->x + re(7272), p->x + re(3176), p->y + re(736));
fft_4_2_leaf(p->x + re(616), p->x + re(4712), p->x + re(2664), p->x + re(6760), p->x + re(7784), p->x + re(3688), p->x + re(1640), p->x + re(5736), p->y + re(744));
fft_8_1_leaf(p->x + re(8040), p->x + re(3944), p->x + re(1896), p->x + re(5992), p->x + re(872), p->x + re(4968), p->x + re(7016), p->x + re(2920), p->y + re(752));
fft_4_2_leaf(p->x + re(360), p->x + re(4456), p->x + re(2408), p->x + re(6504), p->x + re(7528), p->x + re(3432), p->x + re(1384), p->x + re(5480), p->y + re(760));
fft_8_1_leaf(p->x + re(8184), p->x + re(4088), p->x + re(2040), p->x + re(6136), p->x + re(1016), p->x + re(5112), p->x + re(7160), p->x + re(3064), p->y + re(768));
fft_4_2_leaf(p->x + re(504), p->x + re(4600), p->x + re(2552), p->x + re(6648), p->x + re(7672), p->x + re(3576), p->x + re(1528), p->x + re(5624), p->y + re(776));
fft_8_1_leaf(p->x + re(248), p->x + re(4344), p->x + re(2296), p->x + re(6392), p->x + re(1272), p->x + re(5368), p->x + re(7416), p->x + re(3320), p->y + re(784));
fft_8_1_leaf(p->x + re(7928), p->x + re(3832), p->x + re(1784), p->x + re(5880), p->x + re(760), p->x + re(4856), p->x + re(6904), p->x + re(2808), p->y + re(792));
fft_8_1_leaf(p->x + re(120), p->x + re(4216), p->x + re(2168), p->x + re(6264), p->x + re(1144), p->x + re(5240), p->x + re(7288), p->x + re(3192), p->y + re(800));
fft_4_2_leaf(p->x + re(632), p->x + re(4728), p->x + re(2680), p->x + re(6776), p->x + re(7800), p->x + re(3704), p->x + re(1656), p->x + re(5752), p->y + re(808));
fft_8_1_leaf(p->x + re(8056), p->x + re(3960), p->x + re(1912), p->x + re(6008), p->x + re(888), p->x + re(4984), p->x + re(7032), p->x + re(2936), p->y + re(816));
fft_4_2_leaf(p->x + re(376), p->x + re(4472), p->x + re(2424), p->x + re(6520), p->x + re(7544), p->x + re(3448), p->x + re(1400), p->x + re(5496), p->y + re(824));
fft_8_1_leaf(p->x + re(56), p->x + re(4152), p->x + re(2104), p->x + re(6200), p->x + re(1080), p->x + re(5176), p->x + re(7224), p->x + re(3128), p->y + re(832));
fft_4_2_leaf(p->x + re(568), p->x + re(4664), p->x + re(2616), p->x + re(6712), p->x + re(7736), p->x + re(3640), p->x + re(1592), p->x + re(5688), p->y + re(840));
fft_8_1_leaf(p->x + re(312), p->x + re(4408), p->x + re(2360), p->x + re(6456), p->x + re(1336), p->x + re(5432), p->x + re(7480), p->x + re(3384), p->y + re(848));
fft_8_1_leaf(p->x + re(7992), p->x + re(3896), p->x + re(1848), p->x + re(5944), p->x + re(824), p->x + re(4920), p->x + re(6968), p->x + re(2872), p->y + re(856));
fft_8_1_leaf(p->x + re(8120), p->x + re(4024), p->x + re(1976), p->x + re(6072), p->x + re(952), p->x + re(5048), p->x + re(7096), p->x + re(3000), p->y + re(864));
fft_4_2_leaf(p->x + re(440), p->x + re(4536), p->x + re(2488), p->x + re(6584), p->x + re(7608), p->x + re(3512), p->x + re(1464), p->x + re(5560), p->y + re(872));
fft_8_1_leaf(p->x + re(184), p->x + re(4280), p->x + re(2232), p->x + re(6328), p->x + re(1208), p->x + re(5304), p->x + re(7352), p->x + re(3256), p->y + re(880));
fft_8_1_leaf(p->x + re(7864), p->x + re(3768), p->x + re(1720), p->x + re(5816), p->x + re(696), p->x + re(4792), p->x + re(6840), p->x + re(2744), p->y + re(888));
fft_8_1_leaf(p->x + re(24), p->x + re(4120), p->x + re(2072), p->x + re(6168), p->x + re(1048), p->x + re(5144), p->x + re(7192), p->x + re(3096), p->y + re(896));
fft_4_2_leaf(p->x + re(536), p->x + re(4632), p->x + re(2584), p->x + re(6680), p->x + re(7704), p->x + re(3608), p->x + re(1560), p->x + re(5656), p->y + re(904));
fft_8_1_leaf(p->x + re(280), p->x + re(4376), p->x + re(2328), p->x + re(6424), p->x + re(1304), p->x + re(5400), p->x + re(7448), p->x + re(3352), p->y + re(912));
fft_8_1_leaf(p->x + re(7960), p->x + re(3864), p->x + re(1816), p->x + re(5912), p->x + re(792), p->x + re(4888), p->x + re(6936), p->x + re(2840), p->y + re(920));
fft_8_1_leaf(p->x + re(152), p->x + re(4248), p->x + re(2200), p->x + re(6296), p->x + re(1176), p->x + re(5272), p->x + re(7320), p->x + re(3224), p->y + re(928));
fft_4_2_leaf(p->x + re(664), p->x + re(4760), p->x + re(2712), p->x + re(6808), p->x + re(7832), p->x + re(3736), p->x + re(1688), p->x + re(5784), p->y + re(936));
fft_8_1_leaf(p->x + re(8088), p->x + re(3992), p->x + re(1944), p->x + re(6040), p->x + re(920), p->x + re(5016), p->x + re(7064), p->x + re(2968), p->y + re(944));
fft_4_2_leaf(p->x + re(408), p->x + re(4504), p->x + re(2456), p->x + re(6552), p->x + re(7576), p->x + re(3480), p->x + re(1432), p->x + re(5528), p->y + re(952));
fft_8_1_leaf(p->x + re(8152), p->x + re(4056), p->x + re(2008), p->x + re(6104), p->x + re(984), p->x + re(5080), p->x + re(7128), p->x + re(3032), p->y + re(960));
fft_4_2_leaf(p->x + re(472), p->x + re(4568), p->x + re(2520), p->x + re(6616), p->x + re(7640), p->x + re(3544), p->x + re(1496), p->x + re(5592), p->y + re(968));
fft_8_1_leaf(p->x + re(216), p->x + re(4312), p->x + re(2264), p->x + re(6360), p->x + re(1240), p->x + re(5336), p->x + re(7384), p->x + re(3288), p->y + re(976));
fft_8_1_leaf(p->x + re(7896), p->x + re(3800), p->x + re(1752), p->x + re(5848), p->x + re(728), p->x + re(4824), p->x + re(6872), p->x + re(2776), p->y + re(984));
fft_8_1_leaf(p->x + re(88), p->x + re(4184), p->x + re(2136), p->x + re(6232), p->x + re(1112), p->x + re(5208), p->x + re(7256), p->x + re(3160), p->y + re(992));
fft_4_2_leaf(p->x + re(600), p->x + re(4696), p->x + re(2648), p->x + re(6744), p->x + re(7768), p->x + re(3672), p->x + re(1624), p->x + re(5720), p->y + re(1000));
fft_8_1_leaf(p->x + re(8024), p->x + re(3928), p->x + re(1880), p->x + re(5976), p->x + re(856), p->x + re(4952), p->x + re(7000), p->x + re(2904), p->y + re(1008));
fft_4_2_leaf(p->x + re(344), p->x + re(4440), p->x + re(2392), p->x + re(6488), p->x + re(7512), p->x + re(3416), p->x + re(1368), p->x + re(5464), p->y + re(1016));
fft_8_1_leaf(p->x + re(4), p->x + re(4100), p->x + re(2052), p->x + re(6148), p->x + re(1028), p->x + re(5124), p->x + re(7172), p->x + re(3076), p->y + re(1024));
fft_4_2_leaf(p->x + re(516), p->x + re(4612), p->x + re(2564), p->x + re(6660), p->x + re(7684), p->x + re(3588), p->x + re(1540), p->x + re(5636), p->y + re(1032));
fft_8_1_leaf(p->x + re(260), p->x + re(4356), p->x + re(2308), p->x + re(6404), p->x + re(1284), p->x + re(5380), p->x + re(7428), p->x + re(3332), p->y + re(1040));
fft_8_1_leaf(p->x + re(7940), p->x + re(3844), p->x + re(1796), p->x + re(5892), p->x + re(772), p->x + re(4868), p->x + re(6916), p->x + re(2820), p->y + re(1048));
fft_8_1_leaf(p->x + re(132), p->x + re(4228), p->x + re(2180), p->x + re(6276), p->x + re(1156), p->x + re(5252), p->x + re(7300), p->x + re(3204), p->y + re(1056));
fft_4_2_leaf(p->x + re(644), p->x + re(4740), p->x + re(2692), p->x + re(6788), p->x + re(7812), p->x + re(3716), p->x + re(1668), p->x + re(5764), p->y + re(1064));
fft_8_1_leaf(p->x + re(8068), p->x + re(3972), p->x + re(1924), p->x + re(6020), p->x + re(900), p->x + re(4996), p->x + re(7044), p->x + re(2948), p->y + re(1072));
fft_4_2_leaf(p->x + re(388), p->x + re(4484), p->x + re(2436), p->x + re(6532), p->x + re(7556), p->x + re(3460), p->x + re(1412), p->x + re(5508), p->y + re(1080));
fft_8_1_leaf(p->x + re(68), p->x + re(4164), p->x + re(2116), p->x + re(6212), p->x + re(1092), p->x + re(5188), p->x + re(7236), p->x + re(3140), p->y + re(1088));
fft_4_2_leaf(p->x + re(580), p->x + re(4676), p->x + re(2628), p->x + re(6724), p->x + re(7748), p->x + re(3652), p->x + re(1604), p->x + re(5700), p->y + re(1096));
fft_8_1_leaf(p->x + re(324), p->x + re(4420), p->x + re(2372), p->x + re(6468), p->x + re(1348), p->x + re(5444), p->x + re(7492), p->x + re(3396), p->y + re(1104));
fft_8_1_leaf(p->x + re(8004), p->x + re(3908), p->x + re(1860), p->x + re(5956), p->x + re(836), p->x + re(4932), p->x + re(6980), p->x + re(2884), p->y + re(1112));
fft_8_1_leaf(p->x + re(8132), p->x + re(4036), p->x + re(1988), p->x + re(6084), p->x + re(964), p->x + re(5060), p->x + re(7108), p->x + re(3012), p->y + re(1120));
fft_4_2_leaf(p->x + re(452), p->x + re(4548), p->x + re(2500), p->x + re(6596), p->x + re(7620), p->x + re(3524), p->x + re(1476), p->x + re(5572), p->y + re(1128));
fft_8_1_leaf(p->x + re(196), p->x + re(4292), p->x + re(2244), p->x + re(6340), p->x + re(1220), p->x + re(5316), p->x + re(7364), p->x + re(3268), p->y + re(1136));
fft_8_1_leaf(p->x + re(7876), p->x + re(3780), p->x + re(1732), p->x + re(5828), p->x + re(708), p->x + re(4804), p->x + re(6852), p->x + re(2756), p->y + re(1144));
fft_8_1_leaf(p->x + re(36), p->x + re(4132), p->x + re(2084), p->x + re(6180), p->x + re(1060), p->x + re(5156), p->x + re(7204), p->x + re(3108), p->y + re(1152));
fft_4_2_leaf(p->x + re(548), p->x + re(4644), p->x + re(2596), p->x + re(6692), p->x + re(7716), p->x + re(3620), p->x + re(1572), p->x + re(5668), p->y + re(1160));
fft_8_1_leaf(p->x + re(292), p->x + re(4388), p->x + re(2340), p->x + re(6436), p->x + re(1316), p->x + re(5412), p->x + re(7460), p->x + re(3364), p->y + re(1168));
fft_8_1_leaf(p->x + re(7972), p->x + re(3876), p->x + re(1828), p->x + re(5924), p->x + re(804), p->x + re(4900), p->x + re(6948), p->x + re(2852), p->y + re(1176));
fft_8_1_leaf(p->x + re(164), p->x + re(4260), p->x + re(2212), p->x + re(6308), p->x + re(1188), p->x + re(5284), p->x + re(7332), p->x + re(3236), p->y + re(1184));
fft_4_2_leaf(p->x + re(676), p->x + re(4772), p->x + re(2724), p->x + re(6820), p->x + re(7844), p->x + re(3748), p->x + re(1700), p->x + re(5796), p->y + re(1192));
fft_8_1_leaf(p->x + re(8100), p->x + re(4004), p->x + re(1956), p->x + re(6052), p->x + re(932), p->x + re(5028), p->x + re(7076), p->x + re(2980), p->y + re(1200));
fft_4_2_leaf(p->x + re(420), p->x + re(4516), p->x + re(2468), p->x + re(6564), p->x + re(7588), p->x + re(3492), p->x + re(1444), p->x + re(5540), p->y + re(1208));
fft_8_1_leaf(p->x + re(8164), p->x + re(4068), p->x + re(2020), p->x + re(6116), p->x + re(996), p->x + re(5092), p->x + re(7140), p->x + re(3044), p->y + re(1216));
fft_4_2_leaf(p->x + re(484), p->x + re(4580), p->x + re(2532), p->x + re(6628), p->x + re(7652), p->x + re(3556), p->x + re(1508), p->x + re(5604), p->y + re(1224));
fft_8_1_leaf(p->x + re(228), p->x + re(4324), p->x + re(2276), p->x + re(6372), p->x + re(1252), p->x + re(5348), p->x + re(7396), p->x + re(3300), p->y + re(1232));
fft_8_1_leaf(p->x + re(7908), p->x + re(3812), p->x + re(1764), p->x + re(5860), p->x + re(740), p->x + re(4836), p->x + re(6884), p->x + re(2788), p->y + re(1240));
fft_8_1_leaf(p->x + re(100), p->x + re(4196), p->x + re(2148), p->x + re(6244), p->x + re(1124), p->x + re(5220), p->x + re(7268), p->x + re(3172), p->y + re(1248));
fft_4_2_leaf(p->x + re(612), p->x + re(4708), p->x + re(2660), p->x + re(6756), p->x + re(7780), p->x + re(3684), p->x + re(1636), p->x + re(5732), p->y + re(1256));
fft_8_1_leaf(p->x + re(8036), p->x + re(3940), p->x + re(1892), p->x + re(5988), p->x + re(868), p->x + re(4964), p->x + re(7012), p->x + re(2916), p->y + re(1264));
fft_4_2_leaf(p->x + re(356), p->x + re(4452), p->x + re(2404), p->x + re(6500), p->x + re(7524), p->x + re(3428), p->x + re(1380), p->x + re(5476), p->y + re(1272));
fft_8_1_leaf(p->x + re(20), p->x + re(4116), p->x + re(2068), p->x + re(6164), p->x + re(1044), p->x + re(5140), p->x + re(7188), p->x + re(3092), p->y + re(1280));
fft_4_2_leaf(p->x + re(532), p->x + re(4628), p->x + re(2580), p->x + re(6676), p->x + re(7700), p->x + re(3604), p->x + re(1556), p->x + re(5652), p->y + re(1288));
fft_8_1_leaf(p->x + re(276), p->x + re(4372), p->x + re(2324), p->x + re(6420), p->x + re(1300), p->x + re(5396), p->x + re(7444), p->x + re(3348), p->y + re(1296));
fft_8_1_leaf(p->x + re(7956), p->x + re(3860), p->x + re(1812), p->x + re(5908), p->x + re(788), p->x + re(4884), p->x + re(6932), p->x + re(2836), p->y + re(1304));
fft_8_1_leaf(p->x + re(148), p->x + re(4244), p->x + re(2196), p->x + re(6292), p->x + re(1172), p->x + re(5268), p->x + re(7316), p->x + re(3220), p->y + re(1312));
fft_4_2_leaf(p->x + re(660), p->x + re(4756), p->x + re(2708), p->x + re(6804), p->x + re(7828), p->x + re(3732), p->x + re(1684), p->x + re(5780), p->y + re(1320));
fft_8_1_leaf(p->x + re(8084), p->x + re(3988), p->x + re(1940), p->x + re(6036), p->x + re(916), p->x + re(5012), p->x + re(7060), p->x + re(2964), p->y + re(1328));
fft_4_2_leaf(p->x + re(404), p->x + re(4500), p->x + re(2452), p->x + re(6548), p->x + re(7572), p->x + re(3476), p->x + re(1428), p->x + re(5524), p->y + re(1336));
fft_8_1_leaf(p->x + re(84), p->x + re(4180), p->x + re(2132), p->x + re(6228), p->x + re(1108), p->x + re(5204), p->x + re(7252), p->x + re(3156), p->y + re(1344));
fft_4_2_leaf(p->x + re(596), p->x + re(4692), p->x + re(2644), p->x + re(6740), p->x + re(7764), p->x + re(3668), p->x + re(1620), p->x + re(5716), p->y + re(1352));
fft_8_1_leaf(p->x + re(340), p->x + re(4436), p->x + re(2388), p->x + re(6484), p->x + re(1364), p->x + re(5460), p->x + re(7508), p->x + re(3412), p->y + re(1360));
fft_8_1_leaf(p->x + re(8020), p->x + re(3924), p->x + re(1876), p->x + re(5972), p->x + re(852), p->x + re(4948), p->x + re(6996), p->x + re(2900), p->y + re(1368));
fft_8_1_leaf(p->x + re(8148), p->x + re(4052), p->x + re(2004), p->x + re(6100), p->x + re(980), p->x + re(5076), p->x + re(7124), p->x + re(3028), p->y + re(1376));
fft_4_2_leaf(p->x + re(468), p->x + re(4564), p->x + re(2516), p->x + re(6612), p->x + re(7636), p->x + re(3540), p->x + re(1492), p->x + re(5588), p->y + re(1384));
fft_8_1_leaf(p->x + re(212), p->x + re(4308), p->x + re(2260), p->x + re(6356), p->x + re(1236), p->x + re(5332), p->x + re(7380), p->x + re(3284), p->y + re(1392));
fft_8_1_leaf(p->x + re(7892), p->x + re(3796), p->x + re(1748), p->x + re(5844), p->x + re(724), p->x + re(4820), p->x + re(6868), p->x + re(2772), p->y + re(1400));
fft_8_1_leaf(p->x + re(8180), p->x + re(4084), p->x + re(2036), p->x + re(6132), p->x + re(1012), p->x + re(5108), p->x + re(7156), p->x + re(3060), p->y + re(1408));
fft_4_2_leaf(p->x + re(500), p->x + re(4596), p->x + re(2548), p->x + re(6644), p->x + re(7668), p->x + re(3572), p->x + re(1524), p->x + re(5620), p->y + re(1416));
fft_8_1_leaf(p->x + re(244), p->x + re(4340), p->x + re(2292), p->x + re(6388), p->x + re(1268), p->x + re(5364), p->x + re(7412), p->x + re(3316), p->y + re(1424));
fft_8_1_leaf(p->x + re(7924), p->x + re(3828), p->x + re(1780), p->x + re(5876), p->x + re(756), p->x + re(4852), p->x + re(6900), p->x + re(2804), p->y + re(1432));
fft_8_1_leaf(p->x + re(116), p->x + re(4212), p->x + re(2164), p->x + re(6260), p->x + re(1140), p->x + re(5236), p->x + re(7284), p->x + re(3188), p->y + re(1440));
fft_4_2_leaf(p->x + re(628), p->x + re(4724), p->x + re(2676), p->x + re(6772), p->x + re(7796), p->x + re(3700), p->x + re(1652), p->x + re(5748), p->y + re(1448));
fft_8_1_leaf(p->x + re(8052), p->x + re(3956), p->x + re(1908), p->x + re(6004), p->x + re(884), p->x + re(4980), p->x + re(7028), p->x + re(2932), p->y + re(1456));
fft_4_2_leaf(p->x + re(372), p->x + re(4468), p->x + re(2420), p->x + re(6516), p->x + re(7540), p->x + re(3444), p->x + re(1396), p->x + re(5492), p->y + re(1464));
fft_8_1_leaf(p->x + re(52), p->x + re(4148), p->x + re(2100), p->x + re(6196), p->x + re(1076), p->x + re(5172), p->x + re(7220), p->x + re(3124), p->y + re(1472));
fft_4_2_leaf(p->x + re(564), p->x + re(4660), p->x + re(2612), p->x + re(6708), p->x + re(7732), p->x + re(3636), p->x + re(1588), p->x + re(5684), p->y + re(1480));
fft_8_1_leaf(p->x + re(308), p->x + re(4404), p->x + re(2356), p->x + re(6452), p->x + re(1332), p->x + re(5428), p->x + re(7476), p->x + re(3380), p->y + re(1488));
fft_8_1_leaf(p->x + re(7988), p->x + re(3892), p->x + re(1844), p->x + re(5940), p->x + re(820), p->x + re(4916), p->x + re(6964), p->x + re(2868), p->y + re(1496));
fft_8_1_leaf(p->x + re(8116), p->x + re(4020), p->x + re(1972), p->x + re(6068), p->x + re(948), p->x + re(5044), p->x + re(7092), p->x + re(2996), p->y + re(1504));
fft_4_2_leaf(p->x + re(436), p->x + re(4532), p->x + re(2484), p->x + re(6580), p->x + re(7604), p->x + re(3508), p->x + re(1460), p->x + re(5556), p->y + re(1512));
fft_8_1_leaf(p->x + re(180), p->x + re(4276), p->x + re(2228), p->x + re(6324), p->x + re(1204), p->x + re(5300), p->x + re(7348), p->x + re(3252), p->y + re(1520));
fft_8_1_leaf(p->x + re(7860), p->x + re(3764), p->x + re(1716), p->x + re(5812), p->x + re(692), p->x + re(4788), p->x + re(6836), p->x + re(2740), p->y + re(1528));
fft_8_1_leaf(p->x + re(8188), p->x + re(4092), p->x + re(2044), p->x + re(6140), p->x + re(1020), p->x + re(5116), p->x + re(7164), p->x + re(3068), p->y + re(1536));
fft_4_2_leaf(p->x + re(508), p->x + re(4604), p->x + re(2556), p->x + re(6652), p->x + re(7676), p->x + re(3580), p->x + re(1532), p->x + re(5628), p->y + re(1544));
fft_8_1_leaf(p->x + re(252), p->x + re(4348), p->x + re(2300), p->x + re(6396), p->x + re(1276), p->x + re(5372), p->x + re(7420), p->x + re(3324), p->y + re(1552));
fft_8_1_leaf(p->x + re(7932), p->x + re(3836), p->x + re(1788), p->x + re(5884), p->x + re(764), p->x + re(4860), p->x + re(6908), p->x + re(2812), p->y + re(1560));
fft_8_1_leaf(p->x + re(124), p->x + re(4220), p->x + re(2172), p->x + re(6268), p->x + re(1148), p->x + re(5244), p->x + re(7292), p->x + re(3196), p->y + re(1568));
fft_4_2_leaf(p->x + re(636), p->x + re(4732), p->x + re(2684), p->x + re(6780), p->x + re(7804), p->x + re(3708), p->x + re(1660), p->x + re(5756), p->y + re(1576));
fft_8_1_leaf(p->x + re(8060), p->x + re(3964), p->x + re(1916), p->x + re(6012), p->x + re(892), p->x + re(4988), p->x + re(7036), p->x + re(2940), p->y + re(1584));
fft_4_2_leaf(p->x + re(380), p->x + re(4476), p->x + re(2428), p->x + re(6524), p->x + re(7548), p->x + re(3452), p->x + re(1404), p->x + re(5500), p->y + re(1592));
fft_8_1_leaf(p->x + re(60), p->x + re(4156), p->x + re(2108), p->x + re(6204), p->x + re(1084), p->x + re(5180), p->x + re(7228), p->x + re(3132), p->y + re(1600));
fft_4_2_leaf(p->x + re(572), p->x + re(4668), p->x + re(2620), p->x + re(6716), p->x + re(7740), p->x + re(3644), p->x + re(1596), p->x + re(5692), p->y + re(1608));
fft_8_1_leaf(p->x + re(316), p->x + re(4412), p->x + re(2364), p->x + re(6460), p->x + re(1340), p->x + re(5436), p->x + re(7484), p->x + re(3388), p->y + re(1616));
fft_8_1_leaf(p->x + re(7996), p->x + re(3900), p->x + re(1852), p->x + re(5948), p->x + re(828), p->x + re(4924), p->x + re(6972), p->x + re(2876), p->y + re(1624));
fft_8_1_leaf(p->x + re(8124), p->x + re(4028), p->x + re(1980), p->x + re(6076), p->x + re(956), p->x + re(5052), p->x + re(7100), p->x + re(3004), p->y + re(1632));
fft_4_2_leaf(p->x + re(444), p->x + re(4540), p->x + re(2492), p->x + re(6588), p->x + re(7612), p->x + re(3516), p->x + re(1468), p->x + re(5564), p->y + re(1640));
fft_8_1_leaf(p->x + re(188), p->x + re(4284), p->x + re(2236), p->x + re(6332), p->x + re(1212), p->x + re(5308), p->x + re(7356), p->x + re(3260), p->y + re(1648));
fft_8_1_leaf(p->x + re(7868), p->x + re(3772), p->x + re(1724), p->x + re(5820), p->x + re(700), p->x + re(4796), p->x + re(6844), p->x + re(2748), p->y + re(1656));
fft_8_1_leaf(p->x + re(28), p->x + re(4124), p->x + re(2076), p->x + re(6172), p->x + re(1052), p->x + re(5148), p->x + re(7196), p->x + re(3100), p->y + re(1664));
fft_4_2_leaf(p->x + re(540), p->x + re(4636), p->x + re(2588), p->x + re(6684), p->x + re(7708), p->x + re(3612), p->x + re(1564), p->x + re(5660), p->y + re(1672));
fft_8_1_leaf(p->x + re(284), p->x + re(4380), p->x + re(2332), p->x + re(6428), p->x + re(1308), p->x + re(5404), p->x + re(7452), p->x + re(3356), p->y + re(1680));
fft_8_1_leaf(p->x + re(7964), p->x + re(3868), p->x + re(1820), p->x + re(5916), p->x + re(796), p->x + re(4892), p->x + re(6940), p->x + re(2844), p->y + re(1688));
fft_8_1_leaf(p->x + re(156), p->x + re(4252), p->x + re(2204), p->x + re(6300), p->x + re(1180), p->x + re(5276), p->x + re(7324), p->x + re(3228), p->y + re(1696));
fft_4_2_leaf(p->x + re(668), p->x + re(4764), p->x + re(2716), p->x + re(6812), p->x + re(7836), p->x + re(3740), p->x + re(1692), p->x + re(5788), p->y + re(1704));
fft_8_1_leaf(p->x + re(8092), p->x + re(3996), p->x + re(1948), p->x + re(6044), p->x + re(924), p->x + re(5020), p->x + re(7068), p->x + re(2972), p->y + re(1712));
fft_4_2_leaf(p->x + re(412), p->x + re(4508), p->x + re(2460), p->x + re(6556), p->x + re(7580), p->x + re(3484), p->x + re(1436), p->x + re(5532), p->y + re(1720));
fft_8_1_leaf(p->x + re(8156), p->x + re(4060), p->x + re(2012), p->x + re(6108), p->x + re(988), p->x + re(5084), p->x + re(7132), p->x + re(3036), p->y + re(1728));
fft_4_2_leaf(p->x + re(476), p->x + re(4572), p->x + re(2524), p->x + re(6620), p->x + re(7644), p->x + re(3548), p->x + re(1500), p->x + re(5596), p->y + re(1736));
fft_8_1_leaf(p->x + re(220), p->x + re(4316), p->x + re(2268), p->x + re(6364), p->x + re(1244), p->x + re(5340), p->x + re(7388), p->x + re(3292), p->y + re(1744));
fft_8_1_leaf(p->x + re(7900), p->x + re(3804), p->x + re(1756), p->x + re(5852), p->x + re(732), p->x + re(4828), p->x + re(6876), p->x + re(2780), p->y + re(1752));
fft_8_1_leaf(p->x + re(92), p->x + re(4188), p->x + re(2140), p->x + re(6236), p->x + re(1116), p->x + re(5212), p->x + re(7260), p->x + re(3164), p->y + re(1760));
fft_4_2_leaf(p->x + re(604), p->x + re(4700), p->x + re(2652), p->x + re(6748), p->x + re(7772), p->x + re(3676), p->x + re(1628), p->x + re(5724), p->y + re(1768));
fft_8_1_leaf(p->x + re(8028), p->x + re(3932), p->x + re(1884), p->x + re(5980), p->x + re(860), p->x + re(4956), p->x + re(7004), p->x + re(2908), p->y + re(1776));
fft_4_2_leaf(p->x + re(348), p->x + re(4444), p->x + re(2396), p->x + re(6492), p->x + re(7516), p->x + re(3420), p->x + re(1372), p->x + re(5468), p->y + re(1784));
fft_8_1_leaf(p->x + re(12), p->x + re(4108), p->x + re(2060), p->x + re(6156), p->x + re(1036), p->x + re(5132), p->x + re(7180), p->x + re(3084), p->y + re(1792));
fft_4_2_leaf(p->x + re(524), p->x + re(4620), p->x + re(2572), p->x + re(6668), p->x + re(7692), p->x + re(3596), p->x + re(1548), p->x + re(5644), p->y + re(1800));
fft_8_1_leaf(p->x + re(268), p->x + re(4364), p->x + re(2316), p->x + re(6412), p->x + re(1292), p->x + re(5388), p->x + re(7436), p->x + re(3340), p->y + re(1808));
fft_8_1_leaf(p->x + re(7948), p->x + re(3852), p->x + re(1804), p->x + re(5900), p->x + re(780), p->x + re(4876), p->x + re(6924), p->x + re(2828), p->y + re(1816));
fft_8_1_leaf(p->x + re(140), p->x + re(4236), p->x + re(2188), p->x + re(6284), p->x + re(1164), p->x + re(5260), p->x + re(7308), p->x + re(3212), p->y + re(1824));
fft_4_2_leaf(p->x + re(652), p->x + re(4748), p->x + re(2700), p->x + re(6796), p->x + re(7820), p->x + re(3724), p->x + re(1676), p->x + re(5772), p->y + re(1832));
fft_8_1_leaf(p->x + re(8076), p->x + re(3980), p->x + re(1932), p->x + re(6028), p->x + re(908), p->x + re(5004), p->x + re(7052), p->x + re(2956), p->y + re(1840));
fft_4_2_leaf(p->x + re(396), p->x + re(4492), p->x + re(2444), p->x + re(6540), p->x + re(7564), p->x + re(3468), p->x + re(1420), p->x + re(5516), p->y + re(1848));
fft_8_1_leaf(p->x + re(76), p->x + re(4172), p->x + re(2124), p->x + re(6220), p->x + re(1100), p->x + re(5196), p->x + re(7244), p->x + re(3148), p->y + re(1856));
fft_4_2_leaf(p->x + re(588), p->x + re(4684), p->x + re(2636), p->x + re(6732), p->x + re(7756), p->x + re(3660), p->x + re(1612), p->x + re(5708), p->y + re(1864));
fft_8_1_leaf(p->x + re(332), p->x + re(4428), p->x + re(2380), p->x + re(6476), p->x + re(1356), p->x + re(5452), p->x + re(7500), p->x + re(3404), p->y + re(1872));
fft_8_1_leaf(p->x + re(8012), p->x + re(3916), p->x + re(1868), p->x + re(5964), p->x + re(844), p->x + re(4940), p->x + re(6988), p->x + re(2892), p->y + re(1880));
fft_8_1_leaf(p->x + re(8140), p->x + re(4044), p->x + re(1996), p->x + re(6092), p->x + re(972), p->x + re(5068), p->x + re(7116), p->x + re(3020), p->y + re(1888));
fft_4_2_leaf(p->x + re(460), p->x + re(4556), p->x + re(2508), p->x + re(6604), p->x + re(7628), p->x + re(3532), p->x + re(1484), p->x + re(5580), p->y + re(1896));
fft_8_1_leaf(p->x + re(204), p->x + re(4300), p->x + re(2252), p->x + re(6348), p->x + re(1228), p->x + re(5324), p->x + re(7372), p->x + re(3276), p->y + re(1904));
fft_8_1_leaf(p->x + re(7884), p->x + re(3788), p->x + re(1740), p->x + re(5836), p->x + re(716), p->x + re(4812), p->x + re(6860), p->x + re(2764), p->y + re(1912));
fft_8_1_leaf(p->x + re(8172), p->x + re(4076), p->x + re(2028), p->x + re(6124), p->x + re(1004), p->x + re(5100), p->x + re(7148), p->x + re(3052), p->y + re(1920));
fft_4_2_leaf(p->x + re(492), p->x + re(4588), p->x + re(2540), p->x + re(6636), p->x + re(7660), p->x + re(3564), p->x + re(1516), p->x + re(5612), p->y + re(1928));
fft_8_1_leaf(p->x + re(236), p->x + re(4332), p->x + re(2284), p->x + re(6380), p->x + re(1260), p->x + re(5356), p->x + re(7404), p->x + re(3308), p->y + re(1936));
fft_8_1_leaf(p->x + re(7916), p->x + re(3820), p->x + re(1772), p->x + re(5868), p->x + re(748), p->x + re(4844), p->x + re(6892), p->x + re(2796), p->y + re(1944));
fft_8_1_leaf(p->x + re(108), p->x + re(4204), p->x + re(2156), p->x + re(6252), p->x + re(1132), p->x + re(5228), p->x + re(7276), p->x + re(3180), p->y + re(1952));
fft_4_2_leaf(p->x + re(620), p->x + re(4716), p->x + re(2668), p->x + re(6764), p->x + re(7788), p->x + re(3692), p->x + re(1644), p->x + re(5740), p->y + re(1960));
fft_8_1_leaf(p->x + re(8044), p->x + re(3948), p->x + re(1900), p->x + re(5996), p->x + re(876), p->x + re(4972), p->x + re(7020), p->x + re(2924), p->y + re(1968));
fft_4_2_leaf(p->x + re(364), p->x + re(4460), p->x + re(2412), p->x + re(6508), p->x + re(7532), p->x + re(3436), p->x + re(1388), p->x + re(5484), p->y + re(1976));
fft_8_1_leaf(p->x + re(44), p->x + re(4140), p->x + re(2092), p->x + re(6188), p->x + re(1068), p->x + re(5164), p->x + re(7212), p->x + re(3116), p->y + re(1984));
fft_4_2_leaf(p->x + re(556), p->x + re(4652), p->x + re(2604), p->x + re(6700), p->x + re(7724), p->x + re(3628), p->x + re(1580), p->x + re(5676), p->y + re(1992));
fft_8_1_leaf(p->x + re(300), p->x + re(4396), p->x + re(2348), p->x + re(6444), p->x + re(1324), p->x + re(5420), p->x + re(7468), p->x + re(3372), p->y + re(2000));
fft_8_1_leaf(p->x + re(7980), p->x + re(3884), p->x + re(1836), p->x + re(5932), p->x + re(812), p->x + re(4908), p->x + re(6956), p->x + re(2860), p->y + re(2008));
fft_8_1_leaf(p->x + re(8108), p->x + re(4012), p->x + re(1964), p->x + re(6060), p->x + re(940), p->x + re(5036), p->x + re(7084), p->x + re(2988), p->y + re(2016));
fft_4_2_leaf(p->x + re(428), p->x + re(4524), p->x + re(2476), p->x + re(6572), p->x + re(7596), p->x + re(3500), p->x + re(1452), p->x + re(5548), p->y + re(2024));
fft_8_1_leaf(p->x + re(172), p->x + re(4268), p->x + re(2220), p->x + re(6316), p->x + re(1196), p->x + re(5292), p->x + re(7340), p->x + re(3244), p->y + re(2032));
fft_8_1_leaf(p->x + re(7852), p->x + re(3756), p->x + re(1708), p->x + re(5804), p->x + re(684), p->x + re(4780), p->x + re(6828), p->x + re(2732), p->y + re(2040));
fft_8_1_leaf(p->x + re(2), p->x + re(4098), p->x + re(2050), p->x + re(6146), p->x + re(1026), p->x + re(5122), p->x + re(7170), p->x + re(3074), p->y + re(2048));
fft_4_2_leaf(p->x + re(514), p->x + re(4610), p->x + re(2562), p->x + re(6658), p->x + re(7682), p->x + re(3586), p->x + re(1538), p->x + re(5634), p->y + re(2056));
fft_8_1_leaf(p->x + re(258), p->x + re(4354), p->x + re(2306), p->x + re(6402), p->x + re(1282), p->x + re(5378), p->x + re(7426), p->x + re(3330), p->y + re(2064));
fft_8_1_leaf(p->x + re(7938), p->x + re(3842), p->x + re(1794), p->x + re(5890), p->x + re(770), p->x + re(4866), p->x + re(6914), p->x + re(2818), p->y + re(2072));
fft_8_1_leaf(p->x + re(130), p->x + re(4226), p->x + re(2178), p->x + re(6274), p->x + re(1154), p->x + re(5250), p->x + re(7298), p->x + re(3202), p->y + re(2080));
fft_4_2_leaf(p->x + re(642), p->x + re(4738), p->x + re(2690), p->x + re(6786), p->x + re(7810), p->x + re(3714), p->x + re(1666), p->x + re(5762), p->y + re(2088));
fft_8_1_leaf(p->x + re(8066), p->x + re(3970), p->x + re(1922), p->x + re(6018), p->x + re(898), p->x + re(4994), p->x + re(7042), p->x + re(2946), p->y + re(2096));
fft_4_2_leaf(p->x + re(386), p->x + re(4482), p->x + re(2434), p->x + re(6530), p->x + re(7554), p->x + re(3458), p->x + re(1410), p->x + re(5506), p->y + re(2104));
fft_8_1_leaf(p->x + re(66), p->x + re(4162), p->x + re(2114), p->x + re(6210), p->x + re(1090), p->x + re(5186), p->x + re(7234), p->x + re(3138), p->y + re(2112));
fft_4_2_leaf(p->x + re(578), p->x + re(4674), p->x + re(2626), p->x + re(6722), p->x + re(7746), p->x + re(3650), p->x + re(1602), p->x + re(5698), p->y + re(2120));
fft_8_1_leaf(p->x + re(322), p->x + re(4418), p->x + re(2370), p->x + re(6466), p->x + re(1346), p->x + re(5442), p->x + re(7490), p->x + re(3394), p->y + re(2128));
fft_8_1_leaf(p->x + re(8002), p->x + re(3906), p->x + re(1858), p->x + re(5954), p->x + re(834), p->x + re(4930), p->x + re(6978), p->x + re(2882), p->y + re(2136));
fft_8_1_leaf(p->x + re(8130), p->x + re(4034), p->x + re(1986), p->x + re(6082), p->x + re(962), p->x + re(5058), p->x + re(7106), p->x + re(3010), p->y + re(2144));
fft_4_2_leaf(p->x + re(450), p->x + re(4546), p->x + re(2498), p->x + re(6594), p->x + re(7618), p->x + re(3522), p->x + re(1474), p->x + re(5570), p->y + re(2152));
fft_8_1_leaf(p->x + re(194), p->x + re(4290), p->x + re(2242), p->x + re(6338), p->x + re(1218), p->x + re(5314), p->x + re(7362), p->x + re(3266), p->y + re(2160));
fft_8_1_leaf(p->x + re(7874), p->x + re(3778), p->x + re(1730), p->x + re(5826), p->x + re(706), p->x + re(4802), p->x + re(6850), p->x + re(2754), p->y + re(2168));
fft_8_1_leaf(p->x + re(34), p->x + re(4130), p->x + re(2082), p->x + re(6178), p->x + re(1058), p->x + re(5154), p->x + re(7202), p->x + re(3106), p->y + re(2176));
fft_4_2_leaf(p->x + re(546), p->x + re(4642), p->x + re(2594), p->x + re(6690), p->x + re(7714), p->x + re(3618), p->x + re(1570), p->x + re(5666), p->y + re(2184));
fft_8_1_leaf(p->x + re(290), p->x + re(4386), p->x + re(2338), p->x + re(6434), p->x + re(1314), p->x + re(5410), p->x + re(7458), p->x + re(3362), p->y + re(2192));
fft_8_1_leaf(p->x + re(7970), p->x + re(3874), p->x + re(1826), p->x + re(5922), p->x + re(802), p->x + re(4898), p->x + re(6946), p->x + re(2850), p->y + re(2200));
fft_8_1_leaf(p->x + re(162), p->x + re(4258), p->x + re(2210), p->x + re(6306), p->x + re(1186), p->x + re(5282), p->x + re(7330), p->x + re(3234), p->y + re(2208));
fft_4_2_leaf(p->x + re(674), p->x + re(4770), p->x + re(2722), p->x + re(6818), p->x + re(7842), p->x + re(3746), p->x + re(1698), p->x + re(5794), p->y + re(2216));
fft_8_1_leaf(p->x + re(8098), p->x + re(4002), p->x + re(1954), p->x + re(6050), p->x + re(930), p->x + re(5026), p->x + re(7074), p->x + re(2978), p->y + re(2224));
fft_4_2_leaf(p->x + re(418), p->x + re(4514), p->x + re(2466), p->x + re(6562), p->x + re(7586), p->x + re(3490), p->x + re(1442), p->x + re(5538), p->y + re(2232));
fft_8_1_leaf(p->x + re(8162), p->x + re(4066), p->x + re(2018), p->x + re(6114), p->x + re(994), p->x + re(5090), p->x + re(7138), p->x + re(3042), p->y + re(2240));
fft_4_2_leaf(p->x + re(482), p->x + re(4578), p->x + re(2530), p->x + re(6626), p->x + re(7650), p->x + re(3554), p->x + re(1506), p->x + re(5602), p->y + re(2248));
fft_8_1_leaf(p->x + re(226), p->x + re(4322), p->x + re(2274), p->x + re(6370), p->x + re(1250), p->x + re(5346), p->x + re(7394), p->x + re(3298), p->y + re(2256));
fft_8_1_leaf(p->x + re(7906), p->x + re(3810), p->x + re(1762), p->x + re(5858), p->x + re(738), p->x + re(4834), p->x + re(6882), p->x + re(2786), p->y + re(2264));
fft_8_1_leaf(p->x + re(98), p->x + re(4194), p->x + re(2146), p->x + re(6242), p->x + re(1122), p->x + re(5218), p->x + re(7266), p->x + re(3170), p->y + re(2272));
fft_4_2_leaf(p->x + re(610), p->x + re(4706), p->x + re(2658), p->x + re(6754), p->x + re(7778), p->x + re(3682), p->x + re(1634), p->x + re(5730), p->y + re(2280));
fft_8_1_leaf(p->x + re(8034), p->x + re(3938), p->x + re(1890), p->x + re(5986), p->x + re(866), p->x + re(4962), p->x + re(7010), p->x + re(2914), p->y + re(2288));
fft_4_2_leaf(p->x + re(354), p->x + re(4450), p->x + re(2402), p->x + re(6498), p->x + re(7522), p->x + re(3426), p->x + re(1378), p->x + re(5474), p->y + re(2296));
fft_8_1_leaf(p->x + re(18), p->x + re(4114), p->x + re(2066), p->x + re(6162), p->x + re(1042), p->x + re(5138), p->x + re(7186), p->x + re(3090), p->y + re(2304));
fft_4_2_leaf(p->x + re(530), p->x + re(4626), p->x + re(2578), p->x + re(6674), p->x + re(7698), p->x + re(3602), p->x + re(1554), p->x + re(5650), p->y + re(2312));
fft_8_1_leaf(p->x + re(274), p->x + re(4370), p->x + re(2322), p->x + re(6418), p->x + re(1298), p->x + re(5394), p->x + re(7442), p->x + re(3346), p->y + re(2320));
fft_8_1_leaf(p->x + re(7954), p->x + re(3858), p->x + re(1810), p->x + re(5906), p->x + re(786), p->x + re(4882), p->x + re(6930), p->x + re(2834), p->y + re(2328));
fft_8_1_leaf(p->x + re(146), p->x + re(4242), p->x + re(2194), p->x + re(6290), p->x + re(1170), p->x + re(5266), p->x + re(7314), p->x + re(3218), p->y + re(2336));
fft_4_2_leaf(p->x + re(658), p->x + re(4754), p->x + re(2706), p->x + re(6802), p->x + re(7826), p->x + re(3730), p->x + re(1682), p->x + re(5778), p->y + re(2344));
fft_8_1_leaf(p->x + re(8082), p->x + re(3986), p->x + re(1938), p->x + re(6034), p->x + re(914), p->x + re(5010), p->x + re(7058), p->x + re(2962), p->y + re(2352));
fft_4_2_leaf(p->x + re(402), p->x + re(4498), p->x + re(2450), p->x + re(6546), p->x + re(7570), p->x + re(3474), p->x + re(1426), p->x + re(5522), p->y + re(2360));
fft_8_1_leaf(p->x + re(82), p->x + re(4178), p->x + re(2130), p->x + re(6226), p->x + re(1106), p->x + re(5202), p->x + re(7250), p->x + re(3154), p->y + re(2368));
fft_4_2_leaf(p->x + re(594), p->x + re(4690), p->x + re(2642), p->x + re(6738), p->x + re(7762), p->x + re(3666), p->x + re(1618), p->x + re(5714), p->y + re(2376));
fft_8_1_leaf(p->x + re(338), p->x + re(4434), p->x + re(2386), p->x + re(6482), p->x + re(1362), p->x + re(5458), p->x + re(7506), p->x + re(3410), p->y + re(2384));
fft_8_1_leaf(p->x + re(8018), p->x + re(3922), p->x + re(1874), p->x + re(5970), p->x + re(850), p->x + re(4946), p->x + re(6994), p->x + re(2898), p->y + re(2392));
fft_8_1_leaf(p->x + re(8146), p->x + re(4050), p->x + re(2002), p->x + re(6098), p->x + re(978), p->x + re(5074), p->x + re(7122), p->x + re(3026), p->y + re(2400));
fft_4_2_leaf(p->x + re(466), p->x + re(4562), p->x + re(2514), p->x + re(6610), p->x + re(7634), p->x + re(3538), p->x + re(1490), p->x + re(5586), p->y + re(2408));
fft_8_1_leaf(p->x + re(210), p->x + re(4306), p->x + re(2258), p->x + re(6354), p->x + re(1234), p->x + re(5330), p->x + re(7378), p->x + re(3282), p->y + re(2416));
fft_8_1_leaf(p->x + re(7890), p->x + re(3794), p->x + re(1746), p->x + re(5842), p->x + re(722), p->x + re(4818), p->x + re(6866), p->x + re(2770), p->y + re(2424));
fft_8_1_leaf(p->x + re(8178), p->x + re(4082), p->x + re(2034), p->x + re(6130), p->x + re(1010), p->x + re(5106), p->x + re(7154), p->x + re(3058), p->y + re(2432));
fft_4_2_leaf(p->x + re(498), p->x + re(4594), p->x + re(2546), p->x + re(6642), p->x + re(7666), p->x + re(3570), p->x + re(1522), p->x + re(5618), p->y + re(2440));
fft_8_1_leaf(p->x + re(242), p->x + re(4338), p->x + re(2290), p->x + re(6386), p->x + re(1266), p->x + re(5362), p->x + re(7410), p->x + re(3314), p->y + re(2448));
fft_8_1_leaf(p->x + re(7922), p->x + re(3826), p->x + re(1778), p->x + re(5874), p->x + re(754), p->x + re(4850), p->x + re(6898), p->x + re(2802), p->y + re(2456));
fft_8_1_leaf(p->x + re(114), p->x + re(4210), p->x + re(2162), p->x + re(6258), p->x + re(1138), p->x + re(5234), p->x + re(7282), p->x + re(3186), p->y + re(2464));
fft_4_2_leaf(p->x + re(626), p->x + re(4722), p->x + re(2674), p->x + re(6770), p->x + re(7794), p->x + re(3698), p->x + re(1650), p->x + re(5746), p->y + re(2472));
fft_8_1_leaf(p->x + re(8050), p->x + re(3954), p->x + re(1906), p->x + re(6002), p->x + re(882), p->x + re(4978), p->x + re(7026), p->x + re(2930), p->y + re(2480));
fft_4_2_leaf(p->x + re(370), p->x + re(4466), p->x + re(2418), p->x + re(6514), p->x + re(7538), p->x + re(3442), p->x + re(1394), p->x + re(5490), p->y + re(2488));
fft_8_1_leaf(p->x + re(50), p->x + re(4146), p->x + re(2098), p->x + re(6194), p->x + re(1074), p->x + re(5170), p->x + re(7218), p->x + re(3122), p->y + re(2496));
fft_4_2_leaf(p->x + re(562), p->x + re(4658), p->x + re(2610), p->x + re(6706), p->x + re(7730), p->x + re(3634), p->x + re(1586), p->x + re(5682), p->y + re(2504));
fft_8_1_leaf(p->x + re(306), p->x + re(4402), p->x + re(2354), p->x + re(6450), p->x + re(1330), p->x + re(5426), p->x + re(7474), p->x + re(3378), p->y + re(2512));
fft_8_1_leaf(p->x + re(7986), p->x + re(3890), p->x + re(1842), p->x + re(5938), p->x + re(818), p->x + re(4914), p->x + re(6962), p->x + re(2866), p->y + re(2520));
fft_8_1_leaf(p->x + re(8114), p->x + re(4018), p->x + re(1970), p->x + re(6066), p->x + re(946), p->x + re(5042), p->x + re(7090), p->x + re(2994), p->y + re(2528));
fft_4_2_leaf(p->x + re(434), p->x + re(4530), p->x + re(2482), p->x + re(6578), p->x + re(7602), p->x + re(3506), p->x + re(1458), p->x + re(5554), p->y + re(2536));
fft_8_1_leaf(p->x + re(178), p->x + re(4274), p->x + re(2226), p->x + re(6322), p->x + re(1202), p->x + re(5298), p->x + re(7346), p->x + re(3250), p->y + re(2544));
fft_8_1_leaf(p->x + re(7858), p->x + re(3762), p->x + re(1714), p->x + re(5810), p->x + re(690), p->x + re(4786), p->x + re(6834), p->x + re(2738), p->y + re(2552));
fft_8_1_leaf(p->x + re(10), p->x + re(4106), p->x + re(2058), p->x + re(6154), p->x + re(1034), p->x + re(5130), p->x + re(7178), p->x + re(3082), p->y + re(2560));
fft_4_2_leaf(p->x + re(522), p->x + re(4618), p->x + re(2570), p->x + re(6666), p->x + re(7690), p->x + re(3594), p->x + re(1546), p->x + re(5642), p->y + re(2568));
fft_8_1_leaf(p->x + re(266), p->x + re(4362), p->x + re(2314), p->x + re(6410), p->x + re(1290), p->x + re(5386), p->x + re(7434), p->x + re(3338), p->y + re(2576));
fft_8_1_leaf(p->x + re(7946), p->x + re(3850), p->x + re(1802), p->x + re(5898), p->x + re(778), p->x + re(4874), p->x + re(6922), p->x + re(2826), p->y + re(2584));
fft_8_1_leaf(p->x + re(138), p->x + re(4234), p->x + re(2186), p->x + re(6282), p->x + re(1162), p->x + re(5258), p->x + re(7306), p->x + re(3210), p->y + re(2592));
fft_4_2_leaf(p->x + re(650), p->x + re(4746), p->x + re(2698), p->x + re(6794), p->x + re(7818), p->x + re(3722), p->x + re(1674), p->x + re(5770), p->y + re(2600));
fft_8_1_leaf(p->x + re(8074), p->x + re(3978), p->x + re(1930), p->x + re(6026), p->x + re(906), p->x + re(5002), p->x + re(7050), p->x + re(2954), p->y + re(2608));
fft_4_2_leaf(p->x + re(394), p->x + re(4490), p->x + re(2442), p->x + re(6538), p->x + re(7562), p->x + re(3466), p->x + re(1418), p->x + re(5514), p->y + re(2616));
fft_8_1_leaf(p->x + re(74), p->x + re(4170), p->x + re(2122), p->x + re(6218), p->x + re(1098), p->x + re(5194), p->x + re(7242), p->x + re(3146), p->y + re(2624));
fft_4_2_leaf(p->x + re(586), p->x + re(4682), p->x + re(2634), p->x + re(6730), p->x + re(7754), p->x + re(3658), p->x + re(1610), p->x + re(5706), p->y + re(2632));
fft_8_1_leaf(p->x + re(330), p->x + re(4426), p->x + re(2378), p->x + re(6474), p->x + re(1354), p->x + re(5450), p->x + re(7498), p->x + re(3402), p->y + re(2640));
fft_8_1_leaf(p->x + re(8010), p->x + re(3914), p->x + re(1866), p->x + re(5962), p->x + re(842), p->x + re(4938), p->x + re(6986), p->x + re(2890), p->y + re(2648));
fft_8_1_leaf(p->x + re(8138), p->x + re(4042), p->x + re(1994), p->x + re(6090), p->x + re(970), p->x + re(5066), p->x + re(7114), p->x + re(3018), p->y + re(2656));
fft_4_2_leaf(p->x + re(458), p->x + re(4554), p->x + re(2506), p->x + re(6602), p->x + re(7626), p->x + re(3530), p->x + re(1482), p->x + re(5578), p->y + re(2664));
fft_8_1_leaf(p->x + re(202), p->x + re(4298), p->x + re(2250), p->x + re(6346), p->x + re(1226), p->x + re(5322), p->x + re(7370), p->x + re(3274), p->y + re(2672));
fft_8_1_leaf(p->x + re(7882), p->x + re(3786), p->x + re(1738), p->x + re(5834), p->x + re(714), p->x + re(4810), p->x + re(6858), p->x + re(2762), p->y + re(2680));
fft_8_1_leaf(p->x + re(42), p->x + re(4138), p->x + re(2090), p->x + re(6186), p->x + re(1066), p->x + re(5162), p->x + re(7210), p->x + re(3114), p->y + re(2688));
fft_4_2_leaf(p->x + re(554), p->x + re(4650), p->x + re(2602), p->x + re(6698), p->x + re(7722), p->x + re(3626), p->x + re(1578), p->x + re(5674), p->y + re(2696));
fft_8_1_leaf(p->x + re(298), p->x + re(4394), p->x + re(2346), p->x + re(6442), p->x + re(1322), p->x + re(5418), p->x + re(7466), p->x + re(3370), p->y + re(2704));
fft_8_1_leaf(p->x + re(7978), p->x + re(3882), p->x + re(1834), p->x + re(5930), p->x + re(810), p->x + re(4906), p->x + re(6954), p->x + re(2858), p->y + re(2712));
fft_8_1_leaf(p->x + re(170), p->x + re(4266), p->x + re(2218), p->x + re(6314), p->x + re(1194), p->x + re(5290), p->x + re(7338), p->x + re(3242), p->y + re(2720));
fft_4_2_leaf(p->x + re(682), p->x + re(4778), p->x + re(2730), p->x + re(6826), p->x + re(7850), p->x + re(3754), p->x + re(1706), p->x + re(5802), p->y + re(2728));
fft_8_1_leaf(p->x + re(8106), p->x + re(4010), p->x + re(1962), p->x + re(6058), p->x + re(938), p->x + re(5034), p->x + re(7082), p->x + re(2986), p->y + re(2736));
fft_4_2_leaf(p->x + re(426), p->x + re(4522), p->x + re(2474), p->x + re(6570), p->x + re(7594), p->x + re(3498), p->x + re(1450), p->x + re(5546), p->y + re(2744));
fft_8_1_leaf(p->x + re(8170), p->x + re(4074), p->x + re(2026), p->x + re(6122), p->x + re(1002), p->x + re(5098), p->x + re(7146), p->x + re(3050), p->y + re(2752));
fft_4_2_leaf(p->x + re(490), p->x + re(4586), p->x + re(2538), p->x + re(6634), p->x + re(7658), p->x + re(3562), p->x + re(1514), p->x + re(5610), p->y + re(2760));
fft_8_1_leaf(p->x + re(234), p->x + re(4330), p->x + re(2282), p->x + re(6378), p->x + re(1258), p->x + re(5354), p->x + re(7402), p->x + re(3306), p->y + re(2768));
fft_8_1_leaf(p->x + re(7914), p->x + re(3818), p->x + re(1770), p->x + re(5866), p->x + re(746), p->x + re(4842), p->x + re(6890), p->x + re(2794), p->y + re(2776));
fft_8_1_leaf(p->x + re(106), p->x + re(4202), p->x + re(2154), p->x + re(6250), p->x + re(1130), p->x + re(5226), p->x + re(7274), p->x + re(3178), p->y + re(2784));
fft_4_2_leaf(p->x + re(618), p->x + re(4714), p->x + re(2666), p->x + re(6762), p->x + re(7786), p->x + re(3690), p->x + re(1642), p->x + re(5738), p->y + re(2792));
fft_8_1_leaf(p->x + re(8042), p->x + re(3946), p->x + re(1898), p->x + re(5994), p->x + re(874), p->x + re(4970), p->x + re(7018), p->x + re(2922), p->y + re(2800));
fft_4_2_leaf(p->x + re(362), p->x + re(4458), p->x + re(2410), p->x + re(6506), p->x + re(7530), p->x + re(3434), p->x + re(1386), p->x + re(5482), p->y + re(2808));
fft_8_1_leaf(p->x + re(8186), p->x + re(4090), p->x + re(2042), p->x + re(6138), p->x + re(1018), p->x + re(5114), p->x + re(7162), p->x + re(3066), p->y + re(2816));
fft_4_2_leaf(p->x + re(506), p->x + re(4602), p->x + re(2554), p->x + re(6650), p->x + re(7674), p->x + re(3578), p->x + re(1530), p->x + re(5626), p->y + re(2824));
fft_8_1_leaf(p->x + re(250), p->x + re(4346), p->x + re(2298), p->x + re(6394), p->x + re(1274), p->x + re(5370), p->x + re(7418), p->x + re(3322), p->y + re(2832));
fft_8_1_leaf(p->x + re(7930), p->x + re(3834), p->x + re(1786), p->x + re(5882), p->x + re(762), p->x + re(4858), p->x + re(6906), p->x + re(2810), p->y + re(2840));
fft_8_1_leaf(p->x + re(122), p->x + re(4218), p->x + re(2170), p->x + re(6266), p->x + re(1146), p->x + re(5242), p->x + re(7290), p->x + re(3194), p->y + re(2848));
fft_4_2_leaf(p->x + re(634), p->x + re(4730), p->x + re(2682), p->x + re(6778), p->x + re(7802), p->x + re(3706), p->x + re(1658), p->x + re(5754), p->y + re(2856));
fft_8_1_leaf(p->x + re(8058), p->x + re(3962), p->x + re(1914), p->x + re(6010), p->x + re(890), p->x + re(4986), p->x + re(7034), p->x + re(2938), p->y + re(2864));
fft_4_2_leaf(p->x + re(378), p->x + re(4474), p->x + re(2426), p->x + re(6522), p->x + re(7546), p->x + re(3450), p->x + re(1402), p->x + re(5498), p->y + re(2872));
fft_8_1_leaf(p->x + re(58), p->x + re(4154), p->x + re(2106), p->x + re(6202), p->x + re(1082), p->x + re(5178), p->x + re(7226), p->x + re(3130), p->y + re(2880));
fft_4_2_leaf(p->x + re(570), p->x + re(4666), p->x + re(2618), p->x + re(6714), p->x + re(7738), p->x + re(3642), p->x + re(1594), p->x + re(5690), p->y + re(2888));
fft_8_1_leaf(p->x + re(314), p->x + re(4410), p->x + re(2362), p->x + re(6458), p->x + re(1338), p->x + re(5434), p->x + re(7482), p->x + re(3386), p->y + re(2896));
fft_8_1_leaf(p->x + re(7994), p->x + re(3898), p->x + re(1850), p->x + re(5946), p->x + re(826), p->x + re(4922), p->x + re(6970), p->x + re(2874), p->y + re(2904));
fft_8_1_leaf(p->x + re(8122), p->x + re(4026), p->x + re(1978), p->x + re(6074), p->x + re(954), p->x + re(5050), p->x + re(7098), p->x + re(3002), p->y + re(2912));
fft_4_2_leaf(p->x + re(442), p->x + re(4538), p->x + re(2490), p->x + re(6586), p->x + re(7610), p->x + re(3514), p->x + re(1466), p->x + re(5562), p->y + re(2920));
fft_8_1_leaf(p->x + re(186), p->x + re(4282), p->x + re(2234), p->x + re(6330), p->x + re(1210), p->x + re(5306), p->x + re(7354), p->x + re(3258), p->y + re(2928));
fft_8_1_leaf(p->x + re(7866), p->x + re(3770), p->x + re(1722), p->x + re(5818), p->x + re(698), p->x + re(4794), p->x + re(6842), p->x + re(2746), p->y + re(2936));
fft_8_1_leaf(p->x + re(26), p->x + re(4122), p->x + re(2074), p->x + re(6170), p->x + re(1050), p->x + re(5146), p->x + re(7194), p->x + re(3098), p->y + re(2944));
fft_4_2_leaf(p->x + re(538), p->x + re(4634), p->x + re(2586), p->x + re(6682), p->x + re(7706), p->x + re(3610), p->x + re(1562), p->x + re(5658), p->y + re(2952));
fft_8_1_leaf(p->x + re(282), p->x + re(4378), p->x + re(2330), p->x + re(6426), p->x + re(1306), p->x + re(5402), p->x + re(7450), p->x + re(3354), p->y + re(2960));
fft_8_1_leaf(p->x + re(7962), p->x + re(3866), p->x + re(1818), p->x + re(5914), p->x + re(794), p->x + re(4890), p->x + re(6938), p->x + re(2842), p->y + re(2968));
fft_8_1_leaf(p->x + re(154), p->x + re(4250), p->x + re(2202), p->x + re(6298), p->x + re(1178), p->x + re(5274), p->x + re(7322), p->x + re(3226), p->y + re(2976));
fft_4_2_leaf(p->x + re(666), p->x + re(4762), p->x + re(2714), p->x + re(6810), p->x + re(7834), p->x + re(3738), p->x + re(1690), p->x + re(5786), p->y + re(2984));
fft_8_1_leaf(p->x + re(8090), p->x + re(3994), p->x + re(1946), p->x + re(6042), p->x + re(922), p->x + re(5018), p->x + re(7066), p->x + re(2970), p->y + re(2992));
fft_4_2_leaf(p->x + re(410), p->x + re(4506), p->x + re(2458), p->x + re(6554), p->x + re(7578), p->x + re(3482), p->x + re(1434), p->x + re(5530), p->y + re(3000));
fft_8_1_leaf(p->x + re(8154), p->x + re(4058), p->x + re(2010), p->x + re(6106), p->x + re(986), p->x + re(5082), p->x + re(7130), p->x + re(3034), p->y + re(3008));
fft_4_2_leaf(p->x + re(474), p->x + re(4570), p->x + re(2522), p->x + re(6618), p->x + re(7642), p->x + re(3546), p->x + re(1498), p->x + re(5594), p->y + re(3016));
fft_8_1_leaf(p->x + re(218), p->x + re(4314), p->x + re(2266), p->x + re(6362), p->x + re(1242), p->x + re(5338), p->x + re(7386), p->x + re(3290), p->y + re(3024));
fft_8_1_leaf(p->x + re(7898), p->x + re(3802), p->x + re(1754), p->x + re(5850), p->x + re(730), p->x + re(4826), p->x + re(6874), p->x + re(2778), p->y + re(3032));
fft_8_1_leaf(p->x + re(90), p->x + re(4186), p->x + re(2138), p->x + re(6234), p->x + re(1114), p->x + re(5210), p->x + re(7258), p->x + re(3162), p->y + re(3040));
fft_4_2_leaf(p->x + re(602), p->x + re(4698), p->x + re(2650), p->x + re(6746), p->x + re(7770), p->x + re(3674), p->x + re(1626), p->x + re(5722), p->y + re(3048));
fft_8_1_leaf(p->x + re(8026), p->x + re(3930), p->x + re(1882), p->x + re(5978), p->x + re(858), p->x + re(4954), p->x + re(7002), p->x + re(2906), p->y + re(3056));
fft_4_2_leaf(p->x + re(346), p->x + re(4442), p->x + re(2394), p->x + re(6490), p->x + re(7514), p->x + re(3418), p->x + re(1370), p->x + re(5466), p->y + re(3064));
fft_8_1_leaf(p->x + re(8190), p->x + re(4094), p->x + re(2046), p->x + re(6142), p->x + re(1022), p->x + re(5118), p->x + re(7166), p->x + re(3070), p->y + re(3072));
fft_4_2_leaf(p->x + re(510), p->x + re(4606), p->x + re(2558), p->x + re(6654), p->x + re(7678), p->x + re(3582), p->x + re(1534), p->x + re(5630), p->y + re(3080));
fft_8_1_leaf(p->x + re(254), p->x + re(4350), p->x + re(2302), p->x + re(6398), p->x + re(1278), p->x + re(5374), p->x + re(7422), p->x + re(3326), p->y + re(3088));
fft_8_1_leaf(p->x + re(7934), p->x + re(3838), p->x + re(1790), p->x + re(5886), p->x + re(766), p->x + re(4862), p->x + re(6910), p->x + re(2814), p->y + re(3096));
fft_8_1_leaf(p->x + re(126), p->x + re(4222), p->x + re(2174), p->x + re(6270), p->x + re(1150), p->x + re(5246), p->x + re(7294), p->x + re(3198), p->y + re(3104));
fft_4_2_leaf(p->x + re(638), p->x + re(4734), p->x + re(2686), p->x + re(6782), p->x + re(7806), p->x + re(3710), p->x + re(1662), p->x + re(5758), p->y + re(3112));
fft_8_1_leaf(p->x + re(8062), p->x + re(3966), p->x + re(1918), p->x + re(6014), p->x + re(894), p->x + re(4990), p->x + re(7038), p->x + re(2942), p->y + re(3120));
fft_4_2_leaf(p->x + re(382), p->x + re(4478), p->x + re(2430), p->x + re(6526), p->x + re(7550), p->x + re(3454), p->x + re(1406), p->x + re(5502), p->y + re(3128));
fft_8_1_leaf(p->x + re(62), p->x + re(4158), p->x + re(2110), p->x + re(6206), p->x + re(1086), p->x + re(5182), p->x + re(7230), p->x + re(3134), p->y + re(3136));
fft_4_2_leaf(p->x + re(574), p->x + re(4670), p->x + re(2622), p->x + re(6718), p->x + re(7742), p->x + re(3646), p->x + re(1598), p->x + re(5694), p->y + re(3144));
fft_8_1_leaf(p->x + re(318), p->x + re(4414), p->x + re(2366), p->x + re(6462), p->x + re(1342), p->x + re(5438), p->x + re(7486), p->x + re(3390), p->y + re(3152));
fft_8_1_leaf(p->x + re(7998), p->x + re(3902), p->x + re(1854), p->x + re(5950), p->x + re(830), p->x + re(4926), p->x + re(6974), p->x + re(2878), p->y + re(3160));
fft_8_1_leaf(p->x + re(8126), p->x + re(4030), p->x + re(1982), p->x + re(6078), p->x + re(958), p->x + re(5054), p->x + re(7102), p->x + re(3006), p->y + re(3168));
fft_4_2_leaf(p->x + re(446), p->x + re(4542), p->x + re(2494), p->x + re(6590), p->x + re(7614), p->x + re(3518), p->x + re(1470), p->x + re(5566), p->y + re(3176));
fft_8_1_leaf(p->x + re(190), p->x + re(4286), p->x + re(2238), p->x + re(6334), p->x + re(1214), p->x + re(5310), p->x + re(7358), p->x + re(3262), p->y + re(3184));
fft_8_1_leaf(p->x + re(7870), p->x + re(3774), p->x + re(1726), p->x + re(5822), p->x + re(702), p->x + re(4798), p->x + re(6846), p->x + re(2750), p->y + re(3192));
fft_8_1_leaf(p->x + re(30), p->x + re(4126), p->x + re(2078), p->x + re(6174), p->x + re(1054), p->x + re(5150), p->x + re(7198), p->x + re(3102), p->y + re(3200));
fft_4_2_leaf(p->x + re(542), p->x + re(4638), p->x + re(2590), p->x + re(6686), p->x + re(7710), p->x + re(3614), p->x + re(1566), p->x + re(5662), p->y + re(3208));
fft_8_1_leaf(p->x + re(286), p->x + re(4382), p->x + re(2334), p->x + re(6430), p->x + re(1310), p->x + re(5406), p->x + re(7454), p->x + re(3358), p->y + re(3216));
fft_8_1_leaf(p->x + re(7966), p->x + re(3870), p->x + re(1822), p->x + re(5918), p->x + re(798), p->x + re(4894), p->x + re(6942), p->x + re(2846), p->y + re(3224));
fft_8_1_leaf(p->x + re(158), p->x + re(4254), p->x + re(2206), p->x + re(6302), p->x + re(1182), p->x + re(5278), p->x + re(7326), p->x + re(3230), p->y + re(3232));
fft_4_2_leaf(p->x + re(670), p->x + re(4766), p->x + re(2718), p->x + re(6814), p->x + re(7838), p->x + re(3742), p->x + re(1694), p->x + re(5790), p->y + re(3240));
fft_8_1_leaf(p->x + re(8094), p->x + re(3998), p->x + re(1950), p->x + re(6046), p->x + re(926), p->x + re(5022), p->x + re(7070), p->x + re(2974), p->y + re(3248));
fft_4_2_leaf(p->x + re(414), p->x + re(4510), p->x + re(2462), p->x + re(6558), p->x + re(7582), p->x + re(3486), p->x + re(1438), p->x + re(5534), p->y + re(3256));
fft_8_1_leaf(p->x + re(8158), p->x + re(4062), p->x + re(2014), p->x + re(6110), p->x + re(990), p->x + re(5086), p->x + re(7134), p->x + re(3038), p->y + re(3264));
fft_4_2_leaf(p->x + re(478), p->x + re(4574), p->x + re(2526), p->x + re(6622), p->x + re(7646), p->x + re(3550), p->x + re(1502), p->x + re(5598), p->y + re(3272));
fft_8_1_leaf(p->x + re(222), p->x + re(4318), p->x + re(2270), p->x + re(6366), p->x + re(1246), p->x + re(5342), p->x + re(7390), p->x + re(3294), p->y + re(3280));
fft_8_1_leaf(p->x + re(7902), p->x + re(3806), p->x + re(1758), p->x + re(5854), p->x + re(734), p->x + re(4830), p->x + re(6878), p->x + re(2782), p->y + re(3288));
fft_8_1_leaf(p->x + re(94), p->x + re(4190), p->x + re(2142), p->x + re(6238), p->x + re(1118), p->x + re(5214), p->x + re(7262), p->x + re(3166), p->y + re(3296));
fft_4_2_leaf(p->x + re(606), p->x + re(4702), p->x + re(2654), p->x + re(6750), p->x + re(7774), p->x + re(3678), p->x + re(1630), p->x + re(5726), p->y + re(3304));
fft_8_1_leaf(p->x + re(8030), p->x + re(3934), p->x + re(1886), p->x + re(5982), p->x + re(862), p->x + re(4958), p->x + re(7006), p->x + re(2910), p->y + re(3312));
fft_4_2_leaf(p->x + re(350), p->x + re(4446), p->x + re(2398), p->x + re(6494), p->x + re(7518), p->x + re(3422), p->x + re(1374), p->x + re(5470), p->y + re(3320));
fft_8_1_leaf(p->x + re(14), p->x + re(4110), p->x + re(2062), p->x + re(6158), p->x + re(1038), p->x + re(5134), p->x + re(7182), p->x + re(3086), p->y + re(3328));
fft_4_2_leaf(p->x + re(526), p->x + re(4622), p->x + re(2574), p->x + re(6670), p->x + re(7694), p->x + re(3598), p->x + re(1550), p->x + re(5646), p->y + re(3336));
fft_8_1_leaf(p->x + re(270), p->x + re(4366), p->x + re(2318), p->x + re(6414), p->x + re(1294), p->x + re(5390), p->x + re(7438), p->x + re(3342), p->y + re(3344));
fft_8_1_leaf(p->x + re(7950), p->x + re(3854), p->x + re(1806), p->x + re(5902), p->x + re(782), p->x + re(4878), p->x + re(6926), p->x + re(2830), p->y + re(3352));
fft_8_1_leaf(p->x + re(142), p->x + re(4238), p->x + re(2190), p->x + re(6286), p->x + re(1166), p->x + re(5262), p->x + re(7310), p->x + re(3214), p->y + re(3360));
fft_4_2_leaf(p->x + re(654), p->x + re(4750), p->x + re(2702), p->x + re(6798), p->x + re(7822), p->x + re(3726), p->x + re(1678), p->x + re(5774), p->y + re(3368));
fft_8_1_leaf(p->x + re(8078), p->x + re(3982), p->x + re(1934), p->x + re(6030), p->x + re(910), p->x + re(5006), p->x + re(7054), p->x + re(2958), p->y + re(3376));
fft_4_2_leaf(p->x + re(398), p->x + re(4494), p->x + re(2446), p->x + re(6542), p->x + re(7566), p->x + re(3470), p->x + re(1422), p->x + re(5518), p->y + re(3384));
fft_8_1_leaf(p->x + re(78), p->x + re(4174), p->x + re(2126), p->x + re(6222), p->x + re(1102), p->x + re(5198), p->x + re(7246), p->x + re(3150), p->y + re(3392));
fft_4_2_leaf(p->x + re(590), p->x + re(4686), p->x + re(2638), p->x + re(6734), p->x + re(7758), p->x + re(3662), p->x + re(1614), p->x + re(5710), p->y + re(3400));
fft_8_1_leaf(p->x + re(334), p->x + re(4430), p->x + re(2382), p->x + re(6478), p->x + re(1358), p->x + re(5454), p->x + re(7502), p->x + re(3406), p->y + re(3408));
fft_8_1_leaf(p->x + re(8014), p->x + re(3918), p->x + re(1870), p->x + re(5966), p->x + re(846), p->x + re(4942), p->x + re(6990), p->x + re(2894), p->y + re(3416));
fft_8_1_leaf(p->x + re(8142), p->x + re(4046), p->x + re(1998), p->x + re(6094), p->x + re(974), p->x + re(5070), p->x + re(7118), p->x + re(3022), p->y + re(3424));
fft_4_2_leaf(p->x + re(462), p->x + re(4558), p->x + re(2510), p->x + re(6606), p->x + re(7630), p->x + re(3534), p->x + re(1486), p->x + re(5582), p->y + re(3432));
fft_8_1_leaf(p->x + re(206), p->x + re(4302), p->x + re(2254), p->x + re(6350), p->x + re(1230), p->x + re(5326), p->x + re(7374), p->x + re(3278), p->y + re(3440));
fft_8_1_leaf(p->x + re(7886), p->x + re(3790), p->x + re(1742), p->x + re(5838), p->x + re(718), p->x + re(4814), p->x + re(6862), p->x + re(2766), p->y + re(3448));
fft_8_1_leaf(p->x + re(8174), p->x + re(4078), p->x + re(2030), p->x + re(6126), p->x + re(1006), p->x + re(5102), p->x + re(7150), p->x + re(3054), p->y + re(3456));
fft_4_2_leaf(p->x + re(494), p->x + re(4590), p->x + re(2542), p->x + re(6638), p->x + re(7662), p->x + re(3566), p->x + re(1518), p->x + re(5614), p->y + re(3464));
fft_8_1_leaf(p->x + re(238), p->x + re(4334), p->x + re(2286), p->x + re(6382), p->x + re(1262), p->x + re(5358), p->x + re(7406), p->x + re(3310), p->y + re(3472));
fft_8_1_leaf(p->x + re(7918), p->x + re(3822), p->x + re(1774), p->x + re(5870), p->x + re(750), p->x + re(4846), p->x + re(6894), p->x + re(2798), p->y + re(3480));
fft_8_1_leaf(p->x + re(110), p->x + re(4206), p->x + re(2158), p->x + re(6254), p->x + re(1134), p->x + re(5230), p->x + re(7278), p->x + re(3182), p->y + re(3488));
fft_4_2_leaf(p->x + re(622), p->x + re(4718), p->x + re(2670), p->x + re(6766), p->x + re(7790), p->x + re(3694), p->x + re(1646), p->x + re(5742), p->y + re(3496));
fft_8_1_leaf(p->x + re(8046), p->x + re(3950), p->x + re(1902), p->x + re(5998), p->x + re(878), p->x + re(4974), p->x + re(7022), p->x + re(2926), p->y + re(3504));
fft_4_2_leaf(p->x + re(366), p->x + re(4462), p->x + re(2414), p->x + re(6510), p->x + re(7534), p->x + re(3438), p->x + re(1390), p->x + re(5486), p->y + re(3512));
fft_8_1_leaf(p->x + re(46), p->x + re(4142), p->x + re(2094), p->x + re(6190), p->x + re(1070), p->x + re(5166), p->x + re(7214), p->x + re(3118), p->y + re(3520));
fft_4_2_leaf(p->x + re(558), p->x + re(4654), p->x + re(2606), p->x + re(6702), p->x + re(7726), p->x + re(3630), p->x + re(1582), p->x + re(5678), p->y + re(3528));
fft_8_1_leaf(p->x + re(302), p->x + re(4398), p->x + re(2350), p->x + re(6446), p->x + re(1326), p->x + re(5422), p->x + re(7470), p->x + re(3374), p->y + re(3536));
fft_8_1_leaf(p->x + re(7982), p->x + re(3886), p->x + re(1838), p->x + re(5934), p->x + re(814), p->x + re(4910), p->x + re(6958), p->x + re(2862), p->y + re(3544));
fft_8_1_leaf(p->x + re(8110), p->x + re(4014), p->x + re(1966), p->x + re(6062), p->x + re(942), p->x + re(5038), p->x + re(7086), p->x + re(2990), p->y + re(3552));
fft_4_2_leaf(p->x + re(430), p->x + re(4526), p->x + re(2478), p->x + re(6574), p->x + re(7598), p->x + re(3502), p->x + re(1454), p->x + re(5550), p->y + re(3560));
fft_8_1_leaf(p->x + re(174), p->x + re(4270), p->x + re(2222), p->x + re(6318), p->x + re(1198), p->x + re(5294), p->x + re(7342), p->x + re(3246), p->y + re(3568));
fft_8_1_leaf(p->x + re(7854), p->x + re(3758), p->x + re(1710), p->x + re(5806), p->x + re(686), p->x + re(4782), p->x + re(6830), p->x + re(2734), p->y + re(3576));
fft_8_1_leaf(p->x + re(6), p->x + re(4102), p->x + re(2054), p->x + re(6150), p->x + re(1030), p->x + re(5126), p->x + re(7174), p->x + re(3078), p->y + re(3584));
fft_4_2_leaf(p->x + re(518), p->x + re(4614), p->x + re(2566), p->x + re(6662), p->x + re(7686), p->x + re(3590), p->x + re(1542), p->x + re(5638), p->y + re(3592));
fft_8_1_leaf(p->x + re(262), p->x + re(4358), p->x + re(2310), p->x + re(6406), p->x + re(1286), p->x + re(5382), p->x + re(7430), p->x + re(3334), p->y + re(3600));
fft_8_1_leaf(p->x + re(7942), p->x + re(3846), p->x + re(1798), p->x + re(5894), p->x + re(774), p->x + re(4870), p->x + re(6918), p->x + re(2822), p->y + re(3608));
fft_8_1_leaf(p->x + re(134), p->x + re(4230), p->x + re(2182), p->x + re(6278), p->x + re(1158), p->x + re(5254), p->x + re(7302), p->x + re(3206), p->y + re(3616));
fft_4_2_leaf(p->x + re(646), p->x + re(4742), p->x + re(2694), p->x + re(6790), p->x + re(7814), p->x + re(3718), p->x + re(1670), p->x + re(5766), p->y + re(3624));
fft_8_1_leaf(p->x + re(8070), p->x + re(3974), p->x + re(1926), p->x + re(6022), p->x + re(902), p->x + re(4998), p->x + re(7046), p->x + re(2950), p->y + re(3632));
fft_4_2_leaf(p->x + re(390), p->x + re(4486), p->x + re(2438), p->x + re(6534), p->x + re(7558), p->x + re(3462), p->x + re(1414), p->x + re(5510), p->y + re(3640));
fft_8_1_leaf(p->x + re(70), p->x + re(4166), p->x + re(2118), p->x + re(6214), p->x + re(1094), p->x + re(5190), p->x + re(7238), p->x + re(3142), p->y + re(3648));
fft_4_2_leaf(p->x + re(582), p->x + re(4678), p->x + re(2630), p->x + re(6726), p->x + re(7750), p->x + re(3654), p->x + re(1606), p->x + re(5702), p->y + re(3656));
fft_8_1_leaf(p->x + re(326), p->x + re(4422), p->x + re(2374), p->x + re(6470), p->x + re(1350), p->x + re(5446), p->x + re(7494), p->x + re(3398), p->y + re(3664));
fft_8_1_leaf(p->x + re(8006), p->x + re(3910), p->x + re(1862), p->x + re(5958), p->x + re(838), p->x + re(4934), p->x + re(6982), p->x + re(2886), p->y + re(3672));
fft_8_1_leaf(p->x + re(8134), p->x + re(4038), p->x + re(1990), p->x + re(6086), p->x + re(966), p->x + re(5062), p->x + re(7110), p->x + re(3014), p->y + re(3680));
fft_4_2_leaf(p->x + re(454), p->x + re(4550), p->x + re(2502), p->x + re(6598), p->x + re(7622), p->x + re(3526), p->x + re(1478), p->x + re(5574), p->y + re(3688));
fft_8_1_leaf(p->x + re(198), p->x + re(4294), p->x + re(2246), p->x + re(6342), p->x + re(1222), p->x + re(5318), p->x + re(7366), p->x + re(3270), p->y + re(3696));
fft_8_1_leaf(p->x + re(7878), p->x + re(3782), p->x + re(1734), p->x + re(5830), p->x + re(710), p->x + re(4806), p->x + re(6854), p->x + re(2758), p->y + re(3704));
fft_8_1_leaf(p->x + re(38), p->x + re(4134), p->x + re(2086), p->x + re(6182), p->x + re(1062), p->x + re(5158), p->x + re(7206), p->x + re(3110), p->y + re(3712));
fft_4_2_leaf(p->x + re(550), p->x + re(4646), p->x + re(2598), p->x + re(6694), p->x + re(7718), p->x + re(3622), p->x + re(1574), p->x + re(5670), p->y + re(3720));
fft_8_1_leaf(p->x + re(294), p->x + re(4390), p->x + re(2342), p->x + re(6438), p->x + re(1318), p->x + re(5414), p->x + re(7462), p->x + re(3366), p->y + re(3728));
fft_8_1_leaf(p->x + re(7974), p->x + re(3878), p->x + re(1830), p->x + re(5926), p->x + re(806), p->x + re(4902), p->x + re(6950), p->x + re(2854), p->y + re(3736));
fft_8_1_leaf(p->x + re(166), p->x + re(4262), p->x + re(2214), p->x + re(6310), p->x + re(1190), p->x + re(5286), p->x + re(7334), p->x + re(3238), p->y + re(3744));
fft_4_2_leaf(p->x + re(678), p->x + re(4774), p->x + re(2726), p->x + re(6822), p->x + re(7846), p->x + re(3750), p->x + re(1702), p->x + re(5798), p->y + re(3752));
fft_8_1_leaf(p->x + re(8102), p->x + re(4006), p->x + re(1958), p->x + re(6054), p->x + re(934), p->x + re(5030), p->x + re(7078), p->x + re(2982), p->y + re(3760));
fft_4_2_leaf(p->x + re(422), p->x + re(4518), p->x + re(2470), p->x + re(6566), p->x + re(7590), p->x + re(3494), p->x + re(1446), p->x + re(5542), p->y + re(3768));
fft_8_1_leaf(p->x + re(8166), p->x + re(4070), p->x + re(2022), p->x + re(6118), p->x + re(998), p->x + re(5094), p->x + re(7142), p->x + re(3046), p->y + re(3776));
fft_4_2_leaf(p->x + re(486), p->x + re(4582), p->x + re(2534), p->x + re(6630), p->x + re(7654), p->x + re(3558), p->x + re(1510), p->x + re(5606), p->y + re(3784));
fft_8_1_leaf(p->x + re(230), p->x + re(4326), p->x + re(2278), p->x + re(6374), p->x + re(1254), p->x + re(5350), p->x + re(7398), p->x + re(3302), p->y + re(3792));
fft_8_1_leaf(p->x + re(7910), p->x + re(3814), p->x + re(1766), p->x + re(5862), p->x + re(742), p->x + re(4838), p->x + re(6886), p->x + re(2790), p->y + re(3800));
fft_8_1_leaf(p->x + re(102), p->x + re(4198), p->x + re(2150), p->x + re(6246), p->x + re(1126), p->x + re(5222), p->x + re(7270), p->x + re(3174), p->y + re(3808));
fft_4_2_leaf(p->x + re(614), p->x + re(4710), p->x + re(2662), p->x + re(6758), p->x + re(7782), p->x + re(3686), p->x + re(1638), p->x + re(5734), p->y + re(3816));
fft_8_1_leaf(p->x + re(8038), p->x + re(3942), p->x + re(1894), p->x + re(5990), p->x + re(870), p->x + re(4966), p->x + re(7014), p->x + re(2918), p->y + re(3824));
fft_4_2_leaf(p->x + re(358), p->x + re(4454), p->x + re(2406), p->x + re(6502), p->x + re(7526), p->x + re(3430), p->x + re(1382), p->x + re(5478), p->y + re(3832));
fft_8_1_leaf(p->x + re(8182), p->x + re(4086), p->x + re(2038), p->x + re(6134), p->x + re(1014), p->x + re(5110), p->x + re(7158), p->x + re(3062), p->y + re(3840));
fft_4_2_leaf(p->x + re(502), p->x + re(4598), p->x + re(2550), p->x + re(6646), p->x + re(7670), p->x + re(3574), p->x + re(1526), p->x + re(5622), p->y + re(3848));
fft_8_1_leaf(p->x + re(246), p->x + re(4342), p->x + re(2294), p->x + re(6390), p->x + re(1270), p->x + re(5366), p->x + re(7414), p->x + re(3318), p->y + re(3856));
fft_8_1_leaf(p->x + re(7926), p->x + re(3830), p->x + re(1782), p->x + re(5878), p->x + re(758), p->x + re(4854), p->x + re(6902), p->x + re(2806), p->y + re(3864));
fft_8_1_leaf(p->x + re(118), p->x + re(4214), p->x + re(2166), p->x + re(6262), p->x + re(1142), p->x + re(5238), p->x + re(7286), p->x + re(3190), p->y + re(3872));
fft_4_2_leaf(p->x + re(630), p->x + re(4726), p->x + re(2678), p->x + re(6774), p->x + re(7798), p->x + re(3702), p->x + re(1654), p->x + re(5750), p->y + re(3880));
fft_8_1_leaf(p->x + re(8054), p->x + re(3958), p->x + re(1910), p->x + re(6006), p->x + re(886), p->x + re(4982), p->x + re(7030), p->x + re(2934), p->y + re(3888));
fft_4_2_leaf(p->x + re(374), p->x + re(4470), p->x + re(2422), p->x + re(6518), p->x + re(7542), p->x + re(3446), p->x + re(1398), p->x + re(5494), p->y + re(3896));
fft_8_1_leaf(p->x + re(54), p->x + re(4150), p->x + re(2102), p->x + re(6198), p->x + re(1078), p->x + re(5174), p->x + re(7222), p->x + re(3126), p->y + re(3904));
fft_4_2_leaf(p->x + re(566), p->x + re(4662), p->x + re(2614), p->x + re(6710), p->x + re(7734), p->x + re(3638), p->x + re(1590), p->x + re(5686), p->y + re(3912));
fft_8_1_leaf(p->x + re(310), p->x + re(4406), p->x + re(2358), p->x + re(6454), p->x + re(1334), p->x + re(5430), p->x + re(7478), p->x + re(3382), p->y + re(3920));
fft_8_1_leaf(p->x + re(7990), p->x + re(3894), p->x + re(1846), p->x + re(5942), p->x + re(822), p->x + re(4918), p->x + re(6966), p->x + re(2870), p->y + re(3928));
fft_8_1_leaf(p->x + re(8118), p->x + re(4022), p->x + re(1974), p->x + re(6070), p->x + re(950), p->x + re(5046), p->x + re(7094), p->x + re(2998), p->y + re(3936));
fft_4_2_leaf(p->x + re(438), p->x + re(4534), p->x + re(2486), p->x + re(6582), p->x + re(7606), p->x + re(3510), p->x + re(1462), p->x + re(5558), p->y + re(3944));
fft_8_1_leaf(p->x + re(182), p->x + re(4278), p->x + re(2230), p->x + re(6326), p->x + re(1206), p->x + re(5302), p->x + re(7350), p->x + re(3254), p->y + re(3952));
fft_8_1_leaf(p->x + re(7862), p->x + re(3766), p->x + re(1718), p->x + re(5814), p->x + re(694), p->x + re(4790), p->x + re(6838), p->x + re(2742), p->y + re(3960));
fft_8_1_leaf(p->x + re(22), p->x + re(4118), p->x + re(2070), p->x + re(6166), p->x + re(1046), p->x + re(5142), p->x + re(7190), p->x + re(3094), p->y + re(3968));
fft_4_2_leaf(p->x + re(534), p->x + re(4630), p->x + re(2582), p->x + re(6678), p->x + re(7702), p->x + re(3606), p->x + re(1558), p->x + re(5654), p->y + re(3976));
fft_8_1_leaf(p->x + re(278), p->x + re(4374), p->x + re(2326), p->x + re(6422), p->x + re(1302), p->x + re(5398), p->x + re(7446), p->x + re(3350), p->y + re(3984));
fft_8_1_leaf(p->x + re(7958), p->x + re(3862), p->x + re(1814), p->x + re(5910), p->x + re(790), p->x + re(4886), p->x + re(6934), p->x + re(2838), p->y + re(3992));
fft_8_1_leaf(p->x + re(150), p->x + re(4246), p->x + re(2198), p->x + re(6294), p->x + re(1174), p->x + re(5270), p->x + re(7318), p->x + re(3222), p->y + re(4000));
fft_4_2_leaf(p->x + re(662), p->x + re(4758), p->x + re(2710), p->x + re(6806), p->x + re(7830), p->x + re(3734), p->x + re(1686), p->x + re(5782), p->y + re(4008));
fft_8_1_leaf(p->x + re(8086), p->x + re(3990), p->x + re(1942), p->x + re(6038), p->x + re(918), p->x + re(5014), p->x + re(7062), p->x + re(2966), p->y + re(4016));
fft_4_2_leaf(p->x + re(406), p->x + re(4502), p->x + re(2454), p->x + re(6550), p->x + re(7574), p->x + re(3478), p->x + re(1430), p->x + re(5526), p->y + re(4024));
fft_8_1_leaf(p->x + re(8150), p->x + re(4054), p->x + re(2006), p->x + re(6102), p->x + re(982), p->x + re(5078), p->x + re(7126), p->x + re(3030), p->y + re(4032));
fft_4_2_leaf(p->x + re(470), p->x + re(4566), p->x + re(2518), p->x + re(6614), p->x + re(7638), p->x + re(3542), p->x + re(1494), p->x + re(5590), p->y + re(4040));
fft_8_1_leaf(p->x + re(214), p->x + re(4310), p->x + re(2262), p->x + re(6358), p->x + re(1238), p->x + re(5334), p->x + re(7382), p->x + re(3286), p->y + re(4048));
fft_8_1_leaf(p->x + re(7894), p->x + re(3798), p->x + re(1750), p->x + re(5846), p->x + re(726), p->x + re(4822), p->x + re(6870), p->x + re(2774), p->y + re(4056));
fft_8_1_leaf(p->x + re(86), p->x + re(4182), p->x + re(2134), p->x + re(6230), p->x + re(1110), p->x + re(5206), p->x + re(7254), p->x + re(3158), p->y + re(4064));
fft_4_2_leaf(p->x + re(598), p->x + re(4694), p->x + re(2646), p->x + re(6742), p->x + re(7766), p->x + re(3670), p->x + re(1622), p->x + re(5718), p->y + re(4072));
fft_8_1_leaf(p->x + re(8022), p->x + re(3926), p->x + re(1878), p->x + re(5974), p->x + re(854), p->x + re(4950), p->x + re(6998), p->x + re(2902), p->y + re(4080));
fft_4_2_leaf(p->x + re(342), p->x + re(4438), p->x + re(2390), p->x + re(6486), p->x + re(7510), p->x + re(3414), p->x + re(1366), p->x + re(5462), p->y + re(4088));
fft_8_1_leaf(p->x + re(1), p->x + re(4097), p->x + re(2049), p->x + re(6145), p->x + re(1025), p->x + re(5121), p->x + re(7169), p->x + re(3073), p->y + re(4096));
fft_4_2_leaf(p->x + re(513), p->x + re(4609), p->x + re(2561), p->x + re(6657), p->x + re(7681), p->x + re(3585), p->x + re(1537), p->x + re(5633), p->y + re(4104));
fft_8_1_leaf(p->x + re(257), p->x + re(4353), p->x + re(2305), p->x + re(6401), p->x + re(1281), p->x + re(5377), p->x + re(7425), p->x + re(3329), p->y + re(4112));
fft_8_1_leaf(p->x + re(7937), p->x + re(3841), p->x + re(1793), p->x + re(5889), p->x + re(769), p->x + re(4865), p->x + re(6913), p->x + re(2817), p->y + re(4120));
fft_8_1_leaf(p->x + re(129), p->x + re(4225), p->x + re(2177), p->x + re(6273), p->x + re(1153), p->x + re(5249), p->x + re(7297), p->x + re(3201), p->y + re(4128));
fft_4_2_leaf(p->x + re(641), p->x + re(4737), p->x + re(2689), p->x + re(6785), p->x + re(7809), p->x + re(3713), p->x + re(1665), p->x + re(5761), p->y + re(4136));
fft_8_1_leaf(p->x + re(8065), p->x + re(3969), p->x + re(1921), p->x + re(6017), p->x + re(897), p->x + re(4993), p->x + re(7041), p->x + re(2945), p->y + re(4144));
fft_4_2_leaf(p->x + re(385), p->x + re(4481), p->x + re(2433), p->x + re(6529), p->x + re(7553), p->x + re(3457), p->x + re(1409), p->x + re(5505), p->y + re(4152));
fft_8_1_leaf(p->x + re(65), p->x + re(4161), p->x + re(2113), p->x + re(6209), p->x + re(1089), p->x + re(5185), p->x + re(7233), p->x + re(3137), p->y + re(4160));
fft_4_2_leaf(p->x + re(577), p->x + re(4673), p->x + re(2625), p->x + re(6721), p->x + re(7745), p->x + re(3649), p->x + re(1601), p->x + re(5697), p->y + re(4168));
fft_8_1_leaf(p->x + re(321), p->x + re(4417), p->x + re(2369), p->x + re(6465), p->x + re(1345), p->x + re(5441), p->x + re(7489), p->x + re(3393), p->y + re(4176));
fft_8_1_leaf(p->x + re(8001), p->x + re(3905), p->x + re(1857), p->x + re(5953), p->x + re(833), p->x + re(4929), p->x + re(6977), p->x + re(2881), p->y + re(4184));
fft_8_1_leaf(p->x + re(8129), p->x + re(4033), p->x + re(1985), p->x + re(6081), p->x + re(961), p->x + re(5057), p->x + re(7105), p->x + re(3009), p->y + re(4192));
fft_4_2_leaf(p->x + re(449), p->x + re(4545), p->x + re(2497), p->x + re(6593), p->x + re(7617), p->x + re(3521), p->x + re(1473), p->x + re(5569), p->y + re(4200));
fft_8_1_leaf(p->x + re(193), p->x + re(4289), p->x + re(2241), p->x + re(6337), p->x + re(1217), p->x + re(5313), p->x + re(7361), p->x + re(3265), p->y + re(4208));
fft_8_1_leaf(p->x + re(7873), p->x + re(3777), p->x + re(1729), p->x + re(5825), p->x + re(705), p->x + re(4801), p->x + re(6849), p->x + re(2753), p->y + re(4216));
fft_8_1_leaf(p->x + re(33), p->x + re(4129), p->x + re(2081), p->x + re(6177), p->x + re(1057), p->x + re(5153), p->x + re(7201), p->x + re(3105), p->y + re(4224));
fft_4_2_leaf(p->x + re(545), p->x + re(4641), p->x + re(2593), p->x + re(6689), p->x + re(7713), p->x + re(3617), p->x + re(1569), p->x + re(5665), p->y + re(4232));
fft_8_1_leaf(p->x + re(289), p->x + re(4385), p->x + re(2337), p->x + re(6433), p->x + re(1313), p->x + re(5409), p->x + re(7457), p->x + re(3361), p->y + re(4240));
fft_8_1_leaf(p->x + re(7969), p->x + re(3873), p->x + re(1825), p->x + re(5921), p->x + re(801), p->x + re(4897), p->x + re(6945), p->x + re(2849), p->y + re(4248));
fft_8_1_leaf(p->x + re(161), p->x + re(4257), p->x + re(2209), p->x + re(6305), p->x + re(1185), p->x + re(5281), p->x + re(7329), p->x + re(3233), p->y + re(4256));
fft_4_2_leaf(p->x + re(673), p->x + re(4769), p->x + re(2721), p->x + re(6817), p->x + re(7841), p->x + re(3745), p->x + re(1697), p->x + re(5793), p->y + re(4264));
fft_8_1_leaf(p->x + re(8097), p->x + re(4001), p->x + re(1953), p->x + re(6049), p->x + re(929), p->x + re(5025), p->x + re(7073), p->x + re(2977), p->y + re(4272));
fft_4_2_leaf(p->x + re(417), p->x + re(4513), p->x + re(2465), p->x + re(6561), p->x + re(7585), p->x + re(3489), p->x + re(1441), p->x + re(5537), p->y + re(4280));
fft_8_1_leaf(p->x + re(8161), p->x + re(4065), p->x + re(2017), p->x + re(6113), p->x + re(993), p->x + re(5089), p->x + re(7137), p->x + re(3041), p->y + re(4288));
fft_4_2_leaf(p->x + re(481), p->x + re(4577), p->x + re(2529), p->x + re(6625), p->x + re(7649), p->x + re(3553), p->x + re(1505), p->x + re(5601), p->y + re(4296));
fft_8_1_leaf(p->x + re(225), p->x + re(4321), p->x + re(2273), p->x + re(6369), p->x + re(1249), p->x + re(5345), p->x + re(7393), p->x + re(3297), p->y + re(4304));
fft_8_1_leaf(p->x + re(7905), p->x + re(3809), p->x + re(1761), p->x + re(5857), p->x + re(737), p->x + re(4833), p->x + re(6881), p->x + re(2785), p->y + re(4312));
fft_8_1_leaf(p->x + re(97), p->x + re(4193), p->x + re(2145), p->x + re(6241), p->x + re(1121), p->x + re(5217), p->x + re(7265), p->x + re(3169), p->y + re(4320));
fft_4_2_leaf(p->x + re(609), p->x + re(4705), p->x + re(2657), p->x + re(6753), p->x + re(7777), p->x + re(3681), p->x + re(1633), p->x + re(5729), p->y + re(4328));
fft_8_1_leaf(p->x + re(8033), p->x + re(3937), p->x + re(1889), p->x + re(5985), p->x + re(865), p->x + re(4961), p->x + re(7009), p->x + re(2913), p->y + re(4336));
fft_4_2_leaf(p->x + re(353), p->x + re(4449), p->x + re(2401), p->x + re(6497), p->x + re(7521), p->x + re(3425), p->x + re(1377), p->x + re(5473), p->y + re(4344));
fft_8_1_leaf(p->x + re(17), p->x + re(4113), p->x + re(2065), p->x + re(6161), p->x + re(1041), p->x + re(5137), p->x + re(7185), p->x + re(3089), p->y + re(4352));
fft_4_2_leaf(p->x + re(529), p->x + re(4625), p->x + re(2577), p->x + re(6673), p->x + re(7697), p->x + re(3601), p->x + re(1553), p->x + re(5649), p->y + re(4360));
fft_8_1_leaf(p->x + re(273), p->x + re(4369), p->x + re(2321), p->x + re(6417), p->x + re(1297), p->x + re(5393), p->x + re(7441), p->x + re(3345), p->y + re(4368));
fft_8_1_leaf(p->x + re(7953), p->x + re(3857), p->x + re(1809), p->x + re(5905), p->x + re(785), p->x + re(4881), p->x + re(6929), p->x + re(2833), p->y + re(4376));
fft_8_1_leaf(p->x + re(145), p->x + re(4241), p->x + re(2193), p->x + re(6289), p->x + re(1169), p->x + re(5265), p->x + re(7313), p->x + re(3217), p->y + re(4384));
fft_4_2_leaf(p->x + re(657), p->x + re(4753), p->x + re(2705), p->x + re(6801), p->x + re(7825), p->x + re(3729), p->x + re(1681), p->x + re(5777), p->y + re(4392));
fft_8_1_leaf(p->x + re(8081), p->x + re(3985), p->x + re(1937), p->x + re(6033), p->x + re(913), p->x + re(5009), p->x + re(7057), p->x + re(2961), p->y + re(4400));
fft_4_2_leaf(p->x + re(401), p->x + re(4497), p->x + re(2449), p->x + re(6545), p->x + re(7569), p->x + re(3473), p->x + re(1425), p->x + re(5521), p->y + re(4408));
fft_8_1_leaf(p->x + re(81), p->x + re(4177), p->x + re(2129), p->x + re(6225), p->x + re(1105), p->x + re(5201), p->x + re(7249), p->x + re(3153), p->y + re(4416));
fft_4_2_leaf(p->x + re(593), p->x + re(4689), p->x + re(2641), p->x + re(6737), p->x + re(7761), p->x + re(3665), p->x + re(1617), p->x + re(5713), p->y + re(4424));
fft_8_1_leaf(p->x + re(337), p->x + re(4433), p->x + re(2385), p->x + re(6481), p->x + re(1361), p->x + re(5457), p->x + re(7505), p->x + re(3409), p->y + re(4432));
fft_8_1_leaf(p->x + re(8017), p->x + re(3921), p->x + re(1873), p->x + re(5969), p->x + re(849), p->x + re(4945), p->x + re(6993), p->x + re(2897), p->y + re(4440));
fft_8_1_leaf(p->x + re(8145), p->x + re(4049), p->x + re(2001), p->x + re(6097), p->x + re(977), p->x + re(5073), p->x + re(7121), p->x + re(3025), p->y + re(4448));
fft_4_2_leaf(p->x + re(465), p->x + re(4561), p->x + re(2513), p->x + re(6609), p->x + re(7633), p->x + re(3537), p->x + re(1489), p->x + re(5585), p->y + re(4456));
fft_8_1_leaf(p->x + re(209), p->x + re(4305), p->x + re(2257), p->x + re(6353), p->x + re(1233), p->x + re(5329), p->x + re(7377), p->x + re(3281), p->y + re(4464));
fft_8_1_leaf(p->x + re(7889), p->x + re(3793), p->x + re(1745), p->x + re(5841), p->x + re(721), p->x + re(4817), p->x + re(6865), p->x + re(2769), p->y + re(4472));
fft_8_1_leaf(p->x + re(8177), p->x + re(4081), p->x + re(2033), p->x + re(6129), p->x + re(1009), p->x + re(5105), p->x + re(7153), p->x + re(3057), p->y + re(4480));
fft_4_2_leaf(p->x + re(497), p->x + re(4593), p->x + re(2545), p->x + re(6641), p->x + re(7665), p->x + re(3569), p->x + re(1521), p->x + re(5617), p->y + re(4488));
fft_8_1_leaf(p->x + re(241), p->x + re(4337), p->x + re(2289), p->x + re(6385), p->x + re(1265), p->x + re(5361), p->x + re(7409), p->x + re(3313), p->y + re(4496));
fft_8_1_leaf(p->x + re(7921), p->x + re(3825), p->x + re(1777), p->x + re(5873), p->x + re(753), p->x + re(4849), p->x + re(6897), p->x + re(2801), p->y + re(4504));
fft_8_1_leaf(p->x + re(113), p->x + re(4209), p->x + re(2161), p->x + re(6257), p->x + re(1137), p->x + re(5233), p->x + re(7281), p->x + re(3185), p->y + re(4512));
fft_4_2_leaf(p->x + re(625), p->x + re(4721), p->x + re(2673), p->x + re(6769), p->x + re(7793), p->x + re(3697), p->x + re(1649), p->x + re(5745), p->y + re(4520));
fft_8_1_leaf(p->x + re(8049), p->x + re(3953), p->x + re(1905), p->x + re(6001), p->x + re(881), p->x + re(4977), p->x + re(7025), p->x + re(2929), p->y + re(4528));
fft_4_2_leaf(p->x + re(369), p->x + re(4465), p->x + re(2417), p->x + re(6513), p->x + re(7537), p->x + re(3441), p->x + re(1393), p->x + re(5489), p->y + re(4536));
fft_8_1_leaf(p->x + re(49), p->x + re(4145), p->x + re(2097), p->x + re(6193), p->x + re(1073), p->x + re(5169), p->x + re(7217), p->x + re(3121), p->y + re(4544));
fft_4_2_leaf(p->x + re(561), p->x + re(4657), p->x + re(2609), p->x + re(6705), p->x + re(7729), p->x + re(3633), p->x + re(1585), p->x + re(5681), p->y + re(4552));
fft_8_1_leaf(p->x + re(305), p->x + re(4401), p->x + re(2353), p->x + re(6449), p->x + re(1329), p->x + re(5425), p->x + re(7473), p->x + re(3377), p->y + re(4560));
fft_8_1_leaf(p->x + re(7985), p->x + re(3889), p->x + re(1841), p->x + re(5937), p->x + re(817), p->x + re(4913), p->x + re(6961), p->x + re(2865), p->y + re(4568));
fft_8_1_leaf(p->x + re(8113), p->x + re(4017), p->x + re(1969), p->x + re(6065), p->x + re(945), p->x + re(5041), p->x + re(7089), p->x + re(2993), p->y + re(4576));
fft_4_2_leaf(p->x + re(433), p->x + re(4529), p->x + re(2481), p->x + re(6577), p->x + re(7601), p->x + re(3505), p->x + re(1457), p->x + re(5553), p->y + re(4584));
fft_8_1_leaf(p->x + re(177), p->x + re(4273), p->x + re(2225), p->x + re(6321), p->x + re(1201), p->x + re(5297), p->x + re(7345), p->x + re(3249), p->y + re(4592));
fft_8_1_leaf(p->x + re(7857), p->x + re(3761), p->x + re(1713), p->x + re(5809), p->x + re(689), p->x + re(4785), p->x + re(6833), p->x + re(2737), p->y + re(4600));
fft_8_1_leaf(p->x + re(9), p->x + re(4105), p->x + re(2057), p->x + re(6153), p->x + re(1033), p->x + re(5129), p->x + re(7177), p->x + re(3081), p->y + re(4608));
fft_4_2_leaf(p->x + re(521), p->x + re(4617), p->x + re(2569), p->x + re(6665), p->x + re(7689), p->x + re(3593), p->x + re(1545), p->x + re(5641), p->y + re(4616));
fft_8_1_leaf(p->x + re(265), p->x + re(4361), p->x + re(2313), p->x + re(6409), p->x + re(1289), p->x + re(5385), p->x + re(7433), p->x + re(3337), p->y + re(4624));
fft_8_1_leaf(p->x + re(7945), p->x + re(3849), p->x + re(1801), p->x + re(5897), p->x + re(777), p->x + re(4873), p->x + re(6921), p->x + re(2825), p->y + re(4632));
fft_8_1_leaf(p->x + re(137), p->x + re(4233), p->x + re(2185), p->x + re(6281), p->x + re(1161), p->x + re(5257), p->x + re(7305), p->x + re(3209), p->y + re(4640));
fft_4_2_leaf(p->x + re(649), p->x + re(4745), p->x + re(2697), p->x + re(6793), p->x + re(7817), p->x + re(3721), p->x + re(1673), p->x + re(5769), p->y + re(4648));
fft_8_1_leaf(p->x + re(8073), p->x + re(3977), p->x + re(1929), p->x + re(6025), p->x + re(905), p->x + re(5001), p->x + re(7049), p->x + re(2953), p->y + re(4656));
fft_4_2_leaf(p->x + re(393), p->x + re(4489), p->x + re(2441), p->x + re(6537), p->x + re(7561), p->x + re(3465), p->x + re(1417), p->x + re(5513), p->y + re(4664));
fft_8_1_leaf(p->x + re(73), p->x + re(4169), p->x + re(2121), p->x + re(6217), p->x + re(1097), p->x + re(5193), p->x + re(7241), p->x + re(3145), p->y + re(4672));
fft_4_2_leaf(p->x + re(585), p->x + re(4681), p->x + re(2633), p->x + re(6729), p->x + re(7753), p->x + re(3657), p->x + re(1609), p->x + re(5705), p->y + re(4680));
fft_8_1_leaf(p->x + re(329), p->x + re(4425), p->x + re(2377), p->x + re(6473), p->x + re(1353), p->x + re(5449), p->x + re(7497), p->x + re(3401), p->y + re(4688));
fft_8_1_leaf(p->x + re(8009), p->x + re(3913), p->x + re(1865), p->x + re(5961), p->x + re(841), p->x + re(4937), p->x + re(6985), p->x + re(2889), p->y + re(4696));
fft_8_1_leaf(p->x + re(8137), p->x + re(4041), p->x + re(1993), p->x + re(6089), p->x + re(969), p->x + re(5065), p->x + re(7113), p->x + re(3017), p->y + re(4704));
fft_4_2_leaf(p->x + re(457), p->x + re(4553), p->x + re(2505), p->x + re(6601), p->x + re(7625), p->x + re(3529), p->x + re(1481), p->x + re(5577), p->y + re(4712));
fft_8_1_leaf(p->x + re(201), p->x + re(4297), p->x + re(2249), p->x + re(6345), p->x + re(1225), p->x + re(5321), p->x + re(7369), p->x + re(3273), p->y + re(4720));
fft_8_1_leaf(p->x + re(7881), p->x + re(3785), p->x + re(1737), p->x + re(5833), p->x + re(713), p->x + re(4809), p->x + re(6857), p->x + re(2761), p->y + re(4728));
fft_8_1_leaf(p->x + re(41), p->x + re(4137), p->x + re(2089), p->x + re(6185), p->x + re(1065), p->x + re(5161), p->x + re(7209), p->x + re(3113), p->y + re(4736));
fft_4_2_leaf(p->x + re(553), p->x + re(4649), p->x + re(2601), p->x + re(6697), p->x + re(7721), p->x + re(3625), p->x + re(1577), p->x + re(5673), p->y + re(4744));
fft_8_1_leaf(p->x + re(297), p->x + re(4393), p->x + re(2345), p->x + re(6441), p->x + re(1321), p->x + re(5417), p->x + re(7465), p->x + re(3369), p->y + re(4752));
fft_8_1_leaf(p->x + re(7977), p->x + re(3881), p->x + re(1833), p->x + re(5929), p->x + re(809), p->x + re(4905), p->x + re(6953), p->x + re(2857), p->y + re(4760));
fft_8_1_leaf(p->x + re(169), p->x + re(4265), p->x + re(2217), p->x + re(6313), p->x + re(1193), p->x + re(5289), p->x + re(7337), p->x + re(3241), p->y + re(4768));
fft_4_2_leaf(p->x + re(681), p->x + re(4777), p->x + re(2729), p->x + re(6825), p->x + re(7849), p->x + re(3753), p->x + re(1705), p->x + re(5801), p->y + re(4776));
fft_8_1_leaf(p->x + re(8105), p->x + re(4009), p->x + re(1961), p->x + re(6057), p->x + re(937), p->x + re(5033), p->x + re(7081), p->x + re(2985), p->y + re(4784));
fft_4_2_leaf(p->x + re(425), p->x + re(4521), p->x + re(2473), p->x + re(6569), p->x + re(7593), p->x + re(3497), p->x + re(1449), p->x + re(5545), p->y + re(4792));
fft_8_1_leaf(p->x + re(8169), p->x + re(4073), p->x + re(2025), p->x + re(6121), p->x + re(1001), p->x + re(5097), p->x + re(7145), p->x + re(3049), p->y + re(4800));
fft_4_2_leaf(p->x + re(489), p->x + re(4585), p->x + re(2537), p->x + re(6633), p->x + re(7657), p->x + re(3561), p->x + re(1513), p->x + re(5609), p->y + re(4808));
fft_8_1_leaf(p->x + re(233), p->x + re(4329), p->x + re(2281), p->x + re(6377), p->x + re(1257), p->x + re(5353), p->x + re(7401), p->x + re(3305), p->y + re(4816));
fft_8_1_leaf(p->x + re(7913), p->x + re(3817), p->x + re(1769), p->x + re(5865), p->x + re(745), p->x + re(4841), p->x + re(6889), p->x + re(2793), p->y + re(4824));
fft_8_1_leaf(p->x + re(105), p->x + re(4201), p->x + re(2153), p->x + re(6249), p->x + re(1129), p->x + re(5225), p->x + re(7273), p->x + re(3177), p->y + re(4832));
fft_4_2_leaf(p->x + re(617), p->x + re(4713), p->x + re(2665), p->x + re(6761), p->x + re(7785), p->x + re(3689), p->x + re(1641), p->x + re(5737), p->y + re(4840));
fft_8_1_leaf(p->x + re(8041), p->x + re(3945), p->x + re(1897), p->x + re(5993), p->x + re(873), p->x + re(4969), p->x + re(7017), p->x + re(2921), p->y + re(4848));
fft_4_2_leaf(p->x + re(361), p->x + re(4457), p->x + re(2409), p->x + re(6505), p->x + re(7529), p->x + re(3433), p->x + re(1385), p->x + re(5481), p->y + re(4856));
fft_8_1_leaf(p->x + re(8185), p->x + re(4089), p->x + re(2041), p->x + re(6137), p->x + re(1017), p->x + re(5113), p->x + re(7161), p->x + re(3065), p->y + re(4864));
fft_4_2_leaf(p->x + re(505), p->x + re(4601), p->x + re(2553), p->x + re(6649), p->x + re(7673), p->x + re(3577), p->x + re(1529), p->x + re(5625), p->y + re(4872));
fft_8_1_leaf(p->x + re(249), p->x + re(4345), p->x + re(2297), p->x + re(6393), p->x + re(1273), p->x + re(5369), p->x + re(7417), p->x + re(3321), p->y + re(4880));
fft_8_1_leaf(p->x + re(7929), p->x + re(3833), p->x + re(1785), p->x + re(5881), p->x + re(761), p->x + re(4857), p->x + re(6905), p->x + re(2809), p->y + re(4888));
fft_8_1_leaf(p->x + re(121), p->x + re(4217), p->x + re(2169), p->x + re(6265), p->x + re(1145), p->x + re(5241), p->x + re(7289), p->x + re(3193), p->y + re(4896));
fft_4_2_leaf(p->x + re(633), p->x + re(4729), p->x + re(2681), p->x + re(6777), p->x + re(7801), p->x + re(3705), p->x + re(1657), p->x + re(5753), p->y + re(4904));
fft_8_1_leaf(p->x + re(8057), p->x + re(3961), p->x + re(1913), p->x + re(6009), p->x + re(889), p->x + re(4985), p->x + re(7033), p->x + re(2937), p->y + re(4912));
fft_4_2_leaf(p->x + re(377), p->x + re(4473), p->x + re(2425), p->x + re(6521), p->x + re(7545), p->x + re(3449), p->x + re(1401), p->x + re(5497), p->y + re(4920));
fft_8_1_leaf(p->x + re(57), p->x + re(4153), p->x + re(2105), p->x + re(6201), p->x + re(1081), p->x + re(5177), p->x + re(7225), p->x + re(3129), p->y + re(4928));
fft_4_2_leaf(p->x + re(569), p->x + re(4665), p->x + re(2617), p->x + re(6713), p->x + re(7737), p->x + re(3641), p->x + re(1593), p->x + re(5689), p->y + re(4936));
fft_8_1_leaf(p->x + re(313), p->x + re(4409), p->x + re(2361), p->x + re(6457), p->x + re(1337), p->x + re(5433), p->x + re(7481), p->x + re(3385), p->y + re(4944));
fft_8_1_leaf(p->x + re(7993), p->x + re(3897), p->x + re(1849), p->x + re(5945), p->x + re(825), p->x + re(4921), p->x + re(6969), p->x + re(2873), p->y + re(4952));
fft_8_1_leaf(p->x + re(8121), p->x + re(4025), p->x + re(1977), p->x + re(6073), p->x + re(953), p->x + re(5049), p->x + re(7097), p->x + re(3001), p->y + re(4960));
fft_4_2_leaf(p->x + re(441), p->x + re(4537), p->x + re(2489), p->x + re(6585), p->x + re(7609), p->x + re(3513), p->x + re(1465), p->x + re(5561), p->y + re(4968));
fft_8_1_leaf(p->x + re(185), p->x + re(4281), p->x + re(2233), p->x + re(6329), p->x + re(1209), p->x + re(5305), p->x + re(7353), p->x + re(3257), p->y + re(4976));
fft_8_1_leaf(p->x + re(7865), p->x + re(3769), p->x + re(1721), p->x + re(5817), p->x + re(697), p->x + re(4793), p->x + re(6841), p->x + re(2745), p->y + re(4984));
fft_8_1_leaf(p->x + re(25), p->x + re(4121), p->x + re(2073), p->x + re(6169), p->x + re(1049), p->x + re(5145), p->x + re(7193), p->x + re(3097), p->y + re(4992));
fft_4_2_leaf(p->x + re(537), p->x + re(4633), p->x + re(2585), p->x + re(6681), p->x + re(7705), p->x + re(3609), p->x + re(1561), p->x + re(5657), p->y + re(5000));
fft_8_1_leaf(p->x + re(281), p->x + re(4377), p->x + re(2329), p->x + re(6425), p->x + re(1305), p->x + re(5401), p->x + re(7449), p->x + re(3353), p->y + re(5008));
fft_8_1_leaf(p->x + re(7961), p->x + re(3865), p->x + re(1817), p->x + re(5913), p->x + re(793), p->x + re(4889), p->x + re(6937), p->x + re(2841), p->y + re(5016));
fft_8_1_leaf(p->x + re(153), p->x + re(4249), p->x + re(2201), p->x + re(6297), p->x + re(1177), p->x + re(5273), p->x + re(7321), p->x + re(3225), p->y + re(5024));
fft_4_2_leaf(p->x + re(665), p->x + re(4761), p->x + re(2713), p->x + re(6809), p->x + re(7833), p->x + re(3737), p->x + re(1689), p->x + re(5785), p->y + re(5032));
fft_8_1_leaf(p->x + re(8089), p->x + re(3993), p->x + re(1945), p->x + re(6041), p->x + re(921), p->x + re(5017), p->x + re(7065), p->x + re(2969), p->y + re(5040));
fft_4_2_leaf(p->x + re(409), p->x + re(4505), p->x + re(2457), p->x + re(6553), p->x + re(7577), p->x + re(3481), p->x + re(1433), p->x + re(5529), p->y + re(5048));
fft_8_1_leaf(p->x + re(8153), p->x + re(4057), p->x + re(2009), p->x + re(6105), p->x + re(985), p->x + re(5081), p->x + re(7129), p->x + re(3033), p->y + re(5056));
fft_4_2_leaf(p->x + re(473), p->x + re(4569), p->x + re(2521), p->x + re(6617), p->x + re(7641), p->x + re(3545), p->x + re(1497), p->x + re(5593), p->y + re(5064));
fft_8_1_leaf(p->x + re(217), p->x + re(4313), p->x + re(2265), p->x + re(6361), p->x + re(1241), p->x + re(5337), p->x + re(7385), p->x + re(3289), p->y + re(5072));
fft_8_1_leaf(p->x + re(7897), p->x + re(3801), p->x + re(1753), p->x + re(5849), p->x + re(729), p->x + re(4825), p->x + re(6873), p->x + re(2777), p->y + re(5080));
fft_8_1_leaf(p->x + re(89), p->x + re(4185), p->x + re(2137), p->x + re(6233), p->x + re(1113), p->x + re(5209), p->x + re(7257), p->x + re(3161), p->y + re(5088));
fft_4_2_leaf(p->x + re(601), p->x + re(4697), p->x + re(2649), p->x + re(6745), p->x + re(7769), p->x + re(3673), p->x + re(1625), p->x + re(5721), p->y + re(5096));
fft_8_1_leaf(p->x + re(8025), p->x + re(3929), p->x + re(1881), p->x + re(5977), p->x + re(857), p->x + re(4953), p->x + re(7001), p->x + re(2905), p->y + re(5104));
fft_4_2_leaf(p->x + re(345), p->x + re(4441), p->x + re(2393), p->x + re(6489), p->x + re(7513), p->x + re(3417), p->x + re(1369), p->x + re(5465), p->y + re(5112));
fft_8_1_leaf(p->x + re(5), p->x + re(4101), p->x + re(2053), p->x + re(6149), p->x + re(1029), p->x + re(5125), p->x + re(7173), p->x + re(3077), p->y + re(5120));
fft_4_2_leaf(p->x + re(517), p->x + re(4613), p->x + re(2565), p->x + re(6661), p->x + re(7685), p->x + re(3589), p->x + re(1541), p->x + re(5637), p->y + re(5128));
fft_8_1_leaf(p->x + re(261), p->x + re(4357), p->x + re(2309), p->x + re(6405), p->x + re(1285), p->x + re(5381), p->x + re(7429), p->x + re(3333), p->y + re(5136));
fft_8_1_leaf(p->x + re(7941), p->x + re(3845), p->x + re(1797), p->x + re(5893), p->x + re(773), p->x + re(4869), p->x + re(6917), p->x + re(2821), p->y + re(5144));
fft_8_1_leaf(p->x + re(133), p->x + re(4229), p->x + re(2181), p->x + re(6277), p->x + re(1157), p->x + re(5253), p->x + re(7301), p->x + re(3205), p->y + re(5152));
fft_4_2_leaf(p->x + re(645), p->x + re(4741), p->x + re(2693), p->x + re(6789), p->x + re(7813), p->x + re(3717), p->x + re(1669), p->x + re(5765), p->y + re(5160));
fft_8_1_leaf(p->x + re(8069), p->x + re(3973), p->x + re(1925), p->x + re(6021), p->x + re(901), p->x + re(4997), p->x + re(7045), p->x + re(2949), p->y + re(5168));
fft_4_2_leaf(p->x + re(389), p->x + re(4485), p->x + re(2437), p->x + re(6533), p->x + re(7557), p->x + re(3461), p->x + re(1413), p->x + re(5509), p->y + re(5176));
fft_8_1_leaf(p->x + re(69), p->x + re(4165), p->x + re(2117), p->x + re(6213), p->x + re(1093), p->x + re(5189), p->x + re(7237), p->x + re(3141), p->y + re(5184));
fft_4_2_leaf(p->x + re(581), p->x + re(4677), p->x + re(2629), p->x + re(6725), p->x + re(7749), p->x + re(3653), p->x + re(1605), p->x + re(5701), p->y + re(5192));
fft_8_1_leaf(p->x + re(325), p->x + re(4421), p->x + re(2373), p->x + re(6469), p->x + re(1349), p->x + re(5445), p->x + re(7493), p->x + re(3397), p->y + re(5200));
fft_8_1_leaf(p->x + re(8005), p->x + re(3909), p->x + re(1861), p->x + re(5957), p->x + re(837), p->x + re(4933), p->x + re(6981), p->x + re(2885), p->y + re(5208));
fft_8_1_leaf(p->x + re(8133), p->x + re(4037), p->x + re(1989), p->x + re(6085), p->x + re(965), p->x + re(5061), p->x + re(7109), p->x + re(3013), p->y + re(5216));
fft_4_2_leaf(p->x + re(453), p->x + re(4549), p->x + re(2501), p->x + re(6597), p->x + re(7621), p->x + re(3525), p->x + re(1477), p->x + re(5573), p->y + re(5224));
fft_8_1_leaf(p->x + re(197), p->x + re(4293), p->x + re(2245), p->x + re(6341), p->x + re(1221), p->x + re(5317), p->x + re(7365), p->x + re(3269), p->y + re(5232));
fft_8_1_leaf(p->x + re(7877), p->x + re(3781), p->x + re(1733), p->x + re(5829), p->x + re(709), p->x + re(4805), p->x + re(6853), p->x + re(2757), p->y + re(5240));
fft_8_1_leaf(p->x + re(37), p->x + re(4133), p->x + re(2085), p->x + re(6181), p->x + re(1061), p->x + re(5157), p->x + re(7205), p->x + re(3109), p->y + re(5248));
fft_4_2_leaf(p->x + re(549), p->x + re(4645), p->x + re(2597), p->x + re(6693), p->x + re(7717), p->x + re(3621), p->x + re(1573), p->x + re(5669), p->y + re(5256));
fft_8_1_leaf(p->x + re(293), p->x + re(4389), p->x + re(2341), p->x + re(6437), p->x + re(1317), p->x + re(5413), p->x + re(7461), p->x + re(3365), p->y + re(5264));
fft_8_1_leaf(p->x + re(7973), p->x + re(3877), p->x + re(1829), p->x + re(5925), p->x + re(805), p->x + re(4901), p->x + re(6949), p->x + re(2853), p->y + re(5272));
fft_8_1_leaf(p->x + re(165), p->x + re(4261), p->x + re(2213), p->x + re(6309), p->x + re(1189), p->x + re(5285), p->x + re(7333), p->x + re(3237), p->y + re(5280));
fft_4_2_leaf(p->x + re(677), p->x + re(4773), p->x + re(2725), p->x + re(6821), p->x + re(7845), p->x + re(3749), p->x + re(1701), p->x + re(5797), p->y + re(5288));
fft_8_1_leaf(p->x + re(8101), p->x + re(4005), p->x + re(1957), p->x + re(6053), p->x + re(933), p->x + re(5029), p->x + re(7077), p->x + re(2981), p->y + re(5296));
fft_4_2_leaf(p->x + re(421), p->x + re(4517), p->x + re(2469), p->x + re(6565), p->x + re(7589), p->x + re(3493), p->x + re(1445), p->x + re(5541), p->y + re(5304));
fft_8_1_leaf(p->x + re(8165), p->x + re(4069), p->x + re(2021), p->x + re(6117), p->x + re(997), p->x + re(5093), p->x + re(7141), p->x + re(3045), p->y + re(5312));
fft_4_2_leaf(p->x + re(485), p->x + re(4581), p->x + re(2533), p->x + re(6629), p->x + re(7653), p->x + re(3557), p->x + re(1509), p->x + re(5605), p->y + re(5320));
fft_8_1_leaf(p->x + re(229), p->x + re(4325), p->x + re(2277), p->x + re(6373), p->x + re(1253), p->x + re(5349), p->x + re(7397), p->x + re(3301), p->y + re(5328));
fft_8_1_leaf(p->x + re(7909), p->x + re(3813), p->x + re(1765), p->x + re(5861), p->x + re(741), p->x + re(4837), p->x + re(6885), p->x + re(2789), p->y + re(5336));
fft_8_1_leaf(p->x + re(101), p->x + re(4197), p->x + re(2149), p->x + re(6245), p->x + re(1125), p->x + re(5221), p->x + re(7269), p->x + re(3173), p->y + re(5344));
fft_4_2_leaf(p->x + re(613), p->x + re(4709), p->x + re(2661), p->x + re(6757), p->x + re(7781), p->x + re(3685), p->x + re(1637), p->x + re(5733), p->y + re(5352));
fft_8_1_leaf(p->x + re(8037), p->x + re(3941), p->x + re(1893), p->x + re(5989), p->x + re(869), p->x + re(4965), p->x + re(7013), p->x + re(2917), p->y + re(5360));
fft_4_2_leaf(p->x + re(357), p->x + re(4453), p->x + re(2405), p->x + re(6501), p->x + re(7525), p->x + re(3429), p->x + re(1381), p->x + re(5477), p->y + re(5368));
fft_8_1_leaf(p->x + re(21), p->x + re(4117), p->x + re(2069), p->x + re(6165), p->x + re(1045), p->x + re(5141), p->x + re(7189), p->x + re(3093), p->y + re(5376));
fft_4_2_leaf(p->x + re(533), p->x + re(4629), p->x + re(2581), p->x + re(6677), p->x + re(7701), p->x + re(3605), p->x + re(1557), p->x + re(5653), p->y + re(5384));
fft_8_1_leaf(p->x + re(277), p->x + re(4373), p->x + re(2325), p->x + re(6421), p->x + re(1301), p->x + re(5397), p->x + re(7445), p->x + re(3349), p->y + re(5392));
fft_8_1_leaf(p->x + re(7957), p->x + re(3861), p->x + re(1813), p->x + re(5909), p->x + re(789), p->x + re(4885), p->x + re(6933), p->x + re(2837), p->y + re(5400));
fft_8_1_leaf(p->x + re(149), p->x + re(4245), p->x + re(2197), p->x + re(6293), p->x + re(1173), p->x + re(5269), p->x + re(7317), p->x + re(3221), p->y + re(5408));
fft_4_2_leaf(p->x + re(661), p->x + re(4757), p->x + re(2709), p->x + re(6805), p->x + re(7829), p->x + re(3733), p->x + re(1685), p->x + re(5781), p->y + re(5416));
fft_8_1_leaf(p->x + re(8085), p->x + re(3989), p->x + re(1941), p->x + re(6037), p->x + re(917), p->x + re(5013), p->x + re(7061), p->x + re(2965), p->y + re(5424));
fft_4_2_leaf(p->x + re(405), p->x + re(4501), p->x + re(2453), p->x + re(6549), p->x + re(7573), p->x + re(3477), p->x + re(1429), p->x + re(5525), p->y + re(5432));
fft_8_1_leaf(p->x + re(85), p->x + re(4181), p->x + re(2133), p->x + re(6229), p->x + re(1109), p->x + re(5205), p->x + re(7253), p->x + re(3157), p->y + re(5440));
fft_4_2_leaf(p->x + re(597), p->x + re(4693), p->x + re(2645), p->x + re(6741), p->x + re(7765), p->x + re(3669), p->x + re(1621), p->x + re(5717), p->y + re(5448));
fft_8_1_leaf(p->x + re(341), p->x + re(4437), p->x + re(2389), p->x + re(6485), p->x + re(1365), p->x + re(5461), p->x + re(7509), p->x + re(3413), p->y + re(5456));
fft_8_1_leaf(p->x + re(8021), p->x + re(3925), p->x + re(1877), p->x + re(5973), p->x + re(853), p->x + re(4949), p->x + re(6997), p->x + re(2901), p->y + re(5464));
fft_8_1_leaf(p->x + re(8149), p->x + re(4053), p->x + re(2005), p->x + re(6101), p->x + re(981), p->x + re(5077), p->x + re(7125), p->x + re(3029), p->y + re(5472));
fft_4_2_leaf(p->x + re(469), p->x + re(4565), p->x + re(2517), p->x + re(6613), p->x + re(7637), p->x + re(3541), p->x + re(1493), p->x + re(5589), p->y + re(5480));
fft_8_1_leaf(p->x + re(213), p->x + re(4309), p->x + re(2261), p->x + re(6357), p->x + re(1237), p->x + re(5333), p->x + re(7381), p->x + re(3285), p->y + re(5488));
fft_8_1_leaf(p->x + re(7893), p->x + re(3797), p->x + re(1749), p->x + re(5845), p->x + re(725), p->x + re(4821), p->x + re(6869), p->x + re(2773), p->y + re(5496));
fft_8_1_leaf(p->x + re(8181), p->x + re(4085), p->x + re(2037), p->x + re(6133), p->x + re(1013), p->x + re(5109), p->x + re(7157), p->x + re(3061), p->y + re(5504));
fft_4_2_leaf(p->x + re(501), p->x + re(4597), p->x + re(2549), p->x + re(6645), p->x + re(7669), p->x + re(3573), p->x + re(1525), p->x + re(5621), p->y + re(5512));
fft_8_1_leaf(p->x + re(245), p->x + re(4341), p->x + re(2293), p->x + re(6389), p->x + re(1269), p->x + re(5365), p->x + re(7413), p->x + re(3317), p->y + re(5520));
fft_8_1_leaf(p->x + re(7925), p->x + re(3829), p->x + re(1781), p->x + re(5877), p->x + re(757), p->x + re(4853), p->x + re(6901), p->x + re(2805), p->y + re(5528));
fft_8_1_leaf(p->x + re(117), p->x + re(4213), p->x + re(2165), p->x + re(6261), p->x + re(1141), p->x + re(5237), p->x + re(7285), p->x + re(3189), p->y + re(5536));
fft_4_2_leaf(p->x + re(629), p->x + re(4725), p->x + re(2677), p->x + re(6773), p->x + re(7797), p->x + re(3701), p->x + re(1653), p->x + re(5749), p->y + re(5544));
fft_8_1_leaf(p->x + re(8053), p->x + re(3957), p->x + re(1909), p->x + re(6005), p->x + re(885), p->x + re(4981), p->x + re(7029), p->x + re(2933), p->y + re(5552));
fft_4_2_leaf(p->x + re(373), p->x + re(4469), p->x + re(2421), p->x + re(6517), p->x + re(7541), p->x + re(3445), p->x + re(1397), p->x + re(5493), p->y + re(5560));
fft_8_1_leaf(p->x + re(53), p->x + re(4149), p->x + re(2101), p->x + re(6197), p->x + re(1077), p->x + re(5173), p->x + re(7221), p->x + re(3125), p->y + re(5568));
fft_4_2_leaf(p->x + re(565), p->x + re(4661), p->x + re(2613), p->x + re(6709), p->x + re(7733), p->x + re(3637), p->x + re(1589), p->x + re(5685), p->y + re(5576));
fft_8_1_leaf(p->x + re(309), p->x + re(4405), p->x + re(2357), p->x + re(6453), p->x + re(1333), p->x + re(5429), p->x + re(7477), p->x + re(3381), p->y + re(5584));
fft_8_1_leaf(p->x + re(7989), p->x + re(3893), p->x + re(1845), p->x + re(5941), p->x + re(821), p->x + re(4917), p->x + re(6965), p->x + re(2869), p->y + re(5592));
fft_8_1_leaf(p->x + re(8117), p->x + re(4021), p->x + re(1973), p->x + re(6069), p->x + re(949), p->x + re(5045), p->x + re(7093), p->x + re(2997), p->y + re(5600));
fft_4_2_leaf(p->x + re(437), p->x + re(4533), p->x + re(2485), p->x + re(6581), p->x + re(7605), p->x + re(3509), p->x + re(1461), p->x + re(5557), p->y + re(5608));
fft_8_1_leaf(p->x + re(181), p->x + re(4277), p->x + re(2229), p->x + re(6325), p->x + re(1205), p->x + re(5301), p->x + re(7349), p->x + re(3253), p->y + re(5616));
fft_8_1_leaf(p->x + re(7861), p->x + re(3765), p->x + re(1717), p->x + re(5813), p->x + re(693), p->x + re(4789), p->x + re(6837), p->x + re(2741), p->y + re(5624));
fft_8_1_leaf(p->x + re(8189), p->x + re(4093), p->x + re(2045), p->x + re(6141), p->x + re(1021), p->x + re(5117), p->x + re(7165), p->x + re(3069), p->y + re(5632));
fft_4_2_leaf(p->x + re(509), p->x + re(4605), p->x + re(2557), p->x + re(6653), p->x + re(7677), p->x + re(3581), p->x + re(1533), p->x + re(5629), p->y + re(5640));
fft_8_1_leaf(p->x + re(253), p->x + re(4349), p->x + re(2301), p->x + re(6397), p->x + re(1277), p->x + re(5373), p->x + re(7421), p->x + re(3325), p->y + re(5648));
fft_8_1_leaf(p->x + re(7933), p->x + re(3837), p->x + re(1789), p->x + re(5885), p->x + re(765), p->x + re(4861), p->x + re(6909), p->x + re(2813), p->y + re(5656));
fft_8_1_leaf(p->x + re(125), p->x + re(4221), p->x + re(2173), p->x + re(6269), p->x + re(1149), p->x + re(5245), p->x + re(7293), p->x + re(3197), p->y + re(5664));
fft_4_2_leaf(p->x + re(637), p->x + re(4733), p->x + re(2685), p->x + re(6781), p->x + re(7805), p->x + re(3709), p->x + re(1661), p->x + re(5757), p->y + re(5672));
fft_8_1_leaf(p->x + re(8061), p->x + re(3965), p->x + re(1917), p->x + re(6013), p->x + re(893), p->x + re(4989), p->x + re(7037), p->x + re(2941), p->y + re(5680));
fft_4_2_leaf(p->x + re(381), p->x + re(4477), p->x + re(2429), p->x + re(6525), p->x + re(7549), p->x + re(3453), p->x + re(1405), p->x + re(5501), p->y + re(5688));
fft_8_1_leaf(p->x + re(61), p->x + re(4157), p->x + re(2109), p->x + re(6205), p->x + re(1085), p->x + re(5181), p->x + re(7229), p->x + re(3133), p->y + re(5696));
fft_4_2_leaf(p->x + re(573), p->x + re(4669), p->x + re(2621), p->x + re(6717), p->x + re(7741), p->x + re(3645), p->x + re(1597), p->x + re(5693), p->y + re(5704));
fft_8_1_leaf(p->x + re(317), p->x + re(4413), p->x + re(2365), p->x + re(6461), p->x + re(1341), p->x + re(5437), p->x + re(7485), p->x + re(3389), p->y + re(5712));
fft_8_1_leaf(p->x + re(7997), p->x + re(3901), p->x + re(1853), p->x + re(5949), p->x + re(829), p->x + re(4925), p->x + re(6973), p->x + re(2877), p->y + re(5720));
fft_8_1_leaf(p->x + re(8125), p->x + re(4029), p->x + re(1981), p->x + re(6077), p->x + re(957), p->x + re(5053), p->x + re(7101), p->x + re(3005), p->y + re(5728));
fft_4_2_leaf(p->x + re(445), p->x + re(4541), p->x + re(2493), p->x + re(6589), p->x + re(7613), p->x + re(3517), p->x + re(1469), p->x + re(5565), p->y + re(5736));
fft_8_1_leaf(p->x + re(189), p->x + re(4285), p->x + re(2237), p->x + re(6333), p->x + re(1213), p->x + re(5309), p->x + re(7357), p->x + re(3261), p->y + re(5744));
fft_8_1_leaf(p->x + re(7869), p->x + re(3773), p->x + re(1725), p->x + re(5821), p->x + re(701), p->x + re(4797), p->x + re(6845), p->x + re(2749), p->y + re(5752));
fft_8_1_leaf(p->x + re(29), p->x + re(4125), p->x + re(2077), p->x + re(6173), p->x + re(1053), p->x + re(5149), p->x + re(7197), p->x + re(3101), p->y + re(5760));
fft_4_2_leaf(p->x + re(541), p->x + re(4637), p->x + re(2589), p->x + re(6685), p->x + re(7709), p->x + re(3613), p->x + re(1565), p->x + re(5661), p->y + re(5768));
fft_8_1_leaf(p->x + re(285), p->x + re(4381), p->x + re(2333), p->x + re(6429), p->x + re(1309), p->x + re(5405), p->x + re(7453), p->x + re(3357), p->y + re(5776));
fft_8_1_leaf(p->x + re(7965), p->x + re(3869), p->x + re(1821), p->x + re(5917), p->x + re(797), p->x + re(4893), p->x + re(6941), p->x + re(2845), p->y + re(5784));
fft_8_1_leaf(p->x + re(157), p->x + re(4253), p->x + re(2205), p->x + re(6301), p->x + re(1181), p->x + re(5277), p->x + re(7325), p->x + re(3229), p->y + re(5792));
fft_4_2_leaf(p->x + re(669), p->x + re(4765), p->x + re(2717), p->x + re(6813), p->x + re(7837), p->x + re(3741), p->x + re(1693), p->x + re(5789), p->y + re(5800));
fft_8_1_leaf(p->x + re(8093), p->x + re(3997), p->x + re(1949), p->x + re(6045), p->x + re(925), p->x + re(5021), p->x + re(7069), p->x + re(2973), p->y + re(5808));
fft_4_2_leaf(p->x + re(413), p->x + re(4509), p->x + re(2461), p->x + re(6557), p->x + re(7581), p->x + re(3485), p->x + re(1437), p->x + re(5533), p->y + re(5816));
fft_8_1_leaf(p->x + re(8157), p->x + re(4061), p->x + re(2013), p->x + re(6109), p->x + re(989), p->x + re(5085), p->x + re(7133), p->x + re(3037), p->y + re(5824));
fft_4_2_leaf(p->x + re(477), p->x + re(4573), p->x + re(2525), p->x + re(6621), p->x + re(7645), p->x + re(3549), p->x + re(1501), p->x + re(5597), p->y + re(5832));
fft_8_1_leaf(p->x + re(221), p->x + re(4317), p->x + re(2269), p->x + re(6365), p->x + re(1245), p->x + re(5341), p->x + re(7389), p->x + re(3293), p->y + re(5840));
fft_8_1_leaf(p->x + re(7901), p->x + re(3805), p->x + re(1757), p->x + re(5853), p->x + re(733), p->x + re(4829), p->x + re(6877), p->x + re(2781), p->y + re(5848));
fft_8_1_leaf(p->x + re(93), p->x + re(4189), p->x + re(2141), p->x + re(6237), p->x + re(1117), p->x + re(5213), p->x + re(7261), p->x + re(3165), p->y + re(5856));
fft_4_2_leaf(p->x + re(605), p->x + re(4701), p->x + re(2653), p->x + re(6749), p->x + re(7773), p->x + re(3677), p->x + re(1629), p->x + re(5725), p->y + re(5864));
fft_8_1_leaf(p->x + re(8029), p->x + re(3933), p->x + re(1885), p->x + re(5981), p->x + re(861), p->x + re(4957), p->x + re(7005), p->x + re(2909), p->y + re(5872));
fft_4_2_leaf(p->x + re(349), p->x + re(4445), p->x + re(2397), p->x + re(6493), p->x + re(7517), p->x + re(3421), p->x + re(1373), p->x + re(5469), p->y + re(5880));
fft_8_1_leaf(p->x + re(13), p->x + re(4109), p->x + re(2061), p->x + re(6157), p->x + re(1037), p->x + re(5133), p->x + re(7181), p->x + re(3085), p->y + re(5888));
fft_4_2_leaf(p->x + re(525), p->x + re(4621), p->x + re(2573), p->x + re(6669), p->x + re(7693), p->x + re(3597), p->x + re(1549), p->x + re(5645), p->y + re(5896));
fft_8_1_leaf(p->x + re(269), p->x + re(4365), p->x + re(2317), p->x + re(6413), p->x + re(1293), p->x + re(5389), p->x + re(7437), p->x + re(3341), p->y + re(5904));
fft_8_1_leaf(p->x + re(7949), p->x + re(3853), p->x + re(1805), p->x + re(5901), p->x + re(781), p->x + re(4877), p->x + re(6925), p->x + re(2829), p->y + re(5912));
fft_8_1_leaf(p->x + re(141), p->x + re(4237), p->x + re(2189), p->x + re(6285), p->x + re(1165), p->x + re(5261), p->x + re(7309), p->x + re(3213), p->y + re(5920));
fft_4_2_leaf(p->x + re(653), p->x + re(4749), p->x + re(2701), p->x + re(6797), p->x + re(7821), p->x + re(3725), p->x + re(1677), p->x + re(5773), p->y + re(5928));
fft_8_1_leaf(p->x + re(8077), p->x + re(3981), p->x + re(1933), p->x + re(6029), p->x + re(909), p->x + re(5005), p->x + re(7053), p->x + re(2957), p->y + re(5936));
fft_4_2_leaf(p->x + re(397), p->x + re(4493), p->x + re(2445), p->x + re(6541), p->x + re(7565), p->x + re(3469), p->x + re(1421), p->x + re(5517), p->y + re(5944));
fft_8_1_leaf(p->x + re(77), p->x + re(4173), p->x + re(2125), p->x + re(6221), p->x + re(1101), p->x + re(5197), p->x + re(7245), p->x + re(3149), p->y + re(5952));
fft_4_2_leaf(p->x + re(589), p->x + re(4685), p->x + re(2637), p->x + re(6733), p->x + re(7757), p->x + re(3661), p->x + re(1613), p->x + re(5709), p->y + re(5960));
fft_8_1_leaf(p->x + re(333), p->x + re(4429), p->x + re(2381), p->x + re(6477), p->x + re(1357), p->x + re(5453), p->x + re(7501), p->x + re(3405), p->y + re(5968));
fft_8_1_leaf(p->x + re(8013), p->x + re(3917), p->x + re(1869), p->x + re(5965), p->x + re(845), p->x + re(4941), p->x + re(6989), p->x + re(2893), p->y + re(5976));
fft_8_1_leaf(p->x + re(8141), p->x + re(4045), p->x + re(1997), p->x + re(6093), p->x + re(973), p->x + re(5069), p->x + re(7117), p->x + re(3021), p->y + re(5984));
fft_4_2_leaf(p->x + re(461), p->x + re(4557), p->x + re(2509), p->x + re(6605), p->x + re(7629), p->x + re(3533), p->x + re(1485), p->x + re(5581), p->y + re(5992));
fft_8_1_leaf(p->x + re(205), p->x + re(4301), p->x + re(2253), p->x + re(6349), p->x + re(1229), p->x + re(5325), p->x + re(7373), p->x + re(3277), p->y + re(6000));
fft_8_1_leaf(p->x + re(7885), p->x + re(3789), p->x + re(1741), p->x + re(5837), p->x + re(717), p->x + re(4813), p->x + re(6861), p->x + re(2765), p->y + re(6008));
fft_8_1_leaf(p->x + re(8173), p->x + re(4077), p->x + re(2029), p->x + re(6125), p->x + re(1005), p->x + re(5101), p->x + re(7149), p->x + re(3053), p->y + re(6016));
fft_4_2_leaf(p->x + re(493), p->x + re(4589), p->x + re(2541), p->x + re(6637), p->x + re(7661), p->x + re(3565), p->x + re(1517), p->x + re(5613), p->y + re(6024));
fft_8_1_leaf(p->x + re(237), p->x + re(4333), p->x + re(2285), p->x + re(6381), p->x + re(1261), p->x + re(5357), p->x + re(7405), p->x + re(3309), p->y + re(6032));
fft_8_1_leaf(p->x + re(7917), p->x + re(3821), p->x + re(1773), p->x + re(5869), p->x + re(749), p->x + re(4845), p->x + re(6893), p->x + re(2797), p->y + re(6040));
fft_8_1_leaf(p->x + re(109), p->x + re(4205), p->x + re(2157), p->x + re(6253), p->x + re(1133), p->x + re(5229), p->x + re(7277), p->x + re(3181), p->y + re(6048));
fft_4_2_leaf(p->x + re(621), p->x + re(4717), p->x + re(2669), p->x + re(6765), p->x + re(7789), p->x + re(3693), p->x + re(1645), p->x + re(5741), p->y + re(6056));
fft_8_1_leaf(p->x + re(8045), p->x + re(3949), p->x + re(1901), p->x + re(5997), p->x + re(877), p->x + re(4973), p->x + re(7021), p->x + re(2925), p->y + re(6064));
fft_4_2_leaf(p->x + re(365), p->x + re(4461), p->x + re(2413), p->x + re(6509), p->x + re(7533), p->x + re(3437), p->x + re(1389), p->x + re(5485), p->y + re(6072));
fft_8_1_leaf(p->x + re(45), p->x + re(4141), p->x + re(2093), p->x + re(6189), p->x + re(1069), p->x + re(5165), p->x + re(7213), p->x + re(3117), p->y + re(6080));
fft_4_2_leaf(p->x + re(557), p->x + re(4653), p->x + re(2605), p->x + re(6701), p->x + re(7725), p->x + re(3629), p->x + re(1581), p->x + re(5677), p->y + re(6088));
fft_8_1_leaf(p->x + re(301), p->x + re(4397), p->x + re(2349), p->x + re(6445), p->x + re(1325), p->x + re(5421), p->x + re(7469), p->x + re(3373), p->y + re(6096));
fft_8_1_leaf(p->x + re(7981), p->x + re(3885), p->x + re(1837), p->x + re(5933), p->x + re(813), p->x + re(4909), p->x + re(6957), p->x + re(2861), p->y + re(6104));
fft_8_1_leaf(p->x + re(8109), p->x + re(4013), p->x + re(1965), p->x + re(6061), p->x + re(941), p->x + re(5037), p->x + re(7085), p->x + re(2989), p->y + re(6112));
fft_4_2_leaf(p->x + re(429), p->x + re(4525), p->x + re(2477), p->x + re(6573), p->x + re(7597), p->x + re(3501), p->x + re(1453), p->x + re(5549), p->y + re(6120));
fft_8_1_leaf(p->x + re(173), p->x + re(4269), p->x + re(2221), p->x + re(6317), p->x + re(1197), p->x + re(5293), p->x + re(7341), p->x + re(3245), p->y + re(6128));
fft_8_1_leaf(p->x + re(7853), p->x + re(3757), p->x + re(1709), p->x + re(5805), p->x + re(685), p->x + re(4781), p->x + re(6829), p->x + re(2733), p->y + re(6136));
fft_8_1_leaf(p->x + re(8191), p->x + re(4095), p->x + re(2047), p->x + re(6143), p->x + re(1023), p->x + re(5119), p->x + re(7167), p->x + re(3071), p->y + re(6144));
fft_4_2_leaf(p->x + re(511), p->x + re(4607), p->x + re(2559), p->x + re(6655), p->x + re(7679), p->x + re(3583), p->x + re(1535), p->x + re(5631), p->y + re(6152));
fft_8_1_leaf(p->x + re(255), p->x + re(4351), p->x + re(2303), p->x + re(6399), p->x + re(1279), p->x + re(5375), p->x + re(7423), p->x + re(3327), p->y + re(6160));
fft_8_1_leaf(p->x + re(7935), p->x + re(3839), p->x + re(1791), p->x + re(5887), p->x + re(767), p->x + re(4863), p->x + re(6911), p->x + re(2815), p->y + re(6168));
fft_8_1_leaf(p->x + re(127), p->x + re(4223), p->x + re(2175), p->x + re(6271), p->x + re(1151), p->x + re(5247), p->x + re(7295), p->x + re(3199), p->y + re(6176));
fft_4_2_leaf(p->x + re(639), p->x + re(4735), p->x + re(2687), p->x + re(6783), p->x + re(7807), p->x + re(3711), p->x + re(1663), p->x + re(5759), p->y + re(6184));
fft_8_1_leaf(p->x + re(8063), p->x + re(3967), p->x + re(1919), p->x + re(6015), p->x + re(895), p->x + re(4991), p->x + re(7039), p->x + re(2943), p->y + re(6192));
fft_4_2_leaf(p->x + re(383), p->x + re(4479), p->x + re(2431), p->x + re(6527), p->x + re(7551), p->x + re(3455), p->x + re(1407), p->x + re(5503), p->y + re(6200));
fft_8_1_leaf(p->x + re(63), p->x + re(4159), p->x + re(2111), p->x + re(6207), p->x + re(1087), p->x + re(5183), p->x + re(7231), p->x + re(3135), p->y + re(6208));
fft_4_2_leaf(p->x + re(575), p->x + re(4671), p->x + re(2623), p->x + re(6719), p->x + re(7743), p->x + re(3647), p->x + re(1599), p->x + re(5695), p->y + re(6216));
fft_8_1_leaf(p->x + re(319), p->x + re(4415), p->x + re(2367), p->x + re(6463), p->x + re(1343), p->x + re(5439), p->x + re(7487), p->x + re(3391), p->y + re(6224));
fft_8_1_leaf(p->x + re(7999), p->x + re(3903), p->x + re(1855), p->x + re(5951), p->x + re(831), p->x + re(4927), p->x + re(6975), p->x + re(2879), p->y + re(6232));
fft_8_1_leaf(p->x + re(8127), p->x + re(4031), p->x + re(1983), p->x + re(6079), p->x + re(959), p->x + re(5055), p->x + re(7103), p->x + re(3007), p->y + re(6240));
fft_4_2_leaf(p->x + re(447), p->x + re(4543), p->x + re(2495), p->x + re(6591), p->x + re(7615), p->x + re(3519), p->x + re(1471), p->x + re(5567), p->y + re(6248));
fft_8_1_leaf(p->x + re(191), p->x + re(4287), p->x + re(2239), p->x + re(6335), p->x + re(1215), p->x + re(5311), p->x + re(7359), p->x + re(3263), p->y + re(6256));
fft_8_1_leaf(p->x + re(7871), p->x + re(3775), p->x + re(1727), p->x + re(5823), p->x + re(703), p->x + re(4799), p->x + re(6847), p->x + re(2751), p->y + re(6264));
fft_8_1_leaf(p->x + re(31), p->x + re(4127), p->x + re(2079), p->x + re(6175), p->x + re(1055), p->x + re(5151), p->x + re(7199), p->x + re(3103), p->y + re(6272));
fft_4_2_leaf(p->x + re(543), p->x + re(4639), p->x + re(2591), p->x + re(6687), p->x + re(7711), p->x + re(3615), p->x + re(1567), p->x + re(5663), p->y + re(6280));
fft_8_1_leaf(p->x + re(287), p->x + re(4383), p->x + re(2335), p->x + re(6431), p->x + re(1311), p->x + re(5407), p->x + re(7455), p->x + re(3359), p->y + re(6288));
fft_8_1_leaf(p->x + re(7967), p->x + re(3871), p->x + re(1823), p->x + re(5919), p->x + re(799), p->x + re(4895), p->x + re(6943), p->x + re(2847), p->y + re(6296));
fft_8_1_leaf(p->x + re(159), p->x + re(4255), p->x + re(2207), p->x + re(6303), p->x + re(1183), p->x + re(5279), p->x + re(7327), p->x + re(3231), p->y + re(6304));
fft_4_2_leaf(p->x + re(671), p->x + re(4767), p->x + re(2719), p->x + re(6815), p->x + re(7839), p->x + re(3743), p->x + re(1695), p->x + re(5791), p->y + re(6312));
fft_8_1_leaf(p->x + re(8095), p->x + re(3999), p->x + re(1951), p->x + re(6047), p->x + re(927), p->x + re(5023), p->x + re(7071), p->x + re(2975), p->y + re(6320));
fft_4_2_leaf(p->x + re(415), p->x + re(4511), p->x + re(2463), p->x + re(6559), p->x + re(7583), p->x + re(3487), p->x + re(1439), p->x + re(5535), p->y + re(6328));
fft_8_1_leaf(p->x + re(8159), p->x + re(4063), p->x + re(2015), p->x + re(6111), p->x + re(991), p->x + re(5087), p->x + re(7135), p->x + re(3039), p->y + re(6336));
fft_4_2_leaf(p->x + re(479), p->x + re(4575), p->x + re(2527), p->x + re(6623), p->x + re(7647), p->x + re(3551), p->x + re(1503), p->x + re(5599), p->y + re(6344));
fft_8_1_leaf(p->x + re(223), p->x + re(4319), p->x + re(2271), p->x + re(6367), p->x + re(1247), p->x + re(5343), p->x + re(7391), p->x + re(3295), p->y + re(6352));
fft_8_1_leaf(p->x + re(7903), p->x + re(3807), p->x + re(1759), p->x + re(5855), p->x + re(735), p->x + re(4831), p->x + re(6879), p->x + re(2783), p->y + re(6360));
fft_8_1_leaf(p->x + re(95), p->x + re(4191), p->x + re(2143), p->x + re(6239), p->x + re(1119), p->x + re(5215), p->x + re(7263), p->x + re(3167), p->y + re(6368));
fft_4_2_leaf(p->x + re(607), p->x + re(4703), p->x + re(2655), p->x + re(6751), p->x + re(7775), p->x + re(3679), p->x + re(1631), p->x + re(5727), p->y + re(6376));
fft_8_1_leaf(p->x + re(8031), p->x + re(3935), p->x + re(1887), p->x + re(5983), p->x + re(863), p->x + re(4959), p->x + re(7007), p->x + re(2911), p->y + re(6384));
fft_4_2_leaf(p->x + re(351), p->x + re(4447), p->x + re(2399), p->x + re(6495), p->x + re(7519), p->x + re(3423), p->x + re(1375), p->x + re(5471), p->y + re(6392));
fft_8_1_leaf(p->x + re(15), p->x + re(4111), p->x + re(2063), p->x + re(6159), p->x + re(1039), p->x + re(5135), p->x + re(7183), p->x + re(3087), p->y + re(6400));
fft_4_2_leaf(p->x + re(527), p->x + re(4623), p->x + re(2575), p->x + re(6671), p->x + re(7695), p->x + re(3599), p->x + re(1551), p->x + re(5647), p->y + re(6408));
fft_8_1_leaf(p->x + re(271), p->x + re(4367), p->x + re(2319), p->x + re(6415), p->x + re(1295), p->x + re(5391), p->x + re(7439), p->x + re(3343), p->y + re(6416));
fft_8_1_leaf(p->x + re(7951), p->x + re(3855), p->x + re(1807), p->x + re(5903), p->x + re(783), p->x + re(4879), p->x + re(6927), p->x + re(2831), p->y + re(6424));
fft_8_1_leaf(p->x + re(143), p->x + re(4239), p->x + re(2191), p->x + re(6287), p->x + re(1167), p->x + re(5263), p->x + re(7311), p->x + re(3215), p->y + re(6432));
fft_4_2_leaf(p->x + re(655), p->x + re(4751), p->x + re(2703), p->x + re(6799), p->x + re(7823), p->x + re(3727), p->x + re(1679), p->x + re(5775), p->y + re(6440));
fft_8_1_leaf(p->x + re(8079), p->x + re(3983), p->x + re(1935), p->x + re(6031), p->x + re(911), p->x + re(5007), p->x + re(7055), p->x + re(2959), p->y + re(6448));
fft_4_2_leaf(p->x + re(399), p->x + re(4495), p->x + re(2447), p->x + re(6543), p->x + re(7567), p->x + re(3471), p->x + re(1423), p->x + re(5519), p->y + re(6456));
fft_8_1_leaf(p->x + re(79), p->x + re(4175), p->x + re(2127), p->x + re(6223), p->x + re(1103), p->x + re(5199), p->x + re(7247), p->x + re(3151), p->y + re(6464));
fft_4_2_leaf(p->x + re(591), p->x + re(4687), p->x + re(2639), p->x + re(6735), p->x + re(7759), p->x + re(3663), p->x + re(1615), p->x + re(5711), p->y + re(6472));
fft_8_1_leaf(p->x + re(335), p->x + re(4431), p->x + re(2383), p->x + re(6479), p->x + re(1359), p->x + re(5455), p->x + re(7503), p->x + re(3407), p->y + re(6480));
fft_8_1_leaf(p->x + re(8015), p->x + re(3919), p->x + re(1871), p->x + re(5967), p->x + re(847), p->x + re(4943), p->x + re(6991), p->x + re(2895), p->y + re(6488));
fft_8_1_leaf(p->x + re(8143), p->x + re(4047), p->x + re(1999), p->x + re(6095), p->x + re(975), p->x + re(5071), p->x + re(7119), p->x + re(3023), p->y + re(6496));
fft_4_2_leaf(p->x + re(463), p->x + re(4559), p->x + re(2511), p->x + re(6607), p->x + re(7631), p->x + re(3535), p->x + re(1487), p->x + re(5583), p->y + re(6504));
fft_8_1_leaf(p->x + re(207), p->x + re(4303), p->x + re(2255), p->x + re(6351), p->x + re(1231), p->x + re(5327), p->x + re(7375), p->x + re(3279), p->y + re(6512));
fft_8_1_leaf(p->x + re(7887), p->x + re(3791), p->x + re(1743), p->x + re(5839), p->x + re(719), p->x + re(4815), p->x + re(6863), p->x + re(2767), p->y + re(6520));
fft_8_1_leaf(p->x + re(8175), p->x + re(4079), p->x + re(2031), p->x + re(6127), p->x + re(1007), p->x + re(5103), p->x + re(7151), p->x + re(3055), p->y + re(6528));
fft_4_2_leaf(p->x + re(495), p->x + re(4591), p->x + re(2543), p->x + re(6639), p->x + re(7663), p->x + re(3567), p->x + re(1519), p->x + re(5615), p->y + re(6536));
fft_8_1_leaf(p->x + re(239), p->x + re(4335), p->x + re(2287), p->x + re(6383), p->x + re(1263), p->x + re(5359), p->x + re(7407), p->x + re(3311), p->y + re(6544));
fft_8_1_leaf(p->x + re(7919), p->x + re(3823), p->x + re(1775), p->x + re(5871), p->x + re(751), p->x + re(4847), p->x + re(6895), p->x + re(2799), p->y + re(6552));
fft_8_1_leaf(p->x + re(111), p->x + re(4207), p->x + re(2159), p->x + re(6255), p->x + re(1135), p->x + re(5231), p->x + re(7279), p->x + re(3183), p->y + re(6560));
fft_4_2_leaf(p->x + re(623), p->x + re(4719), p->x + re(2671), p->x + re(6767), p->x + re(7791), p->x + re(3695), p->x + re(1647), p->x + re(5743), p->y + re(6568));
fft_8_1_leaf(p->x + re(8047), p->x + re(3951), p->x + re(1903), p->x + re(5999), p->x + re(879), p->x + re(4975), p->x + re(7023), p->x + re(2927), p->y + re(6576));
fft_4_2_leaf(p->x + re(367), p->x + re(4463), p->x + re(2415), p->x + re(6511), p->x + re(7535), p->x + re(3439), p->x + re(1391), p->x + re(5487), p->y + re(6584));
fft_8_1_leaf(p->x + re(47), p->x + re(4143), p->x + re(2095), p->x + re(6191), p->x + re(1071), p->x + re(5167), p->x + re(7215), p->x + re(3119), p->y + re(6592));
fft_4_2_leaf(p->x + re(559), p->x + re(4655), p->x + re(2607), p->x + re(6703), p->x + re(7727), p->x + re(3631), p->x + re(1583), p->x + re(5679), p->y + re(6600));
fft_8_1_leaf(p->x + re(303), p->x + re(4399), p->x + re(2351), p->x + re(6447), p->x + re(1327), p->x + re(5423), p->x + re(7471), p->x + re(3375), p->y + re(6608));
fft_8_1_leaf(p->x + re(7983), p->x + re(3887), p->x + re(1839), p->x + re(5935), p->x + re(815), p->x + re(4911), p->x + re(6959), p->x + re(2863), p->y + re(6616));
fft_8_1_leaf(p->x + re(8111), p->x + re(4015), p->x + re(1967), p->x + re(6063), p->x + re(943), p->x + re(5039), p->x + re(7087), p->x + re(2991), p->y + re(6624));
fft_4_2_leaf(p->x + re(431), p->x + re(4527), p->x + re(2479), p->x + re(6575), p->x + re(7599), p->x + re(3503), p->x + re(1455), p->x + re(5551), p->y + re(6632));
fft_8_1_leaf(p->x + re(175), p->x + re(4271), p->x + re(2223), p->x + re(6319), p->x + re(1199), p->x + re(5295), p->x + re(7343), p->x + re(3247), p->y + re(6640));
fft_8_1_leaf(p->x + re(7855), p->x + re(3759), p->x + re(1711), p->x + re(5807), p->x + re(687), p->x + re(4783), p->x + re(6831), p->x + re(2735), p->y + re(6648));
fft_8_1_leaf(p->x + re(7), p->x + re(4103), p->x + re(2055), p->x + re(6151), p->x + re(1031), p->x + re(5127), p->x + re(7175), p->x + re(3079), p->y + re(6656));
fft_4_2_leaf(p->x + re(519), p->x + re(4615), p->x + re(2567), p->x + re(6663), p->x + re(7687), p->x + re(3591), p->x + re(1543), p->x + re(5639), p->y + re(6664));
fft_8_1_leaf(p->x + re(263), p->x + re(4359), p->x + re(2311), p->x + re(6407), p->x + re(1287), p->x + re(5383), p->x + re(7431), p->x + re(3335), p->y + re(6672));
fft_8_1_leaf(p->x + re(7943), p->x + re(3847), p->x + re(1799), p->x + re(5895), p->x + re(775), p->x + re(4871), p->x + re(6919), p->x + re(2823), p->y + re(6680));
fft_8_1_leaf(p->x + re(135), p->x + re(4231), p->x + re(2183), p->x + re(6279), p->x + re(1159), p->x + re(5255), p->x + re(7303), p->x + re(3207), p->y + re(6688));
fft_4_2_leaf(p->x + re(647), p->x + re(4743), p->x + re(2695), p->x + re(6791), p->x + re(7815), p->x + re(3719), p->x + re(1671), p->x + re(5767), p->y + re(6696));
fft_8_1_leaf(p->x + re(8071), p->x + re(3975), p->x + re(1927), p->x + re(6023), p->x + re(903), p->x + re(4999), p->x + re(7047), p->x + re(2951), p->y + re(6704));
fft_4_2_leaf(p->x + re(391), p->x + re(4487), p->x + re(2439), p->x + re(6535), p->x + re(7559), p->x + re(3463), p->x + re(1415), p->x + re(5511), p->y + re(6712));
fft_8_1_leaf(p->x + re(71), p->x + re(4167), p->x + re(2119), p->x + re(6215), p->x + re(1095), p->x + re(5191), p->x + re(7239), p->x + re(3143), p->y + re(6720));
fft_4_2_leaf(p->x + re(583), p->x + re(4679), p->x + re(2631), p->x + re(6727), p->x + re(7751), p->x + re(3655), p->x + re(1607), p->x + re(5703), p->y + re(6728));
fft_8_1_leaf(p->x + re(327), p->x + re(4423), p->x + re(2375), p->x + re(6471), p->x + re(1351), p->x + re(5447), p->x + re(7495), p->x + re(3399), p->y + re(6736));
fft_8_1_leaf(p->x + re(8007), p->x + re(3911), p->x + re(1863), p->x + re(5959), p->x + re(839), p->x + re(4935), p->x + re(6983), p->x + re(2887), p->y + re(6744));
fft_8_1_leaf(p->x + re(8135), p->x + re(4039), p->x + re(1991), p->x + re(6087), p->x + re(967), p->x + re(5063), p->x + re(7111), p->x + re(3015), p->y + re(6752));
fft_4_2_leaf(p->x + re(455), p->x + re(4551), p->x + re(2503), p->x + re(6599), p->x + re(7623), p->x + re(3527), p->x + re(1479), p->x + re(5575), p->y + re(6760));
fft_8_1_leaf(p->x + re(199), p->x + re(4295), p->x + re(2247), p->x + re(6343), p->x + re(1223), p->x + re(5319), p->x + re(7367), p->x + re(3271), p->y + re(6768));
fft_8_1_leaf(p->x + re(7879), p->x + re(3783), p->x + re(1735), p->x + re(5831), p->x + re(711), p->x + re(4807), p->x + re(6855), p->x + re(2759), p->y + re(6776));
fft_8_1_leaf(p->x + re(39), p->x + re(4135), p->x + re(2087), p->x + re(6183), p->x + re(1063), p->x + re(5159), p->x + re(7207), p->x + re(3111), p->y + re(6784));
fft_4_2_leaf(p->x + re(551), p->x + re(4647), p->x + re(2599), p->x + re(6695), p->x + re(7719), p->x + re(3623), p->x + re(1575), p->x + re(5671), p->y + re(6792));
fft_8_1_leaf(p->x + re(295), p->x + re(4391), p->x + re(2343), p->x + re(6439), p->x + re(1319), p->x + re(5415), p->x + re(7463), p->x + re(3367), p->y + re(6800));
fft_8_1_leaf(p->x + re(7975), p->x + re(3879), p->x + re(1831), p->x + re(5927), p->x + re(807), p->x + re(4903), p->x + re(6951), p->x + re(2855), p->y + re(6808));
fft_8_1_leaf(p->x + re(167), p->x + re(4263), p->x + re(2215), p->x + re(6311), p->x + re(1191), p->x + re(5287), p->x + re(7335), p->x + re(3239), p->y + re(6816));
fft_4_2_leaf(p->x + re(679), p->x + re(4775), p->x + re(2727), p->x + re(6823), p->x + re(7847), p->x + re(3751), p->x + re(1703), p->x + re(5799), p->y + re(6824));
fft_8_1_leaf(p->x + re(8103), p->x + re(4007), p->x + re(1959), p->x + re(6055), p->x + re(935), p->x + re(5031), p->x + re(7079), p->x + re(2983), p->y + re(6832));
fft_4_2_leaf(p->x + re(423), p->x + re(4519), p->x + re(2471), p->x + re(6567), p->x + re(7591), p->x + re(3495), p->x + re(1447), p->x + re(5543), p->y + re(6840));
fft_8_1_leaf(p->x + re(8167), p->x + re(4071), p->x + re(2023), p->x + re(6119), p->x + re(999), p->x + re(5095), p->x + re(7143), p->x + re(3047), p->y + re(6848));
fft_4_2_leaf(p->x + re(487), p->x + re(4583), p->x + re(2535), p->x + re(6631), p->x + re(7655), p->x + re(3559), p->x + re(1511), p->x + re(5607), p->y + re(6856));
fft_8_1_leaf(p->x + re(231), p->x + re(4327), p->x + re(2279), p->x + re(6375), p->x + re(1255), p->x + re(5351), p->x + re(7399), p->x + re(3303), p->y + re(6864));
fft_8_1_leaf(p->x + re(7911), p->x + re(3815), p->x + re(1767), p->x + re(5863), p->x + re(743), p->x + re(4839), p->x + re(6887), p->x + re(2791), p->y + re(6872));
fft_8_1_leaf(p->x + re(103), p->x + re(4199), p->x + re(2151), p->x + re(6247), p->x + re(1127), p->x + re(5223), p->x + re(7271), p->x + re(3175), p->y + re(6880));
fft_4_2_leaf(p->x + re(615), p->x + re(4711), p->x + re(2663), p->x + re(6759), p->x + re(7783), p->x + re(3687), p->x + re(1639), p->x + re(5735), p->y + re(6888));
fft_8_1_leaf(p->x + re(8039), p->x + re(3943), p->x + re(1895), p->x + re(5991), p->x + re(871), p->x + re(4967), p->x + re(7015), p->x + re(2919), p->y + re(6896));
fft_4_2_leaf(p->x + re(359), p->x + re(4455), p->x + re(2407), p->x + re(6503), p->x + re(7527), p->x + re(3431), p->x + re(1383), p->x + re(5479), p->y + re(6904));
fft_8_1_leaf(p->x + re(8183), p->x + re(4087), p->x + re(2039), p->x + re(6135), p->x + re(1015), p->x + re(5111), p->x + re(7159), p->x + re(3063), p->y + re(6912));
fft_4_2_leaf(p->x + re(503), p->x + re(4599), p->x + re(2551), p->x + re(6647), p->x + re(7671), p->x + re(3575), p->x + re(1527), p->x + re(5623), p->y + re(6920));
fft_8_1_leaf(p->x + re(247), p->x + re(4343), p->x + re(2295), p->x + re(6391), p->x + re(1271), p->x + re(5367), p->x + re(7415), p->x + re(3319), p->y + re(6928));
fft_8_1_leaf(p->x + re(7927), p->x + re(3831), p->x + re(1783), p->x + re(5879), p->x + re(759), p->x + re(4855), p->x + re(6903), p->x + re(2807), p->y + re(6936));
fft_8_1_leaf(p->x + re(119), p->x + re(4215), p->x + re(2167), p->x + re(6263), p->x + re(1143), p->x + re(5239), p->x + re(7287), p->x + re(3191), p->y + re(6944));
fft_4_2_leaf(p->x + re(631), p->x + re(4727), p->x + re(2679), p->x + re(6775), p->x + re(7799), p->x + re(3703), p->x + re(1655), p->x + re(5751), p->y + re(6952));
fft_8_1_leaf(p->x + re(8055), p->x + re(3959), p->x + re(1911), p->x + re(6007), p->x + re(887), p->x + re(4983), p->x + re(7031), p->x + re(2935), p->y + re(6960));
fft_4_2_leaf(p->x + re(375), p->x + re(4471), p->x + re(2423), p->x + re(6519), p->x + re(7543), p->x + re(3447), p->x + re(1399), p->x + re(5495), p->y + re(6968));
fft_8_1_leaf(p->x + re(55), p->x + re(4151), p->x + re(2103), p->x + re(6199), p->x + re(1079), p->x + re(5175), p->x + re(7223), p->x + re(3127), p->y + re(6976));
fft_4_2_leaf(p->x + re(567), p->x + re(4663), p->x + re(2615), p->x + re(6711), p->x + re(7735), p->x + re(3639), p->x + re(1591), p->x + re(5687), p->y + re(6984));
fft_8_1_leaf(p->x + re(311), p->x + re(4407), p->x + re(2359), p->x + re(6455), p->x + re(1335), p->x + re(5431), p->x + re(7479), p->x + re(3383), p->y + re(6992));
fft_8_1_leaf(p->x + re(7991), p->x + re(3895), p->x + re(1847), p->x + re(5943), p->x + re(823), p->x + re(4919), p->x + re(6967), p->x + re(2871), p->y + re(7000));
fft_8_1_leaf(p->x + re(8119), p->x + re(4023), p->x + re(1975), p->x + re(6071), p->x + re(951), p->x + re(5047), p->x + re(7095), p->x + re(2999), p->y + re(7008));
fft_4_2_leaf(p->x + re(439), p->x + re(4535), p->x + re(2487), p->x + re(6583), p->x + re(7607), p->x + re(3511), p->x + re(1463), p->x + re(5559), p->y + re(7016));
fft_8_1_leaf(p->x + re(183), p->x + re(4279), p->x + re(2231), p->x + re(6327), p->x + re(1207), p->x + re(5303), p->x + re(7351), p->x + re(3255), p->y + re(7024));
fft_8_1_leaf(p->x + re(7863), p->x + re(3767), p->x + re(1719), p->x + re(5815), p->x + re(695), p->x + re(4791), p->x + re(6839), p->x + re(2743), p->y + re(7032));
fft_8_1_leaf(p->x + re(23), p->x + re(4119), p->x + re(2071), p->x + re(6167), p->x + re(1047), p->x + re(5143), p->x + re(7191), p->x + re(3095), p->y + re(7040));
fft_4_2_leaf(p->x + re(535), p->x + re(4631), p->x + re(2583), p->x + re(6679), p->x + re(7703), p->x + re(3607), p->x + re(1559), p->x + re(5655), p->y + re(7048));
fft_8_1_leaf(p->x + re(279), p->x + re(4375), p->x + re(2327), p->x + re(6423), p->x + re(1303), p->x + re(5399), p->x + re(7447), p->x + re(3351), p->y + re(7056));
fft_8_1_leaf(p->x + re(7959), p->x + re(3863), p->x + re(1815), p->x + re(5911), p->x + re(791), p->x + re(4887), p->x + re(6935), p->x + re(2839), p->y + re(7064));
fft_8_1_leaf(p->x + re(151), p->x + re(4247), p->x + re(2199), p->x + re(6295), p->x + re(1175), p->x + re(5271), p->x + re(7319), p->x + re(3223), p->y + re(7072));
fft_4_2_leaf(p->x + re(663), p->x + re(4759), p->x + re(2711), p->x + re(6807), p->x + re(7831), p->x + re(3735), p->x + re(1687), p->x + re(5783), p->y + re(7080));
fft_8_1_leaf(p->x + re(8087), p->x + re(3991), p->x + re(1943), p->x + re(6039), p->x + re(919), p->x + re(5015), p->x + re(7063), p->x + re(2967), p->y + re(7088));
fft_4_2_leaf(p->x + re(407), p->x + re(4503), p->x + re(2455), p->x + re(6551), p->x + re(7575), p->x + re(3479), p->x + re(1431), p->x + re(5527), p->y + re(7096));
fft_8_1_leaf(p->x + re(8151), p->x + re(4055), p->x + re(2007), p->x + re(6103), p->x + re(983), p->x + re(5079), p->x + re(7127), p->x + re(3031), p->y + re(7104));
fft_4_2_leaf(p->x + re(471), p->x + re(4567), p->x + re(2519), p->x + re(6615), p->x + re(7639), p->x + re(3543), p->x + re(1495), p->x + re(5591), p->y + re(7112));
fft_8_1_leaf(p->x + re(215), p->x + re(4311), p->x + re(2263), p->x + re(6359), p->x + re(1239), p->x + re(5335), p->x + re(7383), p->x + re(3287), p->y + re(7120));
fft_8_1_leaf(p->x + re(7895), p->x + re(3799), p->x + re(1751), p->x + re(5847), p->x + re(727), p->x + re(4823), p->x + re(6871), p->x + re(2775), p->y + re(7128));
fft_8_1_leaf(p->x + re(87), p->x + re(4183), p->x + re(2135), p->x + re(6231), p->x + re(1111), p->x + re(5207), p->x + re(7255), p->x + re(3159), p->y + re(7136));
fft_4_2_leaf(p->x + re(599), p->x + re(4695), p->x + re(2647), p->x + re(6743), p->x + re(7767), p->x + re(3671), p->x + re(1623), p->x + re(5719), p->y + re(7144));
fft_8_1_leaf(p->x + re(8023), p->x + re(3927), p->x + re(1879), p->x + re(5975), p->x + re(855), p->x + re(4951), p->x + re(6999), p->x + re(2903), p->y + re(7152));
fft_4_2_leaf(p->x + re(343), p->x + re(4439), p->x + re(2391), p->x + re(6487), p->x + re(7511), p->x + re(3415), p->x + re(1367), p->x + re(5463), p->y + re(7160));
fft_8_1_leaf(p->x + re(3), p->x + re(4099), p->x + re(2051), p->x + re(6147), p->x + re(1027), p->x + re(5123), p->x + re(7171), p->x + re(3075), p->y + re(7168));
fft_4_2_leaf(p->x + re(515), p->x + re(4611), p->x + re(2563), p->x + re(6659), p->x + re(7683), p->x + re(3587), p->x + re(1539), p->x + re(5635), p->y + re(7176));
fft_8_1_leaf(p->x + re(259), p->x + re(4355), p->x + re(2307), p->x + re(6403), p->x + re(1283), p->x + re(5379), p->x + re(7427), p->x + re(3331), p->y + re(7184));
fft_8_1_leaf(p->x + re(7939), p->x + re(3843), p->x + re(1795), p->x + re(5891), p->x + re(771), p->x + re(4867), p->x + re(6915), p->x + re(2819), p->y + re(7192));
fft_8_1_leaf(p->x + re(131), p->x + re(4227), p->x + re(2179), p->x + re(6275), p->x + re(1155), p->x + re(5251), p->x + re(7299), p->x + re(3203), p->y + re(7200));
fft_4_2_leaf(p->x + re(643), p->x + re(4739), p->x + re(2691), p->x + re(6787), p->x + re(7811), p->x + re(3715), p->x + re(1667), p->x + re(5763), p->y + re(7208));
fft_8_1_leaf(p->x + re(8067), p->x + re(3971), p->x + re(1923), p->x + re(6019), p->x + re(899), p->x + re(4995), p->x + re(7043), p->x + re(2947), p->y + re(7216));
fft_4_2_leaf(p->x + re(387), p->x + re(4483), p->x + re(2435), p->x + re(6531), p->x + re(7555), p->x + re(3459), p->x + re(1411), p->x + re(5507), p->y + re(7224));
fft_8_1_leaf(p->x + re(67), p->x + re(4163), p->x + re(2115), p->x + re(6211), p->x + re(1091), p->x + re(5187), p->x + re(7235), p->x + re(3139), p->y + re(7232));
fft_4_2_leaf(p->x + re(579), p->x + re(4675), p->x + re(2627), p->x + re(6723), p->x + re(7747), p->x + re(3651), p->x + re(1603), p->x + re(5699), p->y + re(7240));
fft_8_1_leaf(p->x + re(323), p->x + re(4419), p->x + re(2371), p->x + re(6467), p->x + re(1347), p->x + re(5443), p->x + re(7491), p->x + re(3395), p->y + re(7248));
fft_8_1_leaf(p->x + re(8003), p->x + re(3907), p->x + re(1859), p->x + re(5955), p->x + re(835), p->x + re(4931), p->x + re(6979), p->x + re(2883), p->y + re(7256));
fft_8_1_leaf(p->x + re(8131), p->x + re(4035), p->x + re(1987), p->x + re(6083), p->x + re(963), p->x + re(5059), p->x + re(7107), p->x + re(3011), p->y + re(7264));
fft_4_2_leaf(p->x + re(451), p->x + re(4547), p->x + re(2499), p->x + re(6595), p->x + re(7619), p->x + re(3523), p->x + re(1475), p->x + re(5571), p->y + re(7272));
fft_8_1_leaf(p->x + re(195), p->x + re(4291), p->x + re(2243), p->x + re(6339), p->x + re(1219), p->x + re(5315), p->x + re(7363), p->x + re(3267), p->y + re(7280));
fft_8_1_leaf(p->x + re(7875), p->x + re(3779), p->x + re(1731), p->x + re(5827), p->x + re(707), p->x + re(4803), p->x + re(6851), p->x + re(2755), p->y + re(7288));
fft_8_1_leaf(p->x + re(35), p->x + re(4131), p->x + re(2083), p->x + re(6179), p->x + re(1059), p->x + re(5155), p->x + re(7203), p->x + re(3107), p->y + re(7296));
fft_4_2_leaf(p->x + re(547), p->x + re(4643), p->x + re(2595), p->x + re(6691), p->x + re(7715), p->x + re(3619), p->x + re(1571), p->x + re(5667), p->y + re(7304));
fft_8_1_leaf(p->x + re(291), p->x + re(4387), p->x + re(2339), p->x + re(6435), p->x + re(1315), p->x + re(5411), p->x + re(7459), p->x + re(3363), p->y + re(7312));
fft_8_1_leaf(p->x + re(7971), p->x + re(3875), p->x + re(1827), p->x + re(5923), p->x + re(803), p->x + re(4899), p->x + re(6947), p->x + re(2851), p->y + re(7320));
fft_8_1_leaf(p->x + re(163), p->x + re(4259), p->x + re(2211), p->x + re(6307), p->x + re(1187), p->x + re(5283), p->x + re(7331), p->x + re(3235), p->y + re(7328));
fft_4_2_leaf(p->x + re(675), p->x + re(4771), p->x + re(2723), p->x + re(6819), p->x + re(7843), p->x + re(3747), p->x + re(1699), p->x + re(5795), p->y + re(7336));
fft_8_1_leaf(p->x + re(8099), p->x + re(4003), p->x + re(1955), p->x + re(6051), p->x + re(931), p->x + re(5027), p->x + re(7075), p->x + re(2979), p->y + re(7344));
fft_4_2_leaf(p->x + re(419), p->x + re(4515), p->x + re(2467), p->x + re(6563), p->x + re(7587), p->x + re(3491), p->x + re(1443), p->x + re(5539), p->y + re(7352));
fft_8_1_leaf(p->x + re(8163), p->x + re(4067), p->x + re(2019), p->x + re(6115), p->x + re(995), p->x + re(5091), p->x + re(7139), p->x + re(3043), p->y + re(7360));
fft_4_2_leaf(p->x + re(483), p->x + re(4579), p->x + re(2531), p->x + re(6627), p->x + re(7651), p->x + re(3555), p->x + re(1507), p->x + re(5603), p->y + re(7368));
fft_8_1_leaf(p->x + re(227), p->x + re(4323), p->x + re(2275), p->x + re(6371), p->x + re(1251), p->x + re(5347), p->x + re(7395), p->x + re(3299), p->y + re(7376));
fft_8_1_leaf(p->x + re(7907), p->x + re(3811), p->x + re(1763), p->x + re(5859), p->x + re(739), p->x + re(4835), p->x + re(6883), p->x + re(2787), p->y + re(7384));
fft_8_1_leaf(p->x + re(99), p->x + re(4195), p->x + re(2147), p->x + re(6243), p->x + re(1123), p->x + re(5219), p->x + re(7267), p->x + re(3171), p->y + re(7392));
fft_4_2_leaf(p->x + re(611), p->x + re(4707), p->x + re(2659), p->x + re(6755), p->x + re(7779), p->x + re(3683), p->x + re(1635), p->x + re(5731), p->y + re(7400));
fft_8_1_leaf(p->x + re(8035), p->x + re(3939), p->x + re(1891), p->x + re(5987), p->x + re(867), p->x + re(4963), p->x + re(7011), p->x + re(2915), p->y + re(7408));
fft_4_2_leaf(p->x + re(355), p->x + re(4451), p->x + re(2403), p->x + re(6499), p->x + re(7523), p->x + re(3427), p->x + re(1379), p->x + re(5475), p->y + re(7416));
fft_8_1_leaf(p->x + re(19), p->x + re(4115), p->x + re(2067), p->x + re(6163), p->x + re(1043), p->x + re(5139), p->x + re(7187), p->x + re(3091), p->y + re(7424));
fft_4_2_leaf(p->x + re(531), p->x + re(4627), p->x + re(2579), p->x + re(6675), p->x + re(7699), p->x + re(3603), p->x + re(1555), p->x + re(5651), p->y + re(7432));
fft_8_1_leaf(p->x + re(275), p->x + re(4371), p->x + re(2323), p->x + re(6419), p->x + re(1299), p->x + re(5395), p->x + re(7443), p->x + re(3347), p->y + re(7440));
fft_8_1_leaf(p->x + re(7955), p->x + re(3859), p->x + re(1811), p->x + re(5907), p->x + re(787), p->x + re(4883), p->x + re(6931), p->x + re(2835), p->y + re(7448));
fft_8_1_leaf(p->x + re(147), p->x + re(4243), p->x + re(2195), p->x + re(6291), p->x + re(1171), p->x + re(5267), p->x + re(7315), p->x + re(3219), p->y + re(7456));
fft_4_2_leaf(p->x + re(659), p->x + re(4755), p->x + re(2707), p->x + re(6803), p->x + re(7827), p->x + re(3731), p->x + re(1683), p->x + re(5779), p->y + re(7464));
fft_8_1_leaf(p->x + re(8083), p->x + re(3987), p->x + re(1939), p->x + re(6035), p->x + re(915), p->x + re(5011), p->x + re(7059), p->x + re(2963), p->y + re(7472));
fft_4_2_leaf(p->x + re(403), p->x + re(4499), p->x + re(2451), p->x + re(6547), p->x + re(7571), p->x + re(3475), p->x + re(1427), p->x + re(5523), p->y + re(7480));
fft_8_1_leaf(p->x + re(83), p->x + re(4179), p->x + re(2131), p->x + re(6227), p->x + re(1107), p->x + re(5203), p->x + re(7251), p->x + re(3155), p->y + re(7488));
fft_4_2_leaf(p->x + re(595), p->x + re(4691), p->x + re(2643), p->x + re(6739), p->x + re(7763), p->x + re(3667), p->x + re(1619), p->x + re(5715), p->y + re(7496));
fft_8_1_leaf(p->x + re(339), p->x + re(4435), p->x + re(2387), p->x + re(6483), p->x + re(1363), p->x + re(5459), p->x + re(7507), p->x + re(3411), p->y + re(7504));
fft_8_1_leaf(p->x + re(8019), p->x + re(3923), p->x + re(1875), p->x + re(5971), p->x + re(851), p->x + re(4947), p->x + re(6995), p->x + re(2899), p->y + re(7512));
fft_8_1_leaf(p->x + re(8147), p->x + re(4051), p->x + re(2003), p->x + re(6099), p->x + re(979), p->x + re(5075), p->x + re(7123), p->x + re(3027), p->y + re(7520));
fft_4_2_leaf(p->x + re(467), p->x + re(4563), p->x + re(2515), p->x + re(6611), p->x + re(7635), p->x + re(3539), p->x + re(1491), p->x + re(5587), p->y + re(7528));
fft_8_1_leaf(p->x + re(211), p->x + re(4307), p->x + re(2259), p->x + re(6355), p->x + re(1235), p->x + re(5331), p->x + re(7379), p->x + re(3283), p->y + re(7536));
fft_8_1_leaf(p->x + re(7891), p->x + re(3795), p->x + re(1747), p->x + re(5843), p->x + re(723), p->x + re(4819), p->x + re(6867), p->x + re(2771), p->y + re(7544));
fft_8_1_leaf(p->x + re(8179), p->x + re(4083), p->x + re(2035), p->x + re(6131), p->x + re(1011), p->x + re(5107), p->x + re(7155), p->x + re(3059), p->y + re(7552));
fft_4_2_leaf(p->x + re(499), p->x + re(4595), p->x + re(2547), p->x + re(6643), p->x + re(7667), p->x + re(3571), p->x + re(1523), p->x + re(5619), p->y + re(7560));
fft_8_1_leaf(p->x + re(243), p->x + re(4339), p->x + re(2291), p->x + re(6387), p->x + re(1267), p->x + re(5363), p->x + re(7411), p->x + re(3315), p->y + re(7568));
fft_8_1_leaf(p->x + re(7923), p->x + re(3827), p->x + re(1779), p->x + re(5875), p->x + re(755), p->x + re(4851), p->x + re(6899), p->x + re(2803), p->y + re(7576));
fft_8_1_leaf(p->x + re(115), p->x + re(4211), p->x + re(2163), p->x + re(6259), p->x + re(1139), p->x + re(5235), p->x + re(7283), p->x + re(3187), p->y + re(7584));
fft_4_2_leaf(p->x + re(627), p->x + re(4723), p->x + re(2675), p->x + re(6771), p->x + re(7795), p->x + re(3699), p->x + re(1651), p->x + re(5747), p->y + re(7592));
fft_8_1_leaf(p->x + re(8051), p->x + re(3955), p->x + re(1907), p->x + re(6003), p->x + re(883), p->x + re(4979), p->x + re(7027), p->x + re(2931), p->y + re(7600));
fft_4_2_leaf(p->x + re(371), p->x + re(4467), p->x + re(2419), p->x + re(6515), p->x + re(7539), p->x + re(3443), p->x + re(1395), p->x + re(5491), p->y + re(7608));
fft_8_1_leaf(p->x + re(51), p->x + re(4147), p->x + re(2099), p->x + re(6195), p->x + re(1075), p->x + re(5171), p->x + re(7219), p->x + re(3123), p->y + re(7616));
fft_4_2_leaf(p->x + re(563), p->x + re(4659), p->x + re(2611), p->x + re(6707), p->x + re(7731), p->x + re(3635), p->x + re(1587), p->x + re(5683), p->y + re(7624));
fft_8_1_leaf(p->x + re(307), p->x + re(4403), p->x + re(2355), p->x + re(6451), p->x + re(1331), p->x + re(5427), p->x + re(7475), p->x + re(3379), p->y + re(7632));
fft_8_1_leaf(p->x + re(7987), p->x + re(3891), p->x + re(1843), p->x + re(5939), p->x + re(819), p->x + re(4915), p->x + re(6963), p->x + re(2867), p->y + re(7640));
fft_8_1_leaf(p->x + re(8115), p->x + re(4019), p->x + re(1971), p->x + re(6067), p->x + re(947), p->x + re(5043), p->x + re(7091), p->x + re(2995), p->y + re(7648));
fft_4_2_leaf(p->x + re(435), p->x + re(4531), p->x + re(2483), p->x + re(6579), p->x + re(7603), p->x + re(3507), p->x + re(1459), p->x + re(5555), p->y + re(7656));
fft_8_1_leaf(p->x + re(179), p->x + re(4275), p->x + re(2227), p->x + re(6323), p->x + re(1203), p->x + re(5299), p->x + re(7347), p->x + re(3251), p->y + re(7664));
fft_8_1_leaf(p->x + re(7859), p->x + re(3763), p->x + re(1715), p->x + re(5811), p->x + re(691), p->x + re(4787), p->x + re(6835), p->x + re(2739), p->y + re(7672));
fft_8_1_leaf(p->x + re(8187), p->x + re(4091), p->x + re(2043), p->x + re(6139), p->x + re(1019), p->x + re(5115), p->x + re(7163), p->x + re(3067), p->y + re(7680));
fft_4_2_leaf(p->x + re(507), p->x + re(4603), p->x + re(2555), p->x + re(6651), p->x + re(7675), p->x + re(3579), p->x + re(1531), p->x + re(5627), p->y + re(7688));
fft_8_1_leaf(p->x + re(251), p->x + re(4347), p->x + re(2299), p->x + re(6395), p->x + re(1275), p->x + re(5371), p->x + re(7419), p->x + re(3323), p->y + re(7696));
fft_8_1_leaf(p->x + re(7931), p->x + re(3835), p->x + re(1787), p->x + re(5883), p->x + re(763), p->x + re(4859), p->x + re(6907), p->x + re(2811), p->y + re(7704));
fft_8_1_leaf(p->x + re(123), p->x + re(4219), p->x + re(2171), p->x + re(6267), p->x + re(1147), p->x + re(5243), p->x + re(7291), p->x + re(3195), p->y + re(7712));
fft_4_2_leaf(p->x + re(635), p->x + re(4731), p->x + re(2683), p->x + re(6779), p->x + re(7803), p->x + re(3707), p->x + re(1659), p->x + re(5755), p->y + re(7720));
fft_8_1_leaf(p->x + re(8059), p->x + re(3963), p->x + re(1915), p->x + re(6011), p->x + re(891), p->x + re(4987), p->x + re(7035), p->x + re(2939), p->y + re(7728));
fft_4_2_leaf(p->x + re(379), p->x + re(4475), p->x + re(2427), p->x + re(6523), p->x + re(7547), p->x + re(3451), p->x + re(1403), p->x + re(5499), p->y + re(7736));
fft_8_1_leaf(p->x + re(59), p->x + re(4155), p->x + re(2107), p->x + re(6203), p->x + re(1083), p->x + re(5179), p->x + re(7227), p->x + re(3131), p->y + re(7744));
fft_4_2_leaf(p->x + re(571), p->x + re(4667), p->x + re(2619), p->x + re(6715), p->x + re(7739), p->x + re(3643), p->x + re(1595), p->x + re(5691), p->y + re(7752));
fft_8_1_leaf(p->x + re(315), p->x + re(4411), p->x + re(2363), p->x + re(6459), p->x + re(1339), p->x + re(5435), p->x + re(7483), p->x + re(3387), p->y + re(7760));
fft_8_1_leaf(p->x + re(7995), p->x + re(3899), p->x + re(1851), p->x + re(5947), p->x + re(827), p->x + re(4923), p->x + re(6971), p->x + re(2875), p->y + re(7768));
fft_8_1_leaf(p->x + re(8123), p->x + re(4027), p->x + re(1979), p->x + re(6075), p->x + re(955), p->x + re(5051), p->x + re(7099), p->x + re(3003), p->y + re(7776));
fft_4_2_leaf(p->x + re(443), p->x + re(4539), p->x + re(2491), p->x + re(6587), p->x + re(7611), p->x + re(3515), p->x + re(1467), p->x + re(5563), p->y + re(7784));
fft_8_1_leaf(p->x + re(187), p->x + re(4283), p->x + re(2235), p->x + re(6331), p->x + re(1211), p->x + re(5307), p->x + re(7355), p->x + re(3259), p->y + re(7792));
fft_8_1_leaf(p->x + re(7867), p->x + re(3771), p->x + re(1723), p->x + re(5819), p->x + re(699), p->x + re(4795), p->x + re(6843), p->x + re(2747), p->y + re(7800));
fft_8_1_leaf(p->x + re(27), p->x + re(4123), p->x + re(2075), p->x + re(6171), p->x + re(1051), p->x + re(5147), p->x + re(7195), p->x + re(3099), p->y + re(7808));
fft_4_2_leaf(p->x + re(539), p->x + re(4635), p->x + re(2587), p->x + re(6683), p->x + re(7707), p->x + re(3611), p->x + re(1563), p->x + re(5659), p->y + re(7816));
fft_8_1_leaf(p->x + re(283), p->x + re(4379), p->x + re(2331), p->x + re(6427), p->x + re(1307), p->x + re(5403), p->x + re(7451), p->x + re(3355), p->y + re(7824));
fft_8_1_leaf(p->x + re(7963), p->x + re(3867), p->x + re(1819), p->x + re(5915), p->x + re(795), p->x + re(4891), p->x + re(6939), p->x + re(2843), p->y + re(7832));
fft_8_1_leaf(p->x + re(155), p->x + re(4251), p->x + re(2203), p->x + re(6299), p->x + re(1179), p->x + re(5275), p->x + re(7323), p->x + re(3227), p->y + re(7840));
fft_4_2_leaf(p->x + re(667), p->x + re(4763), p->x + re(2715), p->x + re(6811), p->x + re(7835), p->x + re(3739), p->x + re(1691), p->x + re(5787), p->y + re(7848));
fft_8_1_leaf(p->x + re(8091), p->x + re(3995), p->x + re(1947), p->x + re(6043), p->x + re(923), p->x + re(5019), p->x + re(7067), p->x + re(2971), p->y + re(7856));
fft_4_2_leaf(p->x + re(411), p->x + re(4507), p->x + re(2459), p->x + re(6555), p->x + re(7579), p->x + re(3483), p->x + re(1435), p->x + re(5531), p->y + re(7864));
fft_8_1_leaf(p->x + re(8155), p->x + re(4059), p->x + re(2011), p->x + re(6107), p->x + re(987), p->x + re(5083), p->x + re(7131), p->x + re(3035), p->y + re(7872));
fft_4_2_leaf(p->x + re(475), p->x + re(4571), p->x + re(2523), p->x + re(6619), p->x + re(7643), p->x + re(3547), p->x + re(1499), p->x + re(5595), p->y + re(7880));
fft_8_1_leaf(p->x + re(219), p->x + re(4315), p->x + re(2267), p->x + re(6363), p->x + re(1243), p->x + re(5339), p->x + re(7387), p->x + re(3291), p->y + re(7888));
fft_8_1_leaf(p->x + re(7899), p->x + re(3803), p->x + re(1755), p->x + re(5851), p->x + re(731), p->x + re(4827), p->x + re(6875), p->x + re(2779), p->y + re(7896));
fft_8_1_leaf(p->x + re(91), p->x + re(4187), p->x + re(2139), p->x + re(6235), p->x + re(1115), p->x + re(5211), p->x + re(7259), p->x + re(3163), p->y + re(7904));
fft_4_2_leaf(p->x + re(603), p->x + re(4699), p->x + re(2651), p->x + re(6747), p->x + re(7771), p->x + re(3675), p->x + re(1627), p->x + re(5723), p->y + re(7912));
fft_8_1_leaf(p->x + re(8027), p->x + re(3931), p->x + re(1883), p->x + re(5979), p->x + re(859), p->x + re(4955), p->x + re(7003), p->x + re(2907), p->y + re(7920));
fft_4_2_leaf(p->x + re(347), p->x + re(4443), p->x + re(2395), p->x + re(6491), p->x + re(7515), p->x + re(3419), p->x + re(1371), p->x + re(5467), p->y + re(7928));
fft_8_1_leaf(p->x + re(11), p->x + re(4107), p->x + re(2059), p->x + re(6155), p->x + re(1035), p->x + re(5131), p->x + re(7179), p->x + re(3083), p->y + re(7936));
fft_4_2_leaf(p->x + re(523), p->x + re(4619), p->x + re(2571), p->x + re(6667), p->x + re(7691), p->x + re(3595), p->x + re(1547), p->x + re(5643), p->y + re(7944));
fft_8_1_leaf(p->x + re(267), p->x + re(4363), p->x + re(2315), p->x + re(6411), p->x + re(1291), p->x + re(5387), p->x + re(7435), p->x + re(3339), p->y + re(7952));
fft_8_1_leaf(p->x + re(7947), p->x + re(3851), p->x + re(1803), p->x + re(5899), p->x + re(779), p->x + re(4875), p->x + re(6923), p->x + re(2827), p->y + re(7960));
fft_8_1_leaf(p->x + re(139), p->x + re(4235), p->x + re(2187), p->x + re(6283), p->x + re(1163), p->x + re(5259), p->x + re(7307), p->x + re(3211), p->y + re(7968));
fft_4_2_leaf(p->x + re(651), p->x + re(4747), p->x + re(2699), p->x + re(6795), p->x + re(7819), p->x + re(3723), p->x + re(1675), p->x + re(5771), p->y + re(7976));
fft_8_1_leaf(p->x + re(8075), p->x + re(3979), p->x + re(1931), p->x + re(6027), p->x + re(907), p->x + re(5003), p->x + re(7051), p->x + re(2955), p->y + re(7984));
fft_4_2_leaf(p->x + re(395), p->x + re(4491), p->x + re(2443), p->x + re(6539), p->x + re(7563), p->x + re(3467), p->x + re(1419), p->x + re(5515), p->y + re(7992));
fft_8_1_leaf(p->x + re(75), p->x + re(4171), p->x + re(2123), p->x + re(6219), p->x + re(1099), p->x + re(5195), p->x + re(7243), p->x + re(3147), p->y + re(8000));
fft_4_2_leaf(p->x + re(587), p->x + re(4683), p->x + re(2635), p->x + re(6731), p->x + re(7755), p->x + re(3659), p->x + re(1611), p->x + re(5707), p->y + re(8008));
fft_8_1_leaf(p->x + re(331), p->x + re(4427), p->x + re(2379), p->x + re(6475), p->x + re(1355), p->x + re(5451), p->x + re(7499), p->x + re(3403), p->y + re(8016));
fft_8_1_leaf(p->x + re(8011), p->x + re(3915), p->x + re(1867), p->x + re(5963), p->x + re(843), p->x + re(4939), p->x + re(6987), p->x + re(2891), p->y + re(8024));
fft_8_1_leaf(p->x + re(8139), p->x + re(4043), p->x + re(1995), p->x + re(6091), p->x + re(971), p->x + re(5067), p->x + re(7115), p->x + re(3019), p->y + re(8032));
fft_4_2_leaf(p->x + re(459), p->x + re(4555), p->x + re(2507), p->x + re(6603), p->x + re(7627), p->x + re(3531), p->x + re(1483), p->x + re(5579), p->y + re(8040));
fft_8_1_leaf(p->x + re(203), p->x + re(4299), p->x + re(2251), p->x + re(6347), p->x + re(1227), p->x + re(5323), p->x + re(7371), p->x + re(3275), p->y + re(8048));
fft_8_1_leaf(p->x + re(7883), p->x + re(3787), p->x + re(1739), p->x + re(5835), p->x + re(715), p->x + re(4811), p->x + re(6859), p->x + re(2763), p->y + re(8056));
fft_8_1_leaf(p->x + re(8171), p->x + re(4075), p->x + re(2027), p->x + re(6123), p->x + re(1003), p->x + re(5099), p->x + re(7147), p->x + re(3051), p->y + re(8064));
fft_4_2_leaf(p->x + re(491), p->x + re(4587), p->x + re(2539), p->x + re(6635), p->x + re(7659), p->x + re(3563), p->x + re(1515), p->x + re(5611), p->y + re(8072));
fft_8_1_leaf(p->x + re(235), p->x + re(4331), p->x + re(2283), p->x + re(6379), p->x + re(1259), p->x + re(5355), p->x + re(7403), p->x + re(3307), p->y + re(8080));
fft_8_1_leaf(p->x + re(7915), p->x + re(3819), p->x + re(1771), p->x + re(5867), p->x + re(747), p->x + re(4843), p->x + re(6891), p->x + re(2795), p->y + re(8088));
fft_8_1_leaf(p->x + re(107), p->x + re(4203), p->x + re(2155), p->x + re(6251), p->x + re(1131), p->x + re(5227), p->x + re(7275), p->x + re(3179), p->y + re(8096));
fft_4_2_leaf(p->x + re(619), p->x + re(4715), p->x + re(2667), p->x + re(6763), p->x + re(7787), p->x + re(3691), p->x + re(1643), p->x + re(5739), p->y + re(8104));
fft_8_1_leaf(p->x + re(8043), p->x + re(3947), p->x + re(1899), p->x + re(5995), p->x + re(875), p->x + re(4971), p->x + re(7019), p->x + re(2923), p->y + re(8112));
fft_4_2_leaf(p->x + re(363), p->x + re(4459), p->x + re(2411), p->x + re(6507), p->x + re(7531), p->x + re(3435), p->x + re(1387), p->x + re(5483), p->y + re(8120));
fft_8_1_leaf(p->x + re(43), p->x + re(4139), p->x + re(2091), p->x + re(6187), p->x + re(1067), p->x + re(5163), p->x + re(7211), p->x + re(3115), p->y + re(8128));
fft_4_2_leaf(p->x + re(555), p->x + re(4651), p->x + re(2603), p->x + re(6699), p->x + re(7723), p->x + re(3627), p->x + re(1579), p->x + re(5675), p->y + re(8136));
fft_8_1_leaf(p->x + re(299), p->x + re(4395), p->x + re(2347), p->x + re(6443), p->x + re(1323), p->x + re(5419), p->x + re(7467), p->x + re(3371), p->y + re(8144));
fft_8_1_leaf(p->x + re(7979), p->x + re(3883), p->x + re(1835), p->x + re(5931), p->x + re(811), p->x + re(4907), p->x + re(6955), p->x + re(2859), p->y + re(8152));
fft_8_1_leaf(p->x + re(8107), p->x + re(4011), p->x + re(1963), p->x + re(6059), p->x + re(939), p->x + re(5035), p->x + re(7083), p->x + re(2987), p->y + re(8160));
fft_4_2_leaf(p->x + re(427), p->x + re(4523), p->x + re(2475), p->x + re(6571), p->x + re(7595), p->x + re(3499), p->x + re(1451), p->x + re(5547), p->y + re(8168));
fft_8_1_leaf(p->x + re(171), p->x + re(4267), p->x + re(2219), p->x + re(6315), p->x + re(1195), p->x + re(5291), p->x + re(7339), p->x + re(3243), p->y + re(8176));
fft_8_1_leaf(p->x + re(7851), p->x + re(3755), p->x + re(1707), p->x + re(5803), p->x + re(683), p->x + re(4779), p->x + re(6827), p->x + re(2731), p->y + re(8184));
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
fft_n_(4, p->y + re(256), p->y + re(260), p->y + re(264), p->y + re(268), p->wre_table[1], p->wim_table[1]);
fft_n_(8, p->y + re(256), p->y + re(264), p->y + re(272), p->y + re(280), p->wre_table[2], p->wim_table[2]);
fft_n_(4, p->y + re(288), p->y + re(292), p->y + re(296), p->y + re(300), p->wre_table[1], p->wim_table[1]);
fft_n_(4, p->y + re(304), p->y + re(308), p->y + re(312), p->y + re(316), p->wre_table[1], p->wim_table[1]);
fft_n_(16, p->y + re(256), p->y + re(272), p->y + re(288), p->y + re(304), p->wre_table[3], p->wim_table[3]);
fft_n_(4, p->y + re(320), p->y + re(324), p->y + re(328), p->y + re(332), p->wre_table[1], p->wim_table[1]);
fft_n_(8, p->y + re(320), p->y + re(328), p->y + re(336), p->y + re(344), p->wre_table[2], p->wim_table[2]);
fft_n_(4, p->y + re(352), p->y + re(356), p->y + re(360), p->y + re(364), p->wre_table[1], p->wim_table[1]);
fft_n_(8, p->y + re(352), p->y + re(360), p->y + re(368), p->y + re(376), p->wre_table[2], p->wim_table[2]);
fft_n_(32, p->y + re(256), p->y + re(288), p->y + re(320), p->y + re(352), p->wre_table[4], p->wim_table[4]);
fft_n_(4, p->y + re(384), p->y + re(388), p->y + re(392), p->y + re(396), p->wre_table[1], p->wim_table[1]);
fft_n_(8, p->y + re(384), p->y + re(392), p->y + re(400), p->y + re(408), p->wre_table[2], p->wim_table[2]);
fft_n_(4, p->y + re(416), p->y + re(420), p->y + re(424), p->y + re(428), p->wre_table[1], p->wim_table[1]);
fft_n_(4, p->y + re(432), p->y + re(436), p->y + re(440), p->y + re(444), p->wre_table[1], p->wim_table[1]);
fft_n_(16, p->y + re(384), p->y + re(400), p->y + re(416), p->y + re(432), p->wre_table[3], p->wim_table[3]);
fft_n_(4, p->y + re(448), p->y + re(452), p->y + re(456), p->y + re(460), p->wre_table[1], p->wim_table[1]);
fft_n_(8, p->y + re(448), p->y + re(456), p->y + re(464), p->y + re(472), p->wre_table[2], p->wim_table[2]);
fft_n_(4, p->y + re(480), p->y + re(484), p->y + re(488), p->y + re(492), p->wre_table[1], p->wim_table[1]);
fft_n_(8, p->y + re(480), p->y + re(488), p->y + re(496), p->y + re(504), p->wre_table[2], p->wim_table[2]);
fft_n_(32, p->y + re(384), p->y + re(416), p->y + re(448), p->y + re(480), p->wre_table[4], p->wim_table[4]);
fft_n_(128, p->y + re(0), p->y + re(128), p->y + re(256), p->y + re(384), p->wre_table[6], p->wim_table[6]);
fft_n_(4, p->y + re(512), p->y + re(516), p->y + re(520), p->y + re(524), p->wre_table[1], p->wim_table[1]);
fft_n_(8, p->y + re(512), p->y + re(520), p->y + re(528), p->y + re(536), p->wre_table[2], p->wim_table[2]);
fft_n_(4, p->y + re(544), p->y + re(548), p->y + re(552), p->y + re(556), p->wre_table[1], p->wim_table[1]);
fft_n_(4, p->y + re(560), p->y + re(564), p->y + re(568), p->y + re(572), p->wre_table[1], p->wim_table[1]);
fft_n_(16, p->y + re(512), p->y + re(528), p->y + re(544), p->y + re(560), p->wre_table[3], p->wim_table[3]);
fft_n_(4, p->y + re(576), p->y + re(580), p->y + re(584), p->y + re(588), p->wre_table[1], p->wim_table[1]);
fft_n_(8, p->y + re(576), p->y + re(584), p->y + re(592), p->y + re(600), p->wre_table[2], p->wim_table[2]);
fft_n_(4, p->y + re(608), p->y + re(612), p->y + re(616), p->y + re(620), p->wre_table[1], p->wim_table[1]);
fft_n_(8, p->y + re(608), p->y + re(616), p->y + re(624), p->y + re(632), p->wre_table[2], p->wim_table[2]);
fft_n_(32, p->y + re(512), p->y + re(544), p->y + re(576), p->y + re(608), p->wre_table[4], p->wim_table[4]);
fft_n_(4, p->y + re(640), p->y + re(644), p->y + re(648), p->y + re(652), p->wre_table[1], p->wim_table[1]);
fft_n_(8, p->y + re(640), p->y + re(648), p->y + re(656), p->y + re(664), p->wre_table[2], p->wim_table[2]);
fft_n_(4, p->y + re(672), p->y + re(676), p->y + re(680), p->y + re(684), p->wre_table[1], p->wim_table[1]);
fft_n_(4, p->y + re(688), p->y + re(692), p->y + re(696), p->y + re(700), p->wre_table[1], p->wim_table[1]);
fft_n_(16, p->y + re(640), p->y + re(656), p->y + re(672), p->y + re(688), p->wre_table[3], p->wim_table[3]);
fft_n_(4, p->y + re(704), p->y + re(708), p->y + re(712), p->y + re(716), p->wre_table[1], p->wim_table[1]);
fft_n_(8, p->y + re(704), p->y + re(712), p->y + re(720), p->y + re(728), p->wre_table[2], p->wim_table[2]);
fft_n_(4, p->y + re(736), p->y + re(740), p->y + re(744), p->y + re(748), p->wre_table[1], p->wim_table[1]);
fft_n_(4, p->y + re(752), p->y + re(756), p->y + re(760), p->y + re(764), p->wre_table[1], p->wim_table[1]);
fft_n_(16, p->y + re(704), p->y + re(720), p->y + re(736), p->y + re(752), p->wre_table[3], p->wim_table[3]);
fft_n_(64, p->y + re(512), p->y + re(576), p->y + re(640), p->y + re(704), p->wre_table[5], p->wim_table[5]);
fft_n_(4, p->y + re(768), p->y + re(772), p->y + re(776), p->y + re(780), p->wre_table[1], p->wim_table[1]);
fft_n_(8, p->y + re(768), p->y + re(776), p->y + re(784), p->y + re(792), p->wre_table[2], p->wim_table[2]);
fft_n_(4, p->y + re(800), p->y + re(804), p->y + re(808), p->y + re(812), p->wre_table[1], p->wim_table[1]);
fft_n_(4, p->y + re(816), p->y + re(820), p->y + re(824), p->y + re(828), p->wre_table[1], p->wim_table[1]);
fft_n_(16, p->y + re(768), p->y + re(784), p->y + re(800), p->y + re(816), p->wre_table[3], p->wim_table[3]);
fft_n_(4, p->y + re(832), p->y + re(836), p->y + re(840), p->y + re(844), p->wre_table[1], p->wim_table[1]);
fft_n_(8, p->y + re(832), p->y + re(840), p->y + re(848), p->y + re(856), p->wre_table[2], p->wim_table[2]);
fft_n_(4, p->y + re(864), p->y + re(868), p->y + re(872), p->y + re(876), p->wre_table[1], p->wim_table[1]);
fft_n_(8, p->y + re(864), p->y + re(872), p->y + re(880), p->y + re(888), p->wre_table[2], p->wim_table[2]);
fft_n_(32, p->y + re(768), p->y + re(800), p->y + re(832), p->y + re(864), p->wre_table[4], p->wim_table[4]);
fft_n_(4, p->y + re(896), p->y + re(900), p->y + re(904), p->y + re(908), p->wre_table[1], p->wim_table[1]);
fft_n_(8, p->y + re(896), p->y + re(904), p->y + re(912), p->y + re(920), p->wre_table[2], p->wim_table[2]);
fft_n_(4, p->y + re(928), p->y + re(932), p->y + re(936), p->y + re(940), p->wre_table[1], p->wim_table[1]);
fft_n_(4, p->y + re(944), p->y + re(948), p->y + re(952), p->y + re(956), p->wre_table[1], p->wim_table[1]);
fft_n_(16, p->y + re(896), p->y + re(912), p->y + re(928), p->y + re(944), p->wre_table[3], p->wim_table[3]);
fft_n_(4, p->y + re(960), p->y + re(964), p->y + re(968), p->y + re(972), p->wre_table[1], p->wim_table[1]);
fft_n_(8, p->y + re(960), p->y + re(968), p->y + re(976), p->y + re(984), p->wre_table[2], p->wim_table[2]);
fft_n_(4, p->y + re(992), p->y + re(996), p->y + re(1000), p->y + re(1004), p->wre_table[1], p->wim_table[1]);
fft_n_(4, p->y + re(1008), p->y + re(1012), p->y + re(1016), p->y + re(1020), p->wre_table[1], p->wim_table[1]);
fft_n_(16, p->y + re(960), p->y + re(976), p->y + re(992), p->y + re(1008), p->wre_table[3], p->wim_table[3]);
fft_n_(64, p->y + re(768), p->y + re(832), p->y + re(896), p->y + re(960), p->wre_table[5], p->wim_table[5]);
fft_n_(256, p->y + re(0), p->y + re(256), p->y + re(512), p->y + re(768), p->wre_table[7], p->wim_table[7]);
fft_n_(4, p->y + re(1024), p->y + re(1028), p->y + re(1032), p->y + re(1036), p->wre_table[1], p->wim_table[1]);
fft_n_(8, p->y + re(1024), p->y + re(1032), p->y + re(1040), p->y + re(1048), p->wre_table[2], p->wim_table[2]);
fft_n_(4, p->y + re(1056), p->y + re(1060), p->y + re(1064), p->y + re(1068), p->wre_table[1], p->wim_table[1]);
fft_n_(4, p->y + re(1072), p->y + re(1076), p->y + re(1080), p->y + re(1084), p->wre_table[1], p->wim_table[1]);
fft_n_(16, p->y + re(1024), p->y + re(1040), p->y + re(1056), p->y + re(1072), p->wre_table[3], p->wim_table[3]);
fft_n_(4, p->y + re(1088), p->y + re(1092), p->y + re(1096), p->y + re(1100), p->wre_table[1], p->wim_table[1]);
fft_n_(8, p->y + re(1088), p->y + re(1096), p->y + re(1104), p->y + re(1112), p->wre_table[2], p->wim_table[2]);
fft_n_(4, p->y + re(1120), p->y + re(1124), p->y + re(1128), p->y + re(1132), p->wre_table[1], p->wim_table[1]);
fft_n_(8, p->y + re(1120), p->y + re(1128), p->y + re(1136), p->y + re(1144), p->wre_table[2], p->wim_table[2]);
fft_n_(32, p->y + re(1024), p->y + re(1056), p->y + re(1088), p->y + re(1120), p->wre_table[4], p->wim_table[4]);
fft_n_(4, p->y + re(1152), p->y + re(1156), p->y + re(1160), p->y + re(1164), p->wre_table[1], p->wim_table[1]);
fft_n_(8, p->y + re(1152), p->y + re(1160), p->y + re(1168), p->y + re(1176), p->wre_table[2], p->wim_table[2]);
fft_n_(4, p->y + re(1184), p->y + re(1188), p->y + re(1192), p->y + re(1196), p->wre_table[1], p->wim_table[1]);
fft_n_(4, p->y + re(1200), p->y + re(1204), p->y + re(1208), p->y + re(1212), p->wre_table[1], p->wim_table[1]);
fft_n_(16, p->y + re(1152), p->y + re(1168), p->y + re(1184), p->y + re(1200), p->wre_table[3], p->wim_table[3]);
fft_n_(4, p->y + re(1216), p->y + re(1220), p->y + re(1224), p->y + re(1228), p->wre_table[1], p->wim_table[1]);
fft_n_(8, p->y + re(1216), p->y + re(1224), p->y + re(1232), p->y + re(1240), p->wre_table[2], p->wim_table[2]);
fft_n_(4, p->y + re(1248), p->y + re(1252), p->y + re(1256), p->y + re(1260), p->wre_table[1], p->wim_table[1]);
fft_n_(4, p->y + re(1264), p->y + re(1268), p->y + re(1272), p->y + re(1276), p->wre_table[1], p->wim_table[1]);
fft_n_(16, p->y + re(1216), p->y + re(1232), p->y + re(1248), p->y + re(1264), p->wre_table[3], p->wim_table[3]);
fft_n_(64, p->y + re(1024), p->y + re(1088), p->y + re(1152), p->y + re(1216), p->wre_table[5], p->wim_table[5]);
fft_n_(4, p->y + re(1280), p->y + re(1284), p->y + re(1288), p->y + re(1292), p->wre_table[1], p->wim_table[1]);
fft_n_(8, p->y + re(1280), p->y + re(1288), p->y + re(1296), p->y + re(1304), p->wre_table[2], p->wim_table[2]);
fft_n_(4, p->y + re(1312), p->y + re(1316), p->y + re(1320), p->y + re(1324), p->wre_table[1], p->wim_table[1]);
fft_n_(4, p->y + re(1328), p->y + re(1332), p->y + re(1336), p->y + re(1340), p->wre_table[1], p->wim_table[1]);
fft_n_(16, p->y + re(1280), p->y + re(1296), p->y + re(1312), p->y + re(1328), p->wre_table[3], p->wim_table[3]);
fft_n_(4, p->y + re(1344), p->y + re(1348), p->y + re(1352), p->y + re(1356), p->wre_table[1], p->wim_table[1]);
fft_n_(8, p->y + re(1344), p->y + re(1352), p->y + re(1360), p->y + re(1368), p->wre_table[2], p->wim_table[2]);
fft_n_(4, p->y + re(1376), p->y + re(1380), p->y + re(1384), p->y + re(1388), p->wre_table[1], p->wim_table[1]);
fft_n_(8, p->y + re(1376), p->y + re(1384), p->y + re(1392), p->y + re(1400), p->wre_table[2], p->wim_table[2]);
fft_n_(32, p->y + re(1280), p->y + re(1312), p->y + re(1344), p->y + re(1376), p->wre_table[4], p->wim_table[4]);
fft_n_(4, p->y + re(1408), p->y + re(1412), p->y + re(1416), p->y + re(1420), p->wre_table[1], p->wim_table[1]);
fft_n_(8, p->y + re(1408), p->y + re(1416), p->y + re(1424), p->y + re(1432), p->wre_table[2], p->wim_table[2]);
fft_n_(4, p->y + re(1440), p->y + re(1444), p->y + re(1448), p->y + re(1452), p->wre_table[1], p->wim_table[1]);
fft_n_(4, p->y + re(1456), p->y + re(1460), p->y + re(1464), p->y + re(1468), p->wre_table[1], p->wim_table[1]);
fft_n_(16, p->y + re(1408), p->y + re(1424), p->y + re(1440), p->y + re(1456), p->wre_table[3], p->wim_table[3]);
fft_n_(4, p->y + re(1472), p->y + re(1476), p->y + re(1480), p->y + re(1484), p->wre_table[1], p->wim_table[1]);
fft_n_(8, p->y + re(1472), p->y + re(1480), p->y + re(1488), p->y + re(1496), p->wre_table[2], p->wim_table[2]);
fft_n_(4, p->y + re(1504), p->y + re(1508), p->y + re(1512), p->y + re(1516), p->wre_table[1], p->wim_table[1]);
fft_n_(8, p->y + re(1504), p->y + re(1512), p->y + re(1520), p->y + re(1528), p->wre_table[2], p->wim_table[2]);
fft_n_(32, p->y + re(1408), p->y + re(1440), p->y + re(1472), p->y + re(1504), p->wre_table[4], p->wim_table[4]);
fft_n_(128, p->y + re(1024), p->y + re(1152), p->y + re(1280), p->y + re(1408), p->wre_table[6], p->wim_table[6]);
fft_n_(4, p->y + re(1536), p->y + re(1540), p->y + re(1544), p->y + re(1548), p->wre_table[1], p->wim_table[1]);
fft_n_(8, p->y + re(1536), p->y + re(1544), p->y + re(1552), p->y + re(1560), p->wre_table[2], p->wim_table[2]);
fft_n_(4, p->y + re(1568), p->y + re(1572), p->y + re(1576), p->y + re(1580), p->wre_table[1], p->wim_table[1]);
fft_n_(4, p->y + re(1584), p->y + re(1588), p->y + re(1592), p->y + re(1596), p->wre_table[1], p->wim_table[1]);
fft_n_(16, p->y + re(1536), p->y + re(1552), p->y + re(1568), p->y + re(1584), p->wre_table[3], p->wim_table[3]);
fft_n_(4, p->y + re(1600), p->y + re(1604), p->y + re(1608), p->y + re(1612), p->wre_table[1], p->wim_table[1]);
fft_n_(8, p->y + re(1600), p->y + re(1608), p->y + re(1616), p->y + re(1624), p->wre_table[2], p->wim_table[2]);
fft_n_(4, p->y + re(1632), p->y + re(1636), p->y + re(1640), p->y + re(1644), p->wre_table[1], p->wim_table[1]);
fft_n_(8, p->y + re(1632), p->y + re(1640), p->y + re(1648), p->y + re(1656), p->wre_table[2], p->wim_table[2]);
fft_n_(32, p->y + re(1536), p->y + re(1568), p->y + re(1600), p->y + re(1632), p->wre_table[4], p->wim_table[4]);
fft_n_(4, p->y + re(1664), p->y + re(1668), p->y + re(1672), p->y + re(1676), p->wre_table[1], p->wim_table[1]);
fft_n_(8, p->y + re(1664), p->y + re(1672), p->y + re(1680), p->y + re(1688), p->wre_table[2], p->wim_table[2]);
fft_n_(4, p->y + re(1696), p->y + re(1700), p->y + re(1704), p->y + re(1708), p->wre_table[1], p->wim_table[1]);
fft_n_(4, p->y + re(1712), p->y + re(1716), p->y + re(1720), p->y + re(1724), p->wre_table[1], p->wim_table[1]);
fft_n_(16, p->y + re(1664), p->y + re(1680), p->y + re(1696), p->y + re(1712), p->wre_table[3], p->wim_table[3]);
fft_n_(4, p->y + re(1728), p->y + re(1732), p->y + re(1736), p->y + re(1740), p->wre_table[1], p->wim_table[1]);
fft_n_(8, p->y + re(1728), p->y + re(1736), p->y + re(1744), p->y + re(1752), p->wre_table[2], p->wim_table[2]);
fft_n_(4, p->y + re(1760), p->y + re(1764), p->y + re(1768), p->y + re(1772), p->wre_table[1], p->wim_table[1]);
fft_n_(4, p->y + re(1776), p->y + re(1780), p->y + re(1784), p->y + re(1788), p->wre_table[1], p->wim_table[1]);
fft_n_(16, p->y + re(1728), p->y + re(1744), p->y + re(1760), p->y + re(1776), p->wre_table[3], p->wim_table[3]);
fft_n_(64, p->y + re(1536), p->y + re(1600), p->y + re(1664), p->y + re(1728), p->wre_table[5], p->wim_table[5]);
fft_n_(4, p->y + re(1792), p->y + re(1796), p->y + re(1800), p->y + re(1804), p->wre_table[1], p->wim_table[1]);
fft_n_(8, p->y + re(1792), p->y + re(1800), p->y + re(1808), p->y + re(1816), p->wre_table[2], p->wim_table[2]);
fft_n_(4, p->y + re(1824), p->y + re(1828), p->y + re(1832), p->y + re(1836), p->wre_table[1], p->wim_table[1]);
fft_n_(4, p->y + re(1840), p->y + re(1844), p->y + re(1848), p->y + re(1852), p->wre_table[1], p->wim_table[1]);
fft_n_(16, p->y + re(1792), p->y + re(1808), p->y + re(1824), p->y + re(1840), p->wre_table[3], p->wim_table[3]);
fft_n_(4, p->y + re(1856), p->y + re(1860), p->y + re(1864), p->y + re(1868), p->wre_table[1], p->wim_table[1]);
fft_n_(8, p->y + re(1856), p->y + re(1864), p->y + re(1872), p->y + re(1880), p->wre_table[2], p->wim_table[2]);
fft_n_(4, p->y + re(1888), p->y + re(1892), p->y + re(1896), p->y + re(1900), p->wre_table[1], p->wim_table[1]);
fft_n_(8, p->y + re(1888), p->y + re(1896), p->y + re(1904), p->y + re(1912), p->wre_table[2], p->wim_table[2]);
fft_n_(32, p->y + re(1792), p->y + re(1824), p->y + re(1856), p->y + re(1888), p->wre_table[4], p->wim_table[4]);
fft_n_(4, p->y + re(1920), p->y + re(1924), p->y + re(1928), p->y + re(1932), p->wre_table[1], p->wim_table[1]);
fft_n_(8, p->y + re(1920), p->y + re(1928), p->y + re(1936), p->y + re(1944), p->wre_table[2], p->wim_table[2]);
fft_n_(4, p->y + re(1952), p->y + re(1956), p->y + re(1960), p->y + re(1964), p->wre_table[1], p->wim_table[1]);
fft_n_(4, p->y + re(1968), p->y + re(1972), p->y + re(1976), p->y + re(1980), p->wre_table[1], p->wim_table[1]);
fft_n_(16, p->y + re(1920), p->y + re(1936), p->y + re(1952), p->y + re(1968), p->wre_table[3], p->wim_table[3]);
fft_n_(4, p->y + re(1984), p->y + re(1988), p->y + re(1992), p->y + re(1996), p->wre_table[1], p->wim_table[1]);
fft_n_(8, p->y + re(1984), p->y + re(1992), p->y + re(2000), p->y + re(2008), p->wre_table[2], p->wim_table[2]);
fft_n_(4, p->y + re(2016), p->y + re(2020), p->y + re(2024), p->y + re(2028), p->wre_table[1], p->wim_table[1]);
fft_n_(8, p->y + re(2016), p->y + re(2024), p->y + re(2032), p->y + re(2040), p->wre_table[2], p->wim_table[2]);
fft_n_(32, p->y + re(1920), p->y + re(1952), p->y + re(1984), p->y + re(2016), p->wre_table[4], p->wim_table[4]);
fft_n_(128, p->y + re(1536), p->y + re(1664), p->y + re(1792), p->y + re(1920), p->wre_table[6], p->wim_table[6]);
fft_n_(512, p->y + re(0), p->y + re(512), p->y + re(1024), p->y + re(1536), p->wre_table[8], p->wim_table[8]);
fft_n_(4, p->y + re(2048), p->y + re(2052), p->y + re(2056), p->y + re(2060), p->wre_table[1], p->wim_table[1]);
fft_n_(8, p->y + re(2048), p->y + re(2056), p->y + re(2064), p->y + re(2072), p->wre_table[2], p->wim_table[2]);
fft_n_(4, p->y + re(2080), p->y + re(2084), p->y + re(2088), p->y + re(2092), p->wre_table[1], p->wim_table[1]);
fft_n_(4, p->y + re(2096), p->y + re(2100), p->y + re(2104), p->y + re(2108), p->wre_table[1], p->wim_table[1]);
fft_n_(16, p->y + re(2048), p->y + re(2064), p->y + re(2080), p->y + re(2096), p->wre_table[3], p->wim_table[3]);
fft_n_(4, p->y + re(2112), p->y + re(2116), p->y + re(2120), p->y + re(2124), p->wre_table[1], p->wim_table[1]);
fft_n_(8, p->y + re(2112), p->y + re(2120), p->y + re(2128), p->y + re(2136), p->wre_table[2], p->wim_table[2]);
fft_n_(4, p->y + re(2144), p->y + re(2148), p->y + re(2152), p->y + re(2156), p->wre_table[1], p->wim_table[1]);
fft_n_(8, p->y + re(2144), p->y + re(2152), p->y + re(2160), p->y + re(2168), p->wre_table[2], p->wim_table[2]);
fft_n_(32, p->y + re(2048), p->y + re(2080), p->y + re(2112), p->y + re(2144), p->wre_table[4], p->wim_table[4]);
fft_n_(4, p->y + re(2176), p->y + re(2180), p->y + re(2184), p->y + re(2188), p->wre_table[1], p->wim_table[1]);
fft_n_(8, p->y + re(2176), p->y + re(2184), p->y + re(2192), p->y + re(2200), p->wre_table[2], p->wim_table[2]);
fft_n_(4, p->y + re(2208), p->y + re(2212), p->y + re(2216), p->y + re(2220), p->wre_table[1], p->wim_table[1]);
fft_n_(4, p->y + re(2224), p->y + re(2228), p->y + re(2232), p->y + re(2236), p->wre_table[1], p->wim_table[1]);
fft_n_(16, p->y + re(2176), p->y + re(2192), p->y + re(2208), p->y + re(2224), p->wre_table[3], p->wim_table[3]);
fft_n_(4, p->y + re(2240), p->y + re(2244), p->y + re(2248), p->y + re(2252), p->wre_table[1], p->wim_table[1]);
fft_n_(8, p->y + re(2240), p->y + re(2248), p->y + re(2256), p->y + re(2264), p->wre_table[2], p->wim_table[2]);
fft_n_(4, p->y + re(2272), p->y + re(2276), p->y + re(2280), p->y + re(2284), p->wre_table[1], p->wim_table[1]);
fft_n_(4, p->y + re(2288), p->y + re(2292), p->y + re(2296), p->y + re(2300), p->wre_table[1], p->wim_table[1]);
fft_n_(16, p->y + re(2240), p->y + re(2256), p->y + re(2272), p->y + re(2288), p->wre_table[3], p->wim_table[3]);
fft_n_(64, p->y + re(2048), p->y + re(2112), p->y + re(2176), p->y + re(2240), p->wre_table[5], p->wim_table[5]);
fft_n_(4, p->y + re(2304), p->y + re(2308), p->y + re(2312), p->y + re(2316), p->wre_table[1], p->wim_table[1]);
fft_n_(8, p->y + re(2304), p->y + re(2312), p->y + re(2320), p->y + re(2328), p->wre_table[2], p->wim_table[2]);
fft_n_(4, p->y + re(2336), p->y + re(2340), p->y + re(2344), p->y + re(2348), p->wre_table[1], p->wim_table[1]);
fft_n_(4, p->y + re(2352), p->y + re(2356), p->y + re(2360), p->y + re(2364), p->wre_table[1], p->wim_table[1]);
fft_n_(16, p->y + re(2304), p->y + re(2320), p->y + re(2336), p->y + re(2352), p->wre_table[3], p->wim_table[3]);
fft_n_(4, p->y + re(2368), p->y + re(2372), p->y + re(2376), p->y + re(2380), p->wre_table[1], p->wim_table[1]);
fft_n_(8, p->y + re(2368), p->y + re(2376), p->y + re(2384), p->y + re(2392), p->wre_table[2], p->wim_table[2]);
fft_n_(4, p->y + re(2400), p->y + re(2404), p->y + re(2408), p->y + re(2412), p->wre_table[1], p->wim_table[1]);
fft_n_(8, p->y + re(2400), p->y + re(2408), p->y + re(2416), p->y + re(2424), p->wre_table[2], p->wim_table[2]);
fft_n_(32, p->y + re(2304), p->y + re(2336), p->y + re(2368), p->y + re(2400), p->wre_table[4], p->wim_table[4]);
fft_n_(4, p->y + re(2432), p->y + re(2436), p->y + re(2440), p->y + re(2444), p->wre_table[1], p->wim_table[1]);
fft_n_(8, p->y + re(2432), p->y + re(2440), p->y + re(2448), p->y + re(2456), p->wre_table[2], p->wim_table[2]);
fft_n_(4, p->y + re(2464), p->y + re(2468), p->y + re(2472), p->y + re(2476), p->wre_table[1], p->wim_table[1]);
fft_n_(4, p->y + re(2480), p->y + re(2484), p->y + re(2488), p->y + re(2492), p->wre_table[1], p->wim_table[1]);
fft_n_(16, p->y + re(2432), p->y + re(2448), p->y + re(2464), p->y + re(2480), p->wre_table[3], p->wim_table[3]);
fft_n_(4, p->y + re(2496), p->y + re(2500), p->y + re(2504), p->y + re(2508), p->wre_table[1], p->wim_table[1]);
fft_n_(8, p->y + re(2496), p->y + re(2504), p->y + re(2512), p->y + re(2520), p->wre_table[2], p->wim_table[2]);
fft_n_(4, p->y + re(2528), p->y + re(2532), p->y + re(2536), p->y + re(2540), p->wre_table[1], p->wim_table[1]);
fft_n_(8, p->y + re(2528), p->y + re(2536), p->y + re(2544), p->y + re(2552), p->wre_table[2], p->wim_table[2]);
fft_n_(32, p->y + re(2432), p->y + re(2464), p->y + re(2496), p->y + re(2528), p->wre_table[4], p->wim_table[4]);
fft_n_(128, p->y + re(2048), p->y + re(2176), p->y + re(2304), p->y + re(2432), p->wre_table[6], p->wim_table[6]);
fft_n_(4, p->y + re(2560), p->y + re(2564), p->y + re(2568), p->y + re(2572), p->wre_table[1], p->wim_table[1]);
fft_n_(8, p->y + re(2560), p->y + re(2568), p->y + re(2576), p->y + re(2584), p->wre_table[2], p->wim_table[2]);
fft_n_(4, p->y + re(2592), p->y + re(2596), p->y + re(2600), p->y + re(2604), p->wre_table[1], p->wim_table[1]);
fft_n_(4, p->y + re(2608), p->y + re(2612), p->y + re(2616), p->y + re(2620), p->wre_table[1], p->wim_table[1]);
fft_n_(16, p->y + re(2560), p->y + re(2576), p->y + re(2592), p->y + re(2608), p->wre_table[3], p->wim_table[3]);
fft_n_(4, p->y + re(2624), p->y + re(2628), p->y + re(2632), p->y + re(2636), p->wre_table[1], p->wim_table[1]);
fft_n_(8, p->y + re(2624), p->y + re(2632), p->y + re(2640), p->y + re(2648), p->wre_table[2], p->wim_table[2]);
fft_n_(4, p->y + re(2656), p->y + re(2660), p->y + re(2664), p->y + re(2668), p->wre_table[1], p->wim_table[1]);
fft_n_(8, p->y + re(2656), p->y + re(2664), p->y + re(2672), p->y + re(2680), p->wre_table[2], p->wim_table[2]);
fft_n_(32, p->y + re(2560), p->y + re(2592), p->y + re(2624), p->y + re(2656), p->wre_table[4], p->wim_table[4]);
fft_n_(4, p->y + re(2688), p->y + re(2692), p->y + re(2696), p->y + re(2700), p->wre_table[1], p->wim_table[1]);
fft_n_(8, p->y + re(2688), p->y + re(2696), p->y + re(2704), p->y + re(2712), p->wre_table[2], p->wim_table[2]);
fft_n_(4, p->y + re(2720), p->y + re(2724), p->y + re(2728), p->y + re(2732), p->wre_table[1], p->wim_table[1]);
fft_n_(4, p->y + re(2736), p->y + re(2740), p->y + re(2744), p->y + re(2748), p->wre_table[1], p->wim_table[1]);
fft_n_(16, p->y + re(2688), p->y + re(2704), p->y + re(2720), p->y + re(2736), p->wre_table[3], p->wim_table[3]);
fft_n_(4, p->y + re(2752), p->y + re(2756), p->y + re(2760), p->y + re(2764), p->wre_table[1], p->wim_table[1]);
fft_n_(8, p->y + re(2752), p->y + re(2760), p->y + re(2768), p->y + re(2776), p->wre_table[2], p->wim_table[2]);
fft_n_(4, p->y + re(2784), p->y + re(2788), p->y + re(2792), p->y + re(2796), p->wre_table[1], p->wim_table[1]);
fft_n_(4, p->y + re(2800), p->y + re(2804), p->y + re(2808), p->y + re(2812), p->wre_table[1], p->wim_table[1]);
fft_n_(16, p->y + re(2752), p->y + re(2768), p->y + re(2784), p->y + re(2800), p->wre_table[3], p->wim_table[3]);
fft_n_(64, p->y + re(2560), p->y + re(2624), p->y + re(2688), p->y + re(2752), p->wre_table[5], p->wim_table[5]);
fft_n_(4, p->y + re(2816), p->y + re(2820), p->y + re(2824), p->y + re(2828), p->wre_table[1], p->wim_table[1]);
fft_n_(8, p->y + re(2816), p->y + re(2824), p->y + re(2832), p->y + re(2840), p->wre_table[2], p->wim_table[2]);
fft_n_(4, p->y + re(2848), p->y + re(2852), p->y + re(2856), p->y + re(2860), p->wre_table[1], p->wim_table[1]);
fft_n_(4, p->y + re(2864), p->y + re(2868), p->y + re(2872), p->y + re(2876), p->wre_table[1], p->wim_table[1]);
fft_n_(16, p->y + re(2816), p->y + re(2832), p->y + re(2848), p->y + re(2864), p->wre_table[3], p->wim_table[3]);
fft_n_(4, p->y + re(2880), p->y + re(2884), p->y + re(2888), p->y + re(2892), p->wre_table[1], p->wim_table[1]);
fft_n_(8, p->y + re(2880), p->y + re(2888), p->y + re(2896), p->y + re(2904), p->wre_table[2], p->wim_table[2]);
fft_n_(4, p->y + re(2912), p->y + re(2916), p->y + re(2920), p->y + re(2924), p->wre_table[1], p->wim_table[1]);
fft_n_(8, p->y + re(2912), p->y + re(2920), p->y + re(2928), p->y + re(2936), p->wre_table[2], p->wim_table[2]);
fft_n_(32, p->y + re(2816), p->y + re(2848), p->y + re(2880), p->y + re(2912), p->wre_table[4], p->wim_table[4]);
fft_n_(4, p->y + re(2944), p->y + re(2948), p->y + re(2952), p->y + re(2956), p->wre_table[1], p->wim_table[1]);
fft_n_(8, p->y + re(2944), p->y + re(2952), p->y + re(2960), p->y + re(2968), p->wre_table[2], p->wim_table[2]);
fft_n_(4, p->y + re(2976), p->y + re(2980), p->y + re(2984), p->y + re(2988), p->wre_table[1], p->wim_table[1]);
fft_n_(4, p->y + re(2992), p->y + re(2996), p->y + re(3000), p->y + re(3004), p->wre_table[1], p->wim_table[1]);
fft_n_(16, p->y + re(2944), p->y + re(2960), p->y + re(2976), p->y + re(2992), p->wre_table[3], p->wim_table[3]);
fft_n_(4, p->y + re(3008), p->y + re(3012), p->y + re(3016), p->y + re(3020), p->wre_table[1], p->wim_table[1]);
fft_n_(8, p->y + re(3008), p->y + re(3016), p->y + re(3024), p->y + re(3032), p->wre_table[2], p->wim_table[2]);
fft_n_(4, p->y + re(3040), p->y + re(3044), p->y + re(3048), p->y + re(3052), p->wre_table[1], p->wim_table[1]);
fft_n_(4, p->y + re(3056), p->y + re(3060), p->y + re(3064), p->y + re(3068), p->wre_table[1], p->wim_table[1]);
fft_n_(16, p->y + re(3008), p->y + re(3024), p->y + re(3040), p->y + re(3056), p->wre_table[3], p->wim_table[3]);
fft_n_(64, p->y + re(2816), p->y + re(2880), p->y + re(2944), p->y + re(3008), p->wre_table[5], p->wim_table[5]);
fft_n_(256, p->y + re(2048), p->y + re(2304), p->y + re(2560), p->y + re(2816), p->wre_table[7], p->wim_table[7]);
fft_n_(4, p->y + re(3072), p->y + re(3076), p->y + re(3080), p->y + re(3084), p->wre_table[1], p->wim_table[1]);
fft_n_(8, p->y + re(3072), p->y + re(3080), p->y + re(3088), p->y + re(3096), p->wre_table[2], p->wim_table[2]);
fft_n_(4, p->y + re(3104), p->y + re(3108), p->y + re(3112), p->y + re(3116), p->wre_table[1], p->wim_table[1]);
fft_n_(4, p->y + re(3120), p->y + re(3124), p->y + re(3128), p->y + re(3132), p->wre_table[1], p->wim_table[1]);
fft_n_(16, p->y + re(3072), p->y + re(3088), p->y + re(3104), p->y + re(3120), p->wre_table[3], p->wim_table[3]);
fft_n_(4, p->y + re(3136), p->y + re(3140), p->y + re(3144), p->y + re(3148), p->wre_table[1], p->wim_table[1]);
fft_n_(8, p->y + re(3136), p->y + re(3144), p->y + re(3152), p->y + re(3160), p->wre_table[2], p->wim_table[2]);
fft_n_(4, p->y + re(3168), p->y + re(3172), p->y + re(3176), p->y + re(3180), p->wre_table[1], p->wim_table[1]);
fft_n_(8, p->y + re(3168), p->y + re(3176), p->y + re(3184), p->y + re(3192), p->wre_table[2], p->wim_table[2]);
fft_n_(32, p->y + re(3072), p->y + re(3104), p->y + re(3136), p->y + re(3168), p->wre_table[4], p->wim_table[4]);
fft_n_(4, p->y + re(3200), p->y + re(3204), p->y + re(3208), p->y + re(3212), p->wre_table[1], p->wim_table[1]);
fft_n_(8, p->y + re(3200), p->y + re(3208), p->y + re(3216), p->y + re(3224), p->wre_table[2], p->wim_table[2]);
fft_n_(4, p->y + re(3232), p->y + re(3236), p->y + re(3240), p->y + re(3244), p->wre_table[1], p->wim_table[1]);
fft_n_(4, p->y + re(3248), p->y + re(3252), p->y + re(3256), p->y + re(3260), p->wre_table[1], p->wim_table[1]);
fft_n_(16, p->y + re(3200), p->y + re(3216), p->y + re(3232), p->y + re(3248), p->wre_table[3], p->wim_table[3]);
fft_n_(4, p->y + re(3264), p->y + re(3268), p->y + re(3272), p->y + re(3276), p->wre_table[1], p->wim_table[1]);
fft_n_(8, p->y + re(3264), p->y + re(3272), p->y + re(3280), p->y + re(3288), p->wre_table[2], p->wim_table[2]);
fft_n_(4, p->y + re(3296), p->y + re(3300), p->y + re(3304), p->y + re(3308), p->wre_table[1], p->wim_table[1]);
fft_n_(4, p->y + re(3312), p->y + re(3316), p->y + re(3320), p->y + re(3324), p->wre_table[1], p->wim_table[1]);
fft_n_(16, p->y + re(3264), p->y + re(3280), p->y + re(3296), p->y + re(3312), p->wre_table[3], p->wim_table[3]);
fft_n_(64, p->y + re(3072), p->y + re(3136), p->y + re(3200), p->y + re(3264), p->wre_table[5], p->wim_table[5]);
fft_n_(4, p->y + re(3328), p->y + re(3332), p->y + re(3336), p->y + re(3340), p->wre_table[1], p->wim_table[1]);
fft_n_(8, p->y + re(3328), p->y + re(3336), p->y + re(3344), p->y + re(3352), p->wre_table[2], p->wim_table[2]);
fft_n_(4, p->y + re(3360), p->y + re(3364), p->y + re(3368), p->y + re(3372), p->wre_table[1], p->wim_table[1]);
fft_n_(4, p->y + re(3376), p->y + re(3380), p->y + re(3384), p->y + re(3388), p->wre_table[1], p->wim_table[1]);
fft_n_(16, p->y + re(3328), p->y + re(3344), p->y + re(3360), p->y + re(3376), p->wre_table[3], p->wim_table[3]);
fft_n_(4, p->y + re(3392), p->y + re(3396), p->y + re(3400), p->y + re(3404), p->wre_table[1], p->wim_table[1]);
fft_n_(8, p->y + re(3392), p->y + re(3400), p->y + re(3408), p->y + re(3416), p->wre_table[2], p->wim_table[2]);
fft_n_(4, p->y + re(3424), p->y + re(3428), p->y + re(3432), p->y + re(3436), p->wre_table[1], p->wim_table[1]);
fft_n_(8, p->y + re(3424), p->y + re(3432), p->y + re(3440), p->y + re(3448), p->wre_table[2], p->wim_table[2]);
fft_n_(32, p->y + re(3328), p->y + re(3360), p->y + re(3392), p->y + re(3424), p->wre_table[4], p->wim_table[4]);
fft_n_(4, p->y + re(3456), p->y + re(3460), p->y + re(3464), p->y + re(3468), p->wre_table[1], p->wim_table[1]);
fft_n_(8, p->y + re(3456), p->y + re(3464), p->y + re(3472), p->y + re(3480), p->wre_table[2], p->wim_table[2]);
fft_n_(4, p->y + re(3488), p->y + re(3492), p->y + re(3496), p->y + re(3500), p->wre_table[1], p->wim_table[1]);
fft_n_(4, p->y + re(3504), p->y + re(3508), p->y + re(3512), p->y + re(3516), p->wre_table[1], p->wim_table[1]);
fft_n_(16, p->y + re(3456), p->y + re(3472), p->y + re(3488), p->y + re(3504), p->wre_table[3], p->wim_table[3]);
fft_n_(4, p->y + re(3520), p->y + re(3524), p->y + re(3528), p->y + re(3532), p->wre_table[1], p->wim_table[1]);
fft_n_(8, p->y + re(3520), p->y + re(3528), p->y + re(3536), p->y + re(3544), p->wre_table[2], p->wim_table[2]);
fft_n_(4, p->y + re(3552), p->y + re(3556), p->y + re(3560), p->y + re(3564), p->wre_table[1], p->wim_table[1]);
fft_n_(8, p->y + re(3552), p->y + re(3560), p->y + re(3568), p->y + re(3576), p->wre_table[2], p->wim_table[2]);
fft_n_(32, p->y + re(3456), p->y + re(3488), p->y + re(3520), p->y + re(3552), p->wre_table[4], p->wim_table[4]);
fft_n_(128, p->y + re(3072), p->y + re(3200), p->y + re(3328), p->y + re(3456), p->wre_table[6], p->wim_table[6]);
fft_n_(4, p->y + re(3584), p->y + re(3588), p->y + re(3592), p->y + re(3596), p->wre_table[1], p->wim_table[1]);
fft_n_(8, p->y + re(3584), p->y + re(3592), p->y + re(3600), p->y + re(3608), p->wre_table[2], p->wim_table[2]);
fft_n_(4, p->y + re(3616), p->y + re(3620), p->y + re(3624), p->y + re(3628), p->wre_table[1], p->wim_table[1]);
fft_n_(4, p->y + re(3632), p->y + re(3636), p->y + re(3640), p->y + re(3644), p->wre_table[1], p->wim_table[1]);
fft_n_(16, p->y + re(3584), p->y + re(3600), p->y + re(3616), p->y + re(3632), p->wre_table[3], p->wim_table[3]);
fft_n_(4, p->y + re(3648), p->y + re(3652), p->y + re(3656), p->y + re(3660), p->wre_table[1], p->wim_table[1]);
fft_n_(8, p->y + re(3648), p->y + re(3656), p->y + re(3664), p->y + re(3672), p->wre_table[2], p->wim_table[2]);
fft_n_(4, p->y + re(3680), p->y + re(3684), p->y + re(3688), p->y + re(3692), p->wre_table[1], p->wim_table[1]);
fft_n_(8, p->y + re(3680), p->y + re(3688), p->y + re(3696), p->y + re(3704), p->wre_table[2], p->wim_table[2]);
fft_n_(32, p->y + re(3584), p->y + re(3616), p->y + re(3648), p->y + re(3680), p->wre_table[4], p->wim_table[4]);
fft_n_(4, p->y + re(3712), p->y + re(3716), p->y + re(3720), p->y + re(3724), p->wre_table[1], p->wim_table[1]);
fft_n_(8, p->y + re(3712), p->y + re(3720), p->y + re(3728), p->y + re(3736), p->wre_table[2], p->wim_table[2]);
fft_n_(4, p->y + re(3744), p->y + re(3748), p->y + re(3752), p->y + re(3756), p->wre_table[1], p->wim_table[1]);
fft_n_(4, p->y + re(3760), p->y + re(3764), p->y + re(3768), p->y + re(3772), p->wre_table[1], p->wim_table[1]);
fft_n_(16, p->y + re(3712), p->y + re(3728), p->y + re(3744), p->y + re(3760), p->wre_table[3], p->wim_table[3]);
fft_n_(4, p->y + re(3776), p->y + re(3780), p->y + re(3784), p->y + re(3788), p->wre_table[1], p->wim_table[1]);
fft_n_(8, p->y + re(3776), p->y + re(3784), p->y + re(3792), p->y + re(3800), p->wre_table[2], p->wim_table[2]);
fft_n_(4, p->y + re(3808), p->y + re(3812), p->y + re(3816), p->y + re(3820), p->wre_table[1], p->wim_table[1]);
fft_n_(4, p->y + re(3824), p->y + re(3828), p->y + re(3832), p->y + re(3836), p->wre_table[1], p->wim_table[1]);
fft_n_(16, p->y + re(3776), p->y + re(3792), p->y + re(3808), p->y + re(3824), p->wre_table[3], p->wim_table[3]);
fft_n_(64, p->y + re(3584), p->y + re(3648), p->y + re(3712), p->y + re(3776), p->wre_table[5], p->wim_table[5]);
fft_n_(4, p->y + re(3840), p->y + re(3844), p->y + re(3848), p->y + re(3852), p->wre_table[1], p->wim_table[1]);
fft_n_(8, p->y + re(3840), p->y + re(3848), p->y + re(3856), p->y + re(3864), p->wre_table[2], p->wim_table[2]);
fft_n_(4, p->y + re(3872), p->y + re(3876), p->y + re(3880), p->y + re(3884), p->wre_table[1], p->wim_table[1]);
fft_n_(4, p->y + re(3888), p->y + re(3892), p->y + re(3896), p->y + re(3900), p->wre_table[1], p->wim_table[1]);
fft_n_(16, p->y + re(3840), p->y + re(3856), p->y + re(3872), p->y + re(3888), p->wre_table[3], p->wim_table[3]);
fft_n_(4, p->y + re(3904), p->y + re(3908), p->y + re(3912), p->y + re(3916), p->wre_table[1], p->wim_table[1]);
fft_n_(8, p->y + re(3904), p->y + re(3912), p->y + re(3920), p->y + re(3928), p->wre_table[2], p->wim_table[2]);
fft_n_(4, p->y + re(3936), p->y + re(3940), p->y + re(3944), p->y + re(3948), p->wre_table[1], p->wim_table[1]);
fft_n_(8, p->y + re(3936), p->y + re(3944), p->y + re(3952), p->y + re(3960), p->wre_table[2], p->wim_table[2]);
fft_n_(32, p->y + re(3840), p->y + re(3872), p->y + re(3904), p->y + re(3936), p->wre_table[4], p->wim_table[4]);
fft_n_(4, p->y + re(3968), p->y + re(3972), p->y + re(3976), p->y + re(3980), p->wre_table[1], p->wim_table[1]);
fft_n_(8, p->y + re(3968), p->y + re(3976), p->y + re(3984), p->y + re(3992), p->wre_table[2], p->wim_table[2]);
fft_n_(4, p->y + re(4000), p->y + re(4004), p->y + re(4008), p->y + re(4012), p->wre_table[1], p->wim_table[1]);
fft_n_(4, p->y + re(4016), p->y + re(4020), p->y + re(4024), p->y + re(4028), p->wre_table[1], p->wim_table[1]);
fft_n_(16, p->y + re(3968), p->y + re(3984), p->y + re(4000), p->y + re(4016), p->wre_table[3], p->wim_table[3]);
fft_n_(4, p->y + re(4032), p->y + re(4036), p->y + re(4040), p->y + re(4044), p->wre_table[1], p->wim_table[1]);
fft_n_(8, p->y + re(4032), p->y + re(4040), p->y + re(4048), p->y + re(4056), p->wre_table[2], p->wim_table[2]);
fft_n_(4, p->y + re(4064), p->y + re(4068), p->y + re(4072), p->y + re(4076), p->wre_table[1], p->wim_table[1]);
fft_n_(4, p->y + re(4080), p->y + re(4084), p->y + re(4088), p->y + re(4092), p->wre_table[1], p->wim_table[1]);
fft_n_(16, p->y + re(4032), p->y + re(4048), p->y + re(4064), p->y + re(4080), p->wre_table[3], p->wim_table[3]);
fft_n_(64, p->y + re(3840), p->y + re(3904), p->y + re(3968), p->y + re(4032), p->wre_table[5], p->wim_table[5]);
fft_n_(256, p->y + re(3072), p->y + re(3328), p->y + re(3584), p->y + re(3840), p->wre_table[7], p->wim_table[7]);
fft_n_(1024, p->y + re(0), p->y + re(1024), p->y + re(2048), p->y + re(3072), p->wre_table[9], p->wim_table[9]);
fft_n_(4, p->y + re(4096), p->y + re(4100), p->y + re(4104), p->y + re(4108), p->wre_table[1], p->wim_table[1]);
fft_n_(8, p->y + re(4096), p->y + re(4104), p->y + re(4112), p->y + re(4120), p->wre_table[2], p->wim_table[2]);
fft_n_(4, p->y + re(4128), p->y + re(4132), p->y + re(4136), p->y + re(4140), p->wre_table[1], p->wim_table[1]);
fft_n_(4, p->y + re(4144), p->y + re(4148), p->y + re(4152), p->y + re(4156), p->wre_table[1], p->wim_table[1]);
fft_n_(16, p->y + re(4096), p->y + re(4112), p->y + re(4128), p->y + re(4144), p->wre_table[3], p->wim_table[3]);
fft_n_(4, p->y + re(4160), p->y + re(4164), p->y + re(4168), p->y + re(4172), p->wre_table[1], p->wim_table[1]);
fft_n_(8, p->y + re(4160), p->y + re(4168), p->y + re(4176), p->y + re(4184), p->wre_table[2], p->wim_table[2]);
fft_n_(4, p->y + re(4192), p->y + re(4196), p->y + re(4200), p->y + re(4204), p->wre_table[1], p->wim_table[1]);
fft_n_(8, p->y + re(4192), p->y + re(4200), p->y + re(4208), p->y + re(4216), p->wre_table[2], p->wim_table[2]);
fft_n_(32, p->y + re(4096), p->y + re(4128), p->y + re(4160), p->y + re(4192), p->wre_table[4], p->wim_table[4]);
fft_n_(4, p->y + re(4224), p->y + re(4228), p->y + re(4232), p->y + re(4236), p->wre_table[1], p->wim_table[1]);
fft_n_(8, p->y + re(4224), p->y + re(4232), p->y + re(4240), p->y + re(4248), p->wre_table[2], p->wim_table[2]);
fft_n_(4, p->y + re(4256), p->y + re(4260), p->y + re(4264), p->y + re(4268), p->wre_table[1], p->wim_table[1]);
fft_n_(4, p->y + re(4272), p->y + re(4276), p->y + re(4280), p->y + re(4284), p->wre_table[1], p->wim_table[1]);
fft_n_(16, p->y + re(4224), p->y + re(4240), p->y + re(4256), p->y + re(4272), p->wre_table[3], p->wim_table[3]);
fft_n_(4, p->y + re(4288), p->y + re(4292), p->y + re(4296), p->y + re(4300), p->wre_table[1], p->wim_table[1]);
fft_n_(8, p->y + re(4288), p->y + re(4296), p->y + re(4304), p->y + re(4312), p->wre_table[2], p->wim_table[2]);
fft_n_(4, p->y + re(4320), p->y + re(4324), p->y + re(4328), p->y + re(4332), p->wre_table[1], p->wim_table[1]);
fft_n_(4, p->y + re(4336), p->y + re(4340), p->y + re(4344), p->y + re(4348), p->wre_table[1], p->wim_table[1]);
fft_n_(16, p->y + re(4288), p->y + re(4304), p->y + re(4320), p->y + re(4336), p->wre_table[3], p->wim_table[3]);
fft_n_(64, p->y + re(4096), p->y + re(4160), p->y + re(4224), p->y + re(4288), p->wre_table[5], p->wim_table[5]);
fft_n_(4, p->y + re(4352), p->y + re(4356), p->y + re(4360), p->y + re(4364), p->wre_table[1], p->wim_table[1]);
fft_n_(8, p->y + re(4352), p->y + re(4360), p->y + re(4368), p->y + re(4376), p->wre_table[2], p->wim_table[2]);
fft_n_(4, p->y + re(4384), p->y + re(4388), p->y + re(4392), p->y + re(4396), p->wre_table[1], p->wim_table[1]);
fft_n_(4, p->y + re(4400), p->y + re(4404), p->y + re(4408), p->y + re(4412), p->wre_table[1], p->wim_table[1]);
fft_n_(16, p->y + re(4352), p->y + re(4368), p->y + re(4384), p->y + re(4400), p->wre_table[3], p->wim_table[3]);
fft_n_(4, p->y + re(4416), p->y + re(4420), p->y + re(4424), p->y + re(4428), p->wre_table[1], p->wim_table[1]);
fft_n_(8, p->y + re(4416), p->y + re(4424), p->y + re(4432), p->y + re(4440), p->wre_table[2], p->wim_table[2]);
fft_n_(4, p->y + re(4448), p->y + re(4452), p->y + re(4456), p->y + re(4460), p->wre_table[1], p->wim_table[1]);
fft_n_(8, p->y + re(4448), p->y + re(4456), p->y + re(4464), p->y + re(4472), p->wre_table[2], p->wim_table[2]);
fft_n_(32, p->y + re(4352), p->y + re(4384), p->y + re(4416), p->y + re(4448), p->wre_table[4], p->wim_table[4]);
fft_n_(4, p->y + re(4480), p->y + re(4484), p->y + re(4488), p->y + re(4492), p->wre_table[1], p->wim_table[1]);
fft_n_(8, p->y + re(4480), p->y + re(4488), p->y + re(4496), p->y + re(4504), p->wre_table[2], p->wim_table[2]);
fft_n_(4, p->y + re(4512), p->y + re(4516), p->y + re(4520), p->y + re(4524), p->wre_table[1], p->wim_table[1]);
fft_n_(4, p->y + re(4528), p->y + re(4532), p->y + re(4536), p->y + re(4540), p->wre_table[1], p->wim_table[1]);
fft_n_(16, p->y + re(4480), p->y + re(4496), p->y + re(4512), p->y + re(4528), p->wre_table[3], p->wim_table[3]);
fft_n_(4, p->y + re(4544), p->y + re(4548), p->y + re(4552), p->y + re(4556), p->wre_table[1], p->wim_table[1]);
fft_n_(8, p->y + re(4544), p->y + re(4552), p->y + re(4560), p->y + re(4568), p->wre_table[2], p->wim_table[2]);
fft_n_(4, p->y + re(4576), p->y + re(4580), p->y + re(4584), p->y + re(4588), p->wre_table[1], p->wim_table[1]);
fft_n_(8, p->y + re(4576), p->y + re(4584), p->y + re(4592), p->y + re(4600), p->wre_table[2], p->wim_table[2]);
fft_n_(32, p->y + re(4480), p->y + re(4512), p->y + re(4544), p->y + re(4576), p->wre_table[4], p->wim_table[4]);
fft_n_(128, p->y + re(4096), p->y + re(4224), p->y + re(4352), p->y + re(4480), p->wre_table[6], p->wim_table[6]);
fft_n_(4, p->y + re(4608), p->y + re(4612), p->y + re(4616), p->y + re(4620), p->wre_table[1], p->wim_table[1]);
fft_n_(8, p->y + re(4608), p->y + re(4616), p->y + re(4624), p->y + re(4632), p->wre_table[2], p->wim_table[2]);
fft_n_(4, p->y + re(4640), p->y + re(4644), p->y + re(4648), p->y + re(4652), p->wre_table[1], p->wim_table[1]);
fft_n_(4, p->y + re(4656), p->y + re(4660), p->y + re(4664), p->y + re(4668), p->wre_table[1], p->wim_table[1]);
fft_n_(16, p->y + re(4608), p->y + re(4624), p->y + re(4640), p->y + re(4656), p->wre_table[3], p->wim_table[3]);
fft_n_(4, p->y + re(4672), p->y + re(4676), p->y + re(4680), p->y + re(4684), p->wre_table[1], p->wim_table[1]);
fft_n_(8, p->y + re(4672), p->y + re(4680), p->y + re(4688), p->y + re(4696), p->wre_table[2], p->wim_table[2]);
fft_n_(4, p->y + re(4704), p->y + re(4708), p->y + re(4712), p->y + re(4716), p->wre_table[1], p->wim_table[1]);
fft_n_(8, p->y + re(4704), p->y + re(4712), p->y + re(4720), p->y + re(4728), p->wre_table[2], p->wim_table[2]);
fft_n_(32, p->y + re(4608), p->y + re(4640), p->y + re(4672), p->y + re(4704), p->wre_table[4], p->wim_table[4]);
fft_n_(4, p->y + re(4736), p->y + re(4740), p->y + re(4744), p->y + re(4748), p->wre_table[1], p->wim_table[1]);
fft_n_(8, p->y + re(4736), p->y + re(4744), p->y + re(4752), p->y + re(4760), p->wre_table[2], p->wim_table[2]);
fft_n_(4, p->y + re(4768), p->y + re(4772), p->y + re(4776), p->y + re(4780), p->wre_table[1], p->wim_table[1]);
fft_n_(4, p->y + re(4784), p->y + re(4788), p->y + re(4792), p->y + re(4796), p->wre_table[1], p->wim_table[1]);
fft_n_(16, p->y + re(4736), p->y + re(4752), p->y + re(4768), p->y + re(4784), p->wre_table[3], p->wim_table[3]);
fft_n_(4, p->y + re(4800), p->y + re(4804), p->y + re(4808), p->y + re(4812), p->wre_table[1], p->wim_table[1]);
fft_n_(8, p->y + re(4800), p->y + re(4808), p->y + re(4816), p->y + re(4824), p->wre_table[2], p->wim_table[2]);
fft_n_(4, p->y + re(4832), p->y + re(4836), p->y + re(4840), p->y + re(4844), p->wre_table[1], p->wim_table[1]);
fft_n_(4, p->y + re(4848), p->y + re(4852), p->y + re(4856), p->y + re(4860), p->wre_table[1], p->wim_table[1]);
fft_n_(16, p->y + re(4800), p->y + re(4816), p->y + re(4832), p->y + re(4848), p->wre_table[3], p->wim_table[3]);
fft_n_(64, p->y + re(4608), p->y + re(4672), p->y + re(4736), p->y + re(4800), p->wre_table[5], p->wim_table[5]);
fft_n_(4, p->y + re(4864), p->y + re(4868), p->y + re(4872), p->y + re(4876), p->wre_table[1], p->wim_table[1]);
fft_n_(8, p->y + re(4864), p->y + re(4872), p->y + re(4880), p->y + re(4888), p->wre_table[2], p->wim_table[2]);
fft_n_(4, p->y + re(4896), p->y + re(4900), p->y + re(4904), p->y + re(4908), p->wre_table[1], p->wim_table[1]);
fft_n_(4, p->y + re(4912), p->y + re(4916), p->y + re(4920), p->y + re(4924), p->wre_table[1], p->wim_table[1]);
fft_n_(16, p->y + re(4864), p->y + re(4880), p->y + re(4896), p->y + re(4912), p->wre_table[3], p->wim_table[3]);
fft_n_(4, p->y + re(4928), p->y + re(4932), p->y + re(4936), p->y + re(4940), p->wre_table[1], p->wim_table[1]);
fft_n_(8, p->y + re(4928), p->y + re(4936), p->y + re(4944), p->y + re(4952), p->wre_table[2], p->wim_table[2]);
fft_n_(4, p->y + re(4960), p->y + re(4964), p->y + re(4968), p->y + re(4972), p->wre_table[1], p->wim_table[1]);
fft_n_(8, p->y + re(4960), p->y + re(4968), p->y + re(4976), p->y + re(4984), p->wre_table[2], p->wim_table[2]);
fft_n_(32, p->y + re(4864), p->y + re(4896), p->y + re(4928), p->y + re(4960), p->wre_table[4], p->wim_table[4]);
fft_n_(4, p->y + re(4992), p->y + re(4996), p->y + re(5000), p->y + re(5004), p->wre_table[1], p->wim_table[1]);
fft_n_(8, p->y + re(4992), p->y + re(5000), p->y + re(5008), p->y + re(5016), p->wre_table[2], p->wim_table[2]);
fft_n_(4, p->y + re(5024), p->y + re(5028), p->y + re(5032), p->y + re(5036), p->wre_table[1], p->wim_table[1]);
fft_n_(4, p->y + re(5040), p->y + re(5044), p->y + re(5048), p->y + re(5052), p->wre_table[1], p->wim_table[1]);
fft_n_(16, p->y + re(4992), p->y + re(5008), p->y + re(5024), p->y + re(5040), p->wre_table[3], p->wim_table[3]);
fft_n_(4, p->y + re(5056), p->y + re(5060), p->y + re(5064), p->y + re(5068), p->wre_table[1], p->wim_table[1]);
fft_n_(8, p->y + re(5056), p->y + re(5064), p->y + re(5072), p->y + re(5080), p->wre_table[2], p->wim_table[2]);
fft_n_(4, p->y + re(5088), p->y + re(5092), p->y + re(5096), p->y + re(5100), p->wre_table[1], p->wim_table[1]);
fft_n_(4, p->y + re(5104), p->y + re(5108), p->y + re(5112), p->y + re(5116), p->wre_table[1], p->wim_table[1]);
fft_n_(16, p->y + re(5056), p->y + re(5072), p->y + re(5088), p->y + re(5104), p->wre_table[3], p->wim_table[3]);
fft_n_(64, p->y + re(4864), p->y + re(4928), p->y + re(4992), p->y + re(5056), p->wre_table[5], p->wim_table[5]);
fft_n_(256, p->y + re(4096), p->y + re(4352), p->y + re(4608), p->y + re(4864), p->wre_table[7], p->wim_table[7]);
fft_n_(4, p->y + re(5120), p->y + re(5124), p->y + re(5128), p->y + re(5132), p->wre_table[1], p->wim_table[1]);
fft_n_(8, p->y + re(5120), p->y + re(5128), p->y + re(5136), p->y + re(5144), p->wre_table[2], p->wim_table[2]);
fft_n_(4, p->y + re(5152), p->y + re(5156), p->y + re(5160), p->y + re(5164), p->wre_table[1], p->wim_table[1]);
fft_n_(4, p->y + re(5168), p->y + re(5172), p->y + re(5176), p->y + re(5180), p->wre_table[1], p->wim_table[1]);
fft_n_(16, p->y + re(5120), p->y + re(5136), p->y + re(5152), p->y + re(5168), p->wre_table[3], p->wim_table[3]);
fft_n_(4, p->y + re(5184), p->y + re(5188), p->y + re(5192), p->y + re(5196), p->wre_table[1], p->wim_table[1]);
fft_n_(8, p->y + re(5184), p->y + re(5192), p->y + re(5200), p->y + re(5208), p->wre_table[2], p->wim_table[2]);
fft_n_(4, p->y + re(5216), p->y + re(5220), p->y + re(5224), p->y + re(5228), p->wre_table[1], p->wim_table[1]);
fft_n_(8, p->y + re(5216), p->y + re(5224), p->y + re(5232), p->y + re(5240), p->wre_table[2], p->wim_table[2]);
fft_n_(32, p->y + re(5120), p->y + re(5152), p->y + re(5184), p->y + re(5216), p->wre_table[4], p->wim_table[4]);
fft_n_(4, p->y + re(5248), p->y + re(5252), p->y + re(5256), p->y + re(5260), p->wre_table[1], p->wim_table[1]);
fft_n_(8, p->y + re(5248), p->y + re(5256), p->y + re(5264), p->y + re(5272), p->wre_table[2], p->wim_table[2]);
fft_n_(4, p->y + re(5280), p->y + re(5284), p->y + re(5288), p->y + re(5292), p->wre_table[1], p->wim_table[1]);
fft_n_(4, p->y + re(5296), p->y + re(5300), p->y + re(5304), p->y + re(5308), p->wre_table[1], p->wim_table[1]);
fft_n_(16, p->y + re(5248), p->y + re(5264), p->y + re(5280), p->y + re(5296), p->wre_table[3], p->wim_table[3]);
fft_n_(4, p->y + re(5312), p->y + re(5316), p->y + re(5320), p->y + re(5324), p->wre_table[1], p->wim_table[1]);
fft_n_(8, p->y + re(5312), p->y + re(5320), p->y + re(5328), p->y + re(5336), p->wre_table[2], p->wim_table[2]);
fft_n_(4, p->y + re(5344), p->y + re(5348), p->y + re(5352), p->y + re(5356), p->wre_table[1], p->wim_table[1]);
fft_n_(4, p->y + re(5360), p->y + re(5364), p->y + re(5368), p->y + re(5372), p->wre_table[1], p->wim_table[1]);
fft_n_(16, p->y + re(5312), p->y + re(5328), p->y + re(5344), p->y + re(5360), p->wre_table[3], p->wim_table[3]);
fft_n_(64, p->y + re(5120), p->y + re(5184), p->y + re(5248), p->y + re(5312), p->wre_table[5], p->wim_table[5]);
fft_n_(4, p->y + re(5376), p->y + re(5380), p->y + re(5384), p->y + re(5388), p->wre_table[1], p->wim_table[1]);
fft_n_(8, p->y + re(5376), p->y + re(5384), p->y + re(5392), p->y + re(5400), p->wre_table[2], p->wim_table[2]);
fft_n_(4, p->y + re(5408), p->y + re(5412), p->y + re(5416), p->y + re(5420), p->wre_table[1], p->wim_table[1]);
fft_n_(4, p->y + re(5424), p->y + re(5428), p->y + re(5432), p->y + re(5436), p->wre_table[1], p->wim_table[1]);
fft_n_(16, p->y + re(5376), p->y + re(5392), p->y + re(5408), p->y + re(5424), p->wre_table[3], p->wim_table[3]);
fft_n_(4, p->y + re(5440), p->y + re(5444), p->y + re(5448), p->y + re(5452), p->wre_table[1], p->wim_table[1]);
fft_n_(8, p->y + re(5440), p->y + re(5448), p->y + re(5456), p->y + re(5464), p->wre_table[2], p->wim_table[2]);
fft_n_(4, p->y + re(5472), p->y + re(5476), p->y + re(5480), p->y + re(5484), p->wre_table[1], p->wim_table[1]);
fft_n_(8, p->y + re(5472), p->y + re(5480), p->y + re(5488), p->y + re(5496), p->wre_table[2], p->wim_table[2]);
fft_n_(32, p->y + re(5376), p->y + re(5408), p->y + re(5440), p->y + re(5472), p->wre_table[4], p->wim_table[4]);
fft_n_(4, p->y + re(5504), p->y + re(5508), p->y + re(5512), p->y + re(5516), p->wre_table[1], p->wim_table[1]);
fft_n_(8, p->y + re(5504), p->y + re(5512), p->y + re(5520), p->y + re(5528), p->wre_table[2], p->wim_table[2]);
fft_n_(4, p->y + re(5536), p->y + re(5540), p->y + re(5544), p->y + re(5548), p->wre_table[1], p->wim_table[1]);
fft_n_(4, p->y + re(5552), p->y + re(5556), p->y + re(5560), p->y + re(5564), p->wre_table[1], p->wim_table[1]);
fft_n_(16, p->y + re(5504), p->y + re(5520), p->y + re(5536), p->y + re(5552), p->wre_table[3], p->wim_table[3]);
fft_n_(4, p->y + re(5568), p->y + re(5572), p->y + re(5576), p->y + re(5580), p->wre_table[1], p->wim_table[1]);
fft_n_(8, p->y + re(5568), p->y + re(5576), p->y + re(5584), p->y + re(5592), p->wre_table[2], p->wim_table[2]);
fft_n_(4, p->y + re(5600), p->y + re(5604), p->y + re(5608), p->y + re(5612), p->wre_table[1], p->wim_table[1]);
fft_n_(8, p->y + re(5600), p->y + re(5608), p->y + re(5616), p->y + re(5624), p->wre_table[2], p->wim_table[2]);
fft_n_(32, p->y + re(5504), p->y + re(5536), p->y + re(5568), p->y + re(5600), p->wre_table[4], p->wim_table[4]);
fft_n_(128, p->y + re(5120), p->y + re(5248), p->y + re(5376), p->y + re(5504), p->wre_table[6], p->wim_table[6]);
fft_n_(4, p->y + re(5632), p->y + re(5636), p->y + re(5640), p->y + re(5644), p->wre_table[1], p->wim_table[1]);
fft_n_(8, p->y + re(5632), p->y + re(5640), p->y + re(5648), p->y + re(5656), p->wre_table[2], p->wim_table[2]);
fft_n_(4, p->y + re(5664), p->y + re(5668), p->y + re(5672), p->y + re(5676), p->wre_table[1], p->wim_table[1]);
fft_n_(4, p->y + re(5680), p->y + re(5684), p->y + re(5688), p->y + re(5692), p->wre_table[1], p->wim_table[1]);
fft_n_(16, p->y + re(5632), p->y + re(5648), p->y + re(5664), p->y + re(5680), p->wre_table[3], p->wim_table[3]);
fft_n_(4, p->y + re(5696), p->y + re(5700), p->y + re(5704), p->y + re(5708), p->wre_table[1], p->wim_table[1]);
fft_n_(8, p->y + re(5696), p->y + re(5704), p->y + re(5712), p->y + re(5720), p->wre_table[2], p->wim_table[2]);
fft_n_(4, p->y + re(5728), p->y + re(5732), p->y + re(5736), p->y + re(5740), p->wre_table[1], p->wim_table[1]);
fft_n_(8, p->y + re(5728), p->y + re(5736), p->y + re(5744), p->y + re(5752), p->wre_table[2], p->wim_table[2]);
fft_n_(32, p->y + re(5632), p->y + re(5664), p->y + re(5696), p->y + re(5728), p->wre_table[4], p->wim_table[4]);
fft_n_(4, p->y + re(5760), p->y + re(5764), p->y + re(5768), p->y + re(5772), p->wre_table[1], p->wim_table[1]);
fft_n_(8, p->y + re(5760), p->y + re(5768), p->y + re(5776), p->y + re(5784), p->wre_table[2], p->wim_table[2]);
fft_n_(4, p->y + re(5792), p->y + re(5796), p->y + re(5800), p->y + re(5804), p->wre_table[1], p->wim_table[1]);
fft_n_(4, p->y + re(5808), p->y + re(5812), p->y + re(5816), p->y + re(5820), p->wre_table[1], p->wim_table[1]);
fft_n_(16, p->y + re(5760), p->y + re(5776), p->y + re(5792), p->y + re(5808), p->wre_table[3], p->wim_table[3]);
fft_n_(4, p->y + re(5824), p->y + re(5828), p->y + re(5832), p->y + re(5836), p->wre_table[1], p->wim_table[1]);
fft_n_(8, p->y + re(5824), p->y + re(5832), p->y + re(5840), p->y + re(5848), p->wre_table[2], p->wim_table[2]);
fft_n_(4, p->y + re(5856), p->y + re(5860), p->y + re(5864), p->y + re(5868), p->wre_table[1], p->wim_table[1]);
fft_n_(4, p->y + re(5872), p->y + re(5876), p->y + re(5880), p->y + re(5884), p->wre_table[1], p->wim_table[1]);
fft_n_(16, p->y + re(5824), p->y + re(5840), p->y + re(5856), p->y + re(5872), p->wre_table[3], p->wim_table[3]);
fft_n_(64, p->y + re(5632), p->y + re(5696), p->y + re(5760), p->y + re(5824), p->wre_table[5], p->wim_table[5]);
fft_n_(4, p->y + re(5888), p->y + re(5892), p->y + re(5896), p->y + re(5900), p->wre_table[1], p->wim_table[1]);
fft_n_(8, p->y + re(5888), p->y + re(5896), p->y + re(5904), p->y + re(5912), p->wre_table[2], p->wim_table[2]);
fft_n_(4, p->y + re(5920), p->y + re(5924), p->y + re(5928), p->y + re(5932), p->wre_table[1], p->wim_table[1]);
fft_n_(4, p->y + re(5936), p->y + re(5940), p->y + re(5944), p->y + re(5948), p->wre_table[1], p->wim_table[1]);
fft_n_(16, p->y + re(5888), p->y + re(5904), p->y + re(5920), p->y + re(5936), p->wre_table[3], p->wim_table[3]);
fft_n_(4, p->y + re(5952), p->y + re(5956), p->y + re(5960), p->y + re(5964), p->wre_table[1], p->wim_table[1]);
fft_n_(8, p->y + re(5952), p->y + re(5960), p->y + re(5968), p->y + re(5976), p->wre_table[2], p->wim_table[2]);
fft_n_(4, p->y + re(5984), p->y + re(5988), p->y + re(5992), p->y + re(5996), p->wre_table[1], p->wim_table[1]);
fft_n_(8, p->y + re(5984), p->y + re(5992), p->y + re(6000), p->y + re(6008), p->wre_table[2], p->wim_table[2]);
fft_n_(32, p->y + re(5888), p->y + re(5920), p->y + re(5952), p->y + re(5984), p->wre_table[4], p->wim_table[4]);
fft_n_(4, p->y + re(6016), p->y + re(6020), p->y + re(6024), p->y + re(6028), p->wre_table[1], p->wim_table[1]);
fft_n_(8, p->y + re(6016), p->y + re(6024), p->y + re(6032), p->y + re(6040), p->wre_table[2], p->wim_table[2]);
fft_n_(4, p->y + re(6048), p->y + re(6052), p->y + re(6056), p->y + re(6060), p->wre_table[1], p->wim_table[1]);
fft_n_(4, p->y + re(6064), p->y + re(6068), p->y + re(6072), p->y + re(6076), p->wre_table[1], p->wim_table[1]);
fft_n_(16, p->y + re(6016), p->y + re(6032), p->y + re(6048), p->y + re(6064), p->wre_table[3], p->wim_table[3]);
fft_n_(4, p->y + re(6080), p->y + re(6084), p->y + re(6088), p->y + re(6092), p->wre_table[1], p->wim_table[1]);
fft_n_(8, p->y + re(6080), p->y + re(6088), p->y + re(6096), p->y + re(6104), p->wre_table[2], p->wim_table[2]);
fft_n_(4, p->y + re(6112), p->y + re(6116), p->y + re(6120), p->y + re(6124), p->wre_table[1], p->wim_table[1]);
fft_n_(8, p->y + re(6112), p->y + re(6120), p->y + re(6128), p->y + re(6136), p->wre_table[2], p->wim_table[2]);
fft_n_(32, p->y + re(6016), p->y + re(6048), p->y + re(6080), p->y + re(6112), p->wre_table[4], p->wim_table[4]);
fft_n_(128, p->y + re(5632), p->y + re(5760), p->y + re(5888), p->y + re(6016), p->wre_table[6], p->wim_table[6]);
fft_n_(512, p->y + re(4096), p->y + re(4608), p->y + re(5120), p->y + re(5632), p->wre_table[8], p->wim_table[8]);
fft_n_(4, p->y + re(6144), p->y + re(6148), p->y + re(6152), p->y + re(6156), p->wre_table[1], p->wim_table[1]);
fft_n_(8, p->y + re(6144), p->y + re(6152), p->y + re(6160), p->y + re(6168), p->wre_table[2], p->wim_table[2]);
fft_n_(4, p->y + re(6176), p->y + re(6180), p->y + re(6184), p->y + re(6188), p->wre_table[1], p->wim_table[1]);
fft_n_(4, p->y + re(6192), p->y + re(6196), p->y + re(6200), p->y + re(6204), p->wre_table[1], p->wim_table[1]);
fft_n_(16, p->y + re(6144), p->y + re(6160), p->y + re(6176), p->y + re(6192), p->wre_table[3], p->wim_table[3]);
fft_n_(4, p->y + re(6208), p->y + re(6212), p->y + re(6216), p->y + re(6220), p->wre_table[1], p->wim_table[1]);
fft_n_(8, p->y + re(6208), p->y + re(6216), p->y + re(6224), p->y + re(6232), p->wre_table[2], p->wim_table[2]);
fft_n_(4, p->y + re(6240), p->y + re(6244), p->y + re(6248), p->y + re(6252), p->wre_table[1], p->wim_table[1]);
fft_n_(8, p->y + re(6240), p->y + re(6248), p->y + re(6256), p->y + re(6264), p->wre_table[2], p->wim_table[2]);
fft_n_(32, p->y + re(6144), p->y + re(6176), p->y + re(6208), p->y + re(6240), p->wre_table[4], p->wim_table[4]);
fft_n_(4, p->y + re(6272), p->y + re(6276), p->y + re(6280), p->y + re(6284), p->wre_table[1], p->wim_table[1]);
fft_n_(8, p->y + re(6272), p->y + re(6280), p->y + re(6288), p->y + re(6296), p->wre_table[2], p->wim_table[2]);
fft_n_(4, p->y + re(6304), p->y + re(6308), p->y + re(6312), p->y + re(6316), p->wre_table[1], p->wim_table[1]);
fft_n_(4, p->y + re(6320), p->y + re(6324), p->y + re(6328), p->y + re(6332), p->wre_table[1], p->wim_table[1]);
fft_n_(16, p->y + re(6272), p->y + re(6288), p->y + re(6304), p->y + re(6320), p->wre_table[3], p->wim_table[3]);
fft_n_(4, p->y + re(6336), p->y + re(6340), p->y + re(6344), p->y + re(6348), p->wre_table[1], p->wim_table[1]);
fft_n_(8, p->y + re(6336), p->y + re(6344), p->y + re(6352), p->y + re(6360), p->wre_table[2], p->wim_table[2]);
fft_n_(4, p->y + re(6368), p->y + re(6372), p->y + re(6376), p->y + re(6380), p->wre_table[1], p->wim_table[1]);
fft_n_(4, p->y + re(6384), p->y + re(6388), p->y + re(6392), p->y + re(6396), p->wre_table[1], p->wim_table[1]);
fft_n_(16, p->y + re(6336), p->y + re(6352), p->y + re(6368), p->y + re(6384), p->wre_table[3], p->wim_table[3]);
fft_n_(64, p->y + re(6144), p->y + re(6208), p->y + re(6272), p->y + re(6336), p->wre_table[5], p->wim_table[5]);
fft_n_(4, p->y + re(6400), p->y + re(6404), p->y + re(6408), p->y + re(6412), p->wre_table[1], p->wim_table[1]);
fft_n_(8, p->y + re(6400), p->y + re(6408), p->y + re(6416), p->y + re(6424), p->wre_table[2], p->wim_table[2]);
fft_n_(4, p->y + re(6432), p->y + re(6436), p->y + re(6440), p->y + re(6444), p->wre_table[1], p->wim_table[1]);
fft_n_(4, p->y + re(6448), p->y + re(6452), p->y + re(6456), p->y + re(6460), p->wre_table[1], p->wim_table[1]);
fft_n_(16, p->y + re(6400), p->y + re(6416), p->y + re(6432), p->y + re(6448), p->wre_table[3], p->wim_table[3]);
fft_n_(4, p->y + re(6464), p->y + re(6468), p->y + re(6472), p->y + re(6476), p->wre_table[1], p->wim_table[1]);
fft_n_(8, p->y + re(6464), p->y + re(6472), p->y + re(6480), p->y + re(6488), p->wre_table[2], p->wim_table[2]);
fft_n_(4, p->y + re(6496), p->y + re(6500), p->y + re(6504), p->y + re(6508), p->wre_table[1], p->wim_table[1]);
fft_n_(8, p->y + re(6496), p->y + re(6504), p->y + re(6512), p->y + re(6520), p->wre_table[2], p->wim_table[2]);
fft_n_(32, p->y + re(6400), p->y + re(6432), p->y + re(6464), p->y + re(6496), p->wre_table[4], p->wim_table[4]);
fft_n_(4, p->y + re(6528), p->y + re(6532), p->y + re(6536), p->y + re(6540), p->wre_table[1], p->wim_table[1]);
fft_n_(8, p->y + re(6528), p->y + re(6536), p->y + re(6544), p->y + re(6552), p->wre_table[2], p->wim_table[2]);
fft_n_(4, p->y + re(6560), p->y + re(6564), p->y + re(6568), p->y + re(6572), p->wre_table[1], p->wim_table[1]);
fft_n_(4, p->y + re(6576), p->y + re(6580), p->y + re(6584), p->y + re(6588), p->wre_table[1], p->wim_table[1]);
fft_n_(16, p->y + re(6528), p->y + re(6544), p->y + re(6560), p->y + re(6576), p->wre_table[3], p->wim_table[3]);
fft_n_(4, p->y + re(6592), p->y + re(6596), p->y + re(6600), p->y + re(6604), p->wre_table[1], p->wim_table[1]);
fft_n_(8, p->y + re(6592), p->y + re(6600), p->y + re(6608), p->y + re(6616), p->wre_table[2], p->wim_table[2]);
fft_n_(4, p->y + re(6624), p->y + re(6628), p->y + re(6632), p->y + re(6636), p->wre_table[1], p->wim_table[1]);
fft_n_(8, p->y + re(6624), p->y + re(6632), p->y + re(6640), p->y + re(6648), p->wre_table[2], p->wim_table[2]);
fft_n_(32, p->y + re(6528), p->y + re(6560), p->y + re(6592), p->y + re(6624), p->wre_table[4], p->wim_table[4]);
fft_n_(128, p->y + re(6144), p->y + re(6272), p->y + re(6400), p->y + re(6528), p->wre_table[6], p->wim_table[6]);
fft_n_(4, p->y + re(6656), p->y + re(6660), p->y + re(6664), p->y + re(6668), p->wre_table[1], p->wim_table[1]);
fft_n_(8, p->y + re(6656), p->y + re(6664), p->y + re(6672), p->y + re(6680), p->wre_table[2], p->wim_table[2]);
fft_n_(4, p->y + re(6688), p->y + re(6692), p->y + re(6696), p->y + re(6700), p->wre_table[1], p->wim_table[1]);
fft_n_(4, p->y + re(6704), p->y + re(6708), p->y + re(6712), p->y + re(6716), p->wre_table[1], p->wim_table[1]);
fft_n_(16, p->y + re(6656), p->y + re(6672), p->y + re(6688), p->y + re(6704), p->wre_table[3], p->wim_table[3]);
fft_n_(4, p->y + re(6720), p->y + re(6724), p->y + re(6728), p->y + re(6732), p->wre_table[1], p->wim_table[1]);
fft_n_(8, p->y + re(6720), p->y + re(6728), p->y + re(6736), p->y + re(6744), p->wre_table[2], p->wim_table[2]);
fft_n_(4, p->y + re(6752), p->y + re(6756), p->y + re(6760), p->y + re(6764), p->wre_table[1], p->wim_table[1]);
fft_n_(8, p->y + re(6752), p->y + re(6760), p->y + re(6768), p->y + re(6776), p->wre_table[2], p->wim_table[2]);
fft_n_(32, p->y + re(6656), p->y + re(6688), p->y + re(6720), p->y + re(6752), p->wre_table[4], p->wim_table[4]);
fft_n_(4, p->y + re(6784), p->y + re(6788), p->y + re(6792), p->y + re(6796), p->wre_table[1], p->wim_table[1]);
fft_n_(8, p->y + re(6784), p->y + re(6792), p->y + re(6800), p->y + re(6808), p->wre_table[2], p->wim_table[2]);
fft_n_(4, p->y + re(6816), p->y + re(6820), p->y + re(6824), p->y + re(6828), p->wre_table[1], p->wim_table[1]);
fft_n_(4, p->y + re(6832), p->y + re(6836), p->y + re(6840), p->y + re(6844), p->wre_table[1], p->wim_table[1]);
fft_n_(16, p->y + re(6784), p->y + re(6800), p->y + re(6816), p->y + re(6832), p->wre_table[3], p->wim_table[3]);
fft_n_(4, p->y + re(6848), p->y + re(6852), p->y + re(6856), p->y + re(6860), p->wre_table[1], p->wim_table[1]);
fft_n_(8, p->y + re(6848), p->y + re(6856), p->y + re(6864), p->y + re(6872), p->wre_table[2], p->wim_table[2]);
fft_n_(4, p->y + re(6880), p->y + re(6884), p->y + re(6888), p->y + re(6892), p->wre_table[1], p->wim_table[1]);
fft_n_(4, p->y + re(6896), p->y + re(6900), p->y + re(6904), p->y + re(6908), p->wre_table[1], p->wim_table[1]);
fft_n_(16, p->y + re(6848), p->y + re(6864), p->y + re(6880), p->y + re(6896), p->wre_table[3], p->wim_table[3]);
fft_n_(64, p->y + re(6656), p->y + re(6720), p->y + re(6784), p->y + re(6848), p->wre_table[5], p->wim_table[5]);
fft_n_(4, p->y + re(6912), p->y + re(6916), p->y + re(6920), p->y + re(6924), p->wre_table[1], p->wim_table[1]);
fft_n_(8, p->y + re(6912), p->y + re(6920), p->y + re(6928), p->y + re(6936), p->wre_table[2], p->wim_table[2]);
fft_n_(4, p->y + re(6944), p->y + re(6948), p->y + re(6952), p->y + re(6956), p->wre_table[1], p->wim_table[1]);
fft_n_(4, p->y + re(6960), p->y + re(6964), p->y + re(6968), p->y + re(6972), p->wre_table[1], p->wim_table[1]);
fft_n_(16, p->y + re(6912), p->y + re(6928), p->y + re(6944), p->y + re(6960), p->wre_table[3], p->wim_table[3]);
fft_n_(4, p->y + re(6976), p->y + re(6980), p->y + re(6984), p->y + re(6988), p->wre_table[1], p->wim_table[1]);
fft_n_(8, p->y + re(6976), p->y + re(6984), p->y + re(6992), p->y + re(7000), p->wre_table[2], p->wim_table[2]);
fft_n_(4, p->y + re(7008), p->y + re(7012), p->y + re(7016), p->y + re(7020), p->wre_table[1], p->wim_table[1]);
fft_n_(8, p->y + re(7008), p->y + re(7016), p->y + re(7024), p->y + re(7032), p->wre_table[2], p->wim_table[2]);
fft_n_(32, p->y + re(6912), p->y + re(6944), p->y + re(6976), p->y + re(7008), p->wre_table[4], p->wim_table[4]);
fft_n_(4, p->y + re(7040), p->y + re(7044), p->y + re(7048), p->y + re(7052), p->wre_table[1], p->wim_table[1]);
fft_n_(8, p->y + re(7040), p->y + re(7048), p->y + re(7056), p->y + re(7064), p->wre_table[2], p->wim_table[2]);
fft_n_(4, p->y + re(7072), p->y + re(7076), p->y + re(7080), p->y + re(7084), p->wre_table[1], p->wim_table[1]);
fft_n_(4, p->y + re(7088), p->y + re(7092), p->y + re(7096), p->y + re(7100), p->wre_table[1], p->wim_table[1]);
fft_n_(16, p->y + re(7040), p->y + re(7056), p->y + re(7072), p->y + re(7088), p->wre_table[3], p->wim_table[3]);
fft_n_(4, p->y + re(7104), p->y + re(7108), p->y + re(7112), p->y + re(7116), p->wre_table[1], p->wim_table[1]);
fft_n_(8, p->y + re(7104), p->y + re(7112), p->y + re(7120), p->y + re(7128), p->wre_table[2], p->wim_table[2]);
fft_n_(4, p->y + re(7136), p->y + re(7140), p->y + re(7144), p->y + re(7148), p->wre_table[1], p->wim_table[1]);
fft_n_(4, p->y + re(7152), p->y + re(7156), p->y + re(7160), p->y + re(7164), p->wre_table[1], p->wim_table[1]);
fft_n_(16, p->y + re(7104), p->y + re(7120), p->y + re(7136), p->y + re(7152), p->wre_table[3], p->wim_table[3]);
fft_n_(64, p->y + re(6912), p->y + re(6976), p->y + re(7040), p->y + re(7104), p->wre_table[5], p->wim_table[5]);
fft_n_(256, p->y + re(6144), p->y + re(6400), p->y + re(6656), p->y + re(6912), p->wre_table[7], p->wim_table[7]);
fft_n_(4, p->y + re(7168), p->y + re(7172), p->y + re(7176), p->y + re(7180), p->wre_table[1], p->wim_table[1]);
fft_n_(8, p->y + re(7168), p->y + re(7176), p->y + re(7184), p->y + re(7192), p->wre_table[2], p->wim_table[2]);
fft_n_(4, p->y + re(7200), p->y + re(7204), p->y + re(7208), p->y + re(7212), p->wre_table[1], p->wim_table[1]);
fft_n_(4, p->y + re(7216), p->y + re(7220), p->y + re(7224), p->y + re(7228), p->wre_table[1], p->wim_table[1]);
fft_n_(16, p->y + re(7168), p->y + re(7184), p->y + re(7200), p->y + re(7216), p->wre_table[3], p->wim_table[3]);
fft_n_(4, p->y + re(7232), p->y + re(7236), p->y + re(7240), p->y + re(7244), p->wre_table[1], p->wim_table[1]);
fft_n_(8, p->y + re(7232), p->y + re(7240), p->y + re(7248), p->y + re(7256), p->wre_table[2], p->wim_table[2]);
fft_n_(4, p->y + re(7264), p->y + re(7268), p->y + re(7272), p->y + re(7276), p->wre_table[1], p->wim_table[1]);
fft_n_(8, p->y + re(7264), p->y + re(7272), p->y + re(7280), p->y + re(7288), p->wre_table[2], p->wim_table[2]);
fft_n_(32, p->y + re(7168), p->y + re(7200), p->y + re(7232), p->y + re(7264), p->wre_table[4], p->wim_table[4]);
fft_n_(4, p->y + re(7296), p->y + re(7300), p->y + re(7304), p->y + re(7308), p->wre_table[1], p->wim_table[1]);
fft_n_(8, p->y + re(7296), p->y + re(7304), p->y + re(7312), p->y + re(7320), p->wre_table[2], p->wim_table[2]);
fft_n_(4, p->y + re(7328), p->y + re(7332), p->y + re(7336), p->y + re(7340), p->wre_table[1], p->wim_table[1]);
fft_n_(4, p->y + re(7344), p->y + re(7348), p->y + re(7352), p->y + re(7356), p->wre_table[1], p->wim_table[1]);
fft_n_(16, p->y + re(7296), p->y + re(7312), p->y + re(7328), p->y + re(7344), p->wre_table[3], p->wim_table[3]);
fft_n_(4, p->y + re(7360), p->y + re(7364), p->y + re(7368), p->y + re(7372), p->wre_table[1], p->wim_table[1]);
fft_n_(8, p->y + re(7360), p->y + re(7368), p->y + re(7376), p->y + re(7384), p->wre_table[2], p->wim_table[2]);
fft_n_(4, p->y + re(7392), p->y + re(7396), p->y + re(7400), p->y + re(7404), p->wre_table[1], p->wim_table[1]);
fft_n_(4, p->y + re(7408), p->y + re(7412), p->y + re(7416), p->y + re(7420), p->wre_table[1], p->wim_table[1]);
fft_n_(16, p->y + re(7360), p->y + re(7376), p->y + re(7392), p->y + re(7408), p->wre_table[3], p->wim_table[3]);
fft_n_(64, p->y + re(7168), p->y + re(7232), p->y + re(7296), p->y + re(7360), p->wre_table[5], p->wim_table[5]);
fft_n_(4, p->y + re(7424), p->y + re(7428), p->y + re(7432), p->y + re(7436), p->wre_table[1], p->wim_table[1]);
fft_n_(8, p->y + re(7424), p->y + re(7432), p->y + re(7440), p->y + re(7448), p->wre_table[2], p->wim_table[2]);
fft_n_(4, p->y + re(7456), p->y + re(7460), p->y + re(7464), p->y + re(7468), p->wre_table[1], p->wim_table[1]);
fft_n_(4, p->y + re(7472), p->y + re(7476), p->y + re(7480), p->y + re(7484), p->wre_table[1], p->wim_table[1]);
fft_n_(16, p->y + re(7424), p->y + re(7440), p->y + re(7456), p->y + re(7472), p->wre_table[3], p->wim_table[3]);
fft_n_(4, p->y + re(7488), p->y + re(7492), p->y + re(7496), p->y + re(7500), p->wre_table[1], p->wim_table[1]);
fft_n_(8, p->y + re(7488), p->y + re(7496), p->y + re(7504), p->y + re(7512), p->wre_table[2], p->wim_table[2]);
fft_n_(4, p->y + re(7520), p->y + re(7524), p->y + re(7528), p->y + re(7532), p->wre_table[1], p->wim_table[1]);
fft_n_(8, p->y + re(7520), p->y + re(7528), p->y + re(7536), p->y + re(7544), p->wre_table[2], p->wim_table[2]);
fft_n_(32, p->y + re(7424), p->y + re(7456), p->y + re(7488), p->y + re(7520), p->wre_table[4], p->wim_table[4]);
fft_n_(4, p->y + re(7552), p->y + re(7556), p->y + re(7560), p->y + re(7564), p->wre_table[1], p->wim_table[1]);
fft_n_(8, p->y + re(7552), p->y + re(7560), p->y + re(7568), p->y + re(7576), p->wre_table[2], p->wim_table[2]);
fft_n_(4, p->y + re(7584), p->y + re(7588), p->y + re(7592), p->y + re(7596), p->wre_table[1], p->wim_table[1]);
fft_n_(4, p->y + re(7600), p->y + re(7604), p->y + re(7608), p->y + re(7612), p->wre_table[1], p->wim_table[1]);
fft_n_(16, p->y + re(7552), p->y + re(7568), p->y + re(7584), p->y + re(7600), p->wre_table[3], p->wim_table[3]);
fft_n_(4, p->y + re(7616), p->y + re(7620), p->y + re(7624), p->y + re(7628), p->wre_table[1], p->wim_table[1]);
fft_n_(8, p->y + re(7616), p->y + re(7624), p->y + re(7632), p->y + re(7640), p->wre_table[2], p->wim_table[2]);
fft_n_(4, p->y + re(7648), p->y + re(7652), p->y + re(7656), p->y + re(7660), p->wre_table[1], p->wim_table[1]);
fft_n_(8, p->y + re(7648), p->y + re(7656), p->y + re(7664), p->y + re(7672), p->wre_table[2], p->wim_table[2]);
fft_n_(32, p->y + re(7552), p->y + re(7584), p->y + re(7616), p->y + re(7648), p->wre_table[4], p->wim_table[4]);
fft_n_(128, p->y + re(7168), p->y + re(7296), p->y + re(7424), p->y + re(7552), p->wre_table[6], p->wim_table[6]);
fft_n_(4, p->y + re(7680), p->y + re(7684), p->y + re(7688), p->y + re(7692), p->wre_table[1], p->wim_table[1]);
fft_n_(8, p->y + re(7680), p->y + re(7688), p->y + re(7696), p->y + re(7704), p->wre_table[2], p->wim_table[2]);
fft_n_(4, p->y + re(7712), p->y + re(7716), p->y + re(7720), p->y + re(7724), p->wre_table[1], p->wim_table[1]);
fft_n_(4, p->y + re(7728), p->y + re(7732), p->y + re(7736), p->y + re(7740), p->wre_table[1], p->wim_table[1]);
fft_n_(16, p->y + re(7680), p->y + re(7696), p->y + re(7712), p->y + re(7728), p->wre_table[3], p->wim_table[3]);
fft_n_(4, p->y + re(7744), p->y + re(7748), p->y + re(7752), p->y + re(7756), p->wre_table[1], p->wim_table[1]);
fft_n_(8, p->y + re(7744), p->y + re(7752), p->y + re(7760), p->y + re(7768), p->wre_table[2], p->wim_table[2]);
fft_n_(4, p->y + re(7776), p->y + re(7780), p->y + re(7784), p->y + re(7788), p->wre_table[1], p->wim_table[1]);
fft_n_(8, p->y + re(7776), p->y + re(7784), p->y + re(7792), p->y + re(7800), p->wre_table[2], p->wim_table[2]);
fft_n_(32, p->y + re(7680), p->y + re(7712), p->y + re(7744), p->y + re(7776), p->wre_table[4], p->wim_table[4]);
fft_n_(4, p->y + re(7808), p->y + re(7812), p->y + re(7816), p->y + re(7820), p->wre_table[1], p->wim_table[1]);
fft_n_(8, p->y + re(7808), p->y + re(7816), p->y + re(7824), p->y + re(7832), p->wre_table[2], p->wim_table[2]);
fft_n_(4, p->y + re(7840), p->y + re(7844), p->y + re(7848), p->y + re(7852), p->wre_table[1], p->wim_table[1]);
fft_n_(4, p->y + re(7856), p->y + re(7860), p->y + re(7864), p->y + re(7868), p->wre_table[1], p->wim_table[1]);
fft_n_(16, p->y + re(7808), p->y + re(7824), p->y + re(7840), p->y + re(7856), p->wre_table[3], p->wim_table[3]);
fft_n_(4, p->y + re(7872), p->y + re(7876), p->y + re(7880), p->y + re(7884), p->wre_table[1], p->wim_table[1]);
fft_n_(8, p->y + re(7872), p->y + re(7880), p->y + re(7888), p->y + re(7896), p->wre_table[2], p->wim_table[2]);
fft_n_(4, p->y + re(7904), p->y + re(7908), p->y + re(7912), p->y + re(7916), p->wre_table[1], p->wim_table[1]);
fft_n_(4, p->y + re(7920), p->y + re(7924), p->y + re(7928), p->y + re(7932), p->wre_table[1], p->wim_table[1]);
fft_n_(16, p->y + re(7872), p->y + re(7888), p->y + re(7904), p->y + re(7920), p->wre_table[3], p->wim_table[3]);
fft_n_(64, p->y + re(7680), p->y + re(7744), p->y + re(7808), p->y + re(7872), p->wre_table[5], p->wim_table[5]);
fft_n_(4, p->y + re(7936), p->y + re(7940), p->y + re(7944), p->y + re(7948), p->wre_table[1], p->wim_table[1]);
fft_n_(8, p->y + re(7936), p->y + re(7944), p->y + re(7952), p->y + re(7960), p->wre_table[2], p->wim_table[2]);
fft_n_(4, p->y + re(7968), p->y + re(7972), p->y + re(7976), p->y + re(7980), p->wre_table[1], p->wim_table[1]);
fft_n_(4, p->y + re(7984), p->y + re(7988), p->y + re(7992), p->y + re(7996), p->wre_table[1], p->wim_table[1]);
fft_n_(16, p->y + re(7936), p->y + re(7952), p->y + re(7968), p->y + re(7984), p->wre_table[3], p->wim_table[3]);
fft_n_(4, p->y + re(8000), p->y + re(8004), p->y + re(8008), p->y + re(8012), p->wre_table[1], p->wim_table[1]);
fft_n_(8, p->y + re(8000), p->y + re(8008), p->y + re(8016), p->y + re(8024), p->wre_table[2], p->wim_table[2]);
fft_n_(4, p->y + re(8032), p->y + re(8036), p->y + re(8040), p->y + re(8044), p->wre_table[1], p->wim_table[1]);
fft_n_(8, p->y + re(8032), p->y + re(8040), p->y + re(8048), p->y + re(8056), p->wre_table[2], p->wim_table[2]);
fft_n_(32, p->y + re(7936), p->y + re(7968), p->y + re(8000), p->y + re(8032), p->wre_table[4], p->wim_table[4]);
fft_n_(4, p->y + re(8064), p->y + re(8068), p->y + re(8072), p->y + re(8076), p->wre_table[1], p->wim_table[1]);
fft_n_(8, p->y + re(8064), p->y + re(8072), p->y + re(8080), p->y + re(8088), p->wre_table[2], p->wim_table[2]);
fft_n_(4, p->y + re(8096), p->y + re(8100), p->y + re(8104), p->y + re(8108), p->wre_table[1], p->wim_table[1]);
fft_n_(4, p->y + re(8112), p->y + re(8116), p->y + re(8120), p->y + re(8124), p->wre_table[1], p->wim_table[1]);
fft_n_(16, p->y + re(8064), p->y + re(8080), p->y + re(8096), p->y + re(8112), p->wre_table[3], p->wim_table[3]);
fft_n_(4, p->y + re(8128), p->y + re(8132), p->y + re(8136), p->y + re(8140), p->wre_table[1], p->wim_table[1]);
fft_n_(8, p->y + re(8128), p->y + re(8136), p->y + re(8144), p->y + re(8152), p->wre_table[2], p->wim_table[2]);
fft_n_(4, p->y + re(8160), p->y + re(8164), p->y + re(8168), p->y + re(8172), p->wre_table[1], p->wim_table[1]);
fft_n_(8, p->y + re(8160), p->y + re(8168), p->y + re(8176), p->y + re(8184), p->wre_table[2], p->wim_table[2]);
fft_n_(32, p->y + re(8064), p->y + re(8096), p->y + re(8128), p->y + re(8160), p->wre_table[4], p->wim_table[4]);
fft_n_(128, p->y + re(7680), p->y + re(7808), p->y + re(7936), p->y + re(8064), p->wre_table[6], p->wim_table[6]);
fft_n_(512, p->y + re(6144), p->y + re(6656), p->y + re(7168), p->y + re(7680), p->wre_table[8], p->wim_table[8]);
fft_n_(2048, p->y + re(0), p->y + re(2048), p->y + re(4096), p->y + re(6144), p->wre_table[10], p->wim_table[10]);
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
	fft_init(&p, 8192, -1);

    srand((unsigned int)time(0));
    for (size_t i = 0; i < p.n; i++) {
        p.x[re(i)] = (float)((double)rand() / (double)RAND_MAX);
        p.x[im(i)] = (float)((double)rand() / (double)RAND_MAX);
    }
    //FILE *fid = fopen("./input1024.dat", "w");
    //for (int i = 0; i < N; i++)
    //    fprintf(fid, "%3.16f, %3.16f\n", x[re(i)], x[im(i)]);
    //fclose(fid);


















	fftwf_complex * fftIn = (fftwf_complex*)fftwf_malloc(sizeof(fftwf_complex) * 8192);
    fftwf_complex * fftOut = (fftwf_complex*)fftwf_malloc(sizeof(fftwf_complex) * 8192);

    float * cast = (float *)fftIn;
	for (size_t i = 0; i < 8192; ++i) {
	    cast[re(i)] = p.x[re(i)];
	    cast[im(i)] = p.x[im(i)];
    }

    fftwf_plan plan = fftwf_plan_dft_1d(8192, fftIn, fftOut, FFTW_FORWARD, FFTW_ESTIMATE);


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
	for (int i = 0; i < 8192; ++i) {
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



