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

ifft_8_1_leaf(p->x + re(0), p->x + re(512), p->x + re(256), p->x + re(768), p->x + re(128), p->x + re(640), p->x + re(896), p->x + re(384), p->y + re(0));
ifft_4_2_leaf(p->x + re(64), p->x + re(576), p->x + re(320), p->x + re(832), p->x + re(960), p->x + re(448), p->x + re(192), p->x + re(704), p->y + re(8));
ifft_8_1_leaf(p->x + re(32), p->x + re(544), p->x + re(288), p->x + re(800), p->x + re(160), p->x + re(672), p->x + re(928), p->x + re(416), p->y + re(16));
ifft_8_1_leaf(p->x + re(992), p->x + re(480), p->x + re(224), p->x + re(736), p->x + re(96), p->x + re(608), p->x + re(864), p->x + re(352), p->y + re(24));
ifft_8_1_leaf(p->x + re(16), p->x + re(528), p->x + re(272), p->x + re(784), p->x + re(144), p->x + re(656), p->x + re(912), p->x + re(400), p->y + re(32));
ifft_4_2_leaf(p->x + re(80), p->x + re(592), p->x + re(336), p->x + re(848), p->x + re(976), p->x + re(464), p->x + re(208), p->x + re(720), p->y + re(40));
ifft_8_1_leaf(p->x + re(1008), p->x + re(496), p->x + re(240), p->x + re(752), p->x + re(112), p->x + re(624), p->x + re(880), p->x + re(368), p->y + re(48));
ifft_4_2_leaf(p->x + re(48), p->x + re(560), p->x + re(304), p->x + re(816), p->x + re(944), p->x + re(432), p->x + re(176), p->x + re(688), p->y + re(56));
ifft_8_1_leaf(p->x + re(8), p->x + re(520), p->x + re(264), p->x + re(776), p->x + re(136), p->x + re(648), p->x + re(904), p->x + re(392), p->y + re(64));
ifft_4_2_leaf(p->x + re(72), p->x + re(584), p->x + re(328), p->x + re(840), p->x + re(968), p->x + re(456), p->x + re(200), p->x + re(712), p->y + re(72));
ifft_8_1_leaf(p->x + re(40), p->x + re(552), p->x + re(296), p->x + re(808), p->x + re(168), p->x + re(680), p->x + re(936), p->x + re(424), p->y + re(80));
ifft_8_1_leaf(p->x + re(1000), p->x + re(488), p->x + re(232), p->x + re(744), p->x + re(104), p->x + re(616), p->x + re(872), p->x + re(360), p->y + re(88));
ifft_8_1_leaf(p->x + re(1016), p->x + re(504), p->x + re(248), p->x + re(760), p->x + re(120), p->x + re(632), p->x + re(888), p->x + re(376), p->y + re(96));
ifft_4_2_leaf(p->x + re(56), p->x + re(568), p->x + re(312), p->x + re(824), p->x + re(952), p->x + re(440), p->x + re(184), p->x + re(696), p->y + re(104));
ifft_8_1_leaf(p->x + re(24), p->x + re(536), p->x + re(280), p->x + re(792), p->x + re(152), p->x + re(664), p->x + re(920), p->x + re(408), p->y + re(112));
ifft_8_1_leaf(p->x + re(984), p->x + re(472), p->x + re(216), p->x + re(728), p->x + re(88), p->x + re(600), p->x + re(856), p->x + re(344), p->y + re(120));
ifft_8_1_leaf(p->x + re(4), p->x + re(516), p->x + re(260), p->x + re(772), p->x + re(132), p->x + re(644), p->x + re(900), p->x + re(388), p->y + re(128));
ifft_4_2_leaf(p->x + re(68), p->x + re(580), p->x + re(324), p->x + re(836), p->x + re(964), p->x + re(452), p->x + re(196), p->x + re(708), p->y + re(136));
ifft_8_1_leaf(p->x + re(36), p->x + re(548), p->x + re(292), p->x + re(804), p->x + re(164), p->x + re(676), p->x + re(932), p->x + re(420), p->y + re(144));
ifft_8_1_leaf(p->x + re(996), p->x + re(484), p->x + re(228), p->x + re(740), p->x + re(100), p->x + re(612), p->x + re(868), p->x + re(356), p->y + re(152));
ifft_8_1_leaf(p->x + re(20), p->x + re(532), p->x + re(276), p->x + re(788), p->x + re(148), p->x + re(660), p->x + re(916), p->x + re(404), p->y + re(160));
ifft_4_2_leaf(p->x + re(84), p->x + re(596), p->x + re(340), p->x + re(852), p->x + re(980), p->x + re(468), p->x + re(212), p->x + re(724), p->y + re(168));
ifft_8_1_leaf(p->x + re(1012), p->x + re(500), p->x + re(244), p->x + re(756), p->x + re(116), p->x + re(628), p->x + re(884), p->x + re(372), p->y + re(176));
ifft_4_2_leaf(p->x + re(52), p->x + re(564), p->x + re(308), p->x + re(820), p->x + re(948), p->x + re(436), p->x + re(180), p->x + re(692), p->y + re(184));
ifft_8_1_leaf(p->x + re(1020), p->x + re(508), p->x + re(252), p->x + re(764), p->x + re(124), p->x + re(636), p->x + re(892), p->x + re(380), p->y + re(192));
ifft_4_2_leaf(p->x + re(60), p->x + re(572), p->x + re(316), p->x + re(828), p->x + re(956), p->x + re(444), p->x + re(188), p->x + re(700), p->y + re(200));
ifft_8_1_leaf(p->x + re(28), p->x + re(540), p->x + re(284), p->x + re(796), p->x + re(156), p->x + re(668), p->x + re(924), p->x + re(412), p->y + re(208));
ifft_8_1_leaf(p->x + re(988), p->x + re(476), p->x + re(220), p->x + re(732), p->x + re(92), p->x + re(604), p->x + re(860), p->x + re(348), p->y + re(216));
ifft_8_1_leaf(p->x + re(12), p->x + re(524), p->x + re(268), p->x + re(780), p->x + re(140), p->x + re(652), p->x + re(908), p->x + re(396), p->y + re(224));
ifft_4_2_leaf(p->x + re(76), p->x + re(588), p->x + re(332), p->x + re(844), p->x + re(972), p->x + re(460), p->x + re(204), p->x + re(716), p->y + re(232));
ifft_8_1_leaf(p->x + re(1004), p->x + re(492), p->x + re(236), p->x + re(748), p->x + re(108), p->x + re(620), p->x + re(876), p->x + re(364), p->y + re(240));
ifft_4_2_leaf(p->x + re(44), p->x + re(556), p->x + re(300), p->x + re(812), p->x + re(940), p->x + re(428), p->x + re(172), p->x + re(684), p->y + re(248));
ifft_8_1_leaf(p->x + re(2), p->x + re(514), p->x + re(258), p->x + re(770), p->x + re(130), p->x + re(642), p->x + re(898), p->x + re(386), p->y + re(256));
ifft_4_2_leaf(p->x + re(66), p->x + re(578), p->x + re(322), p->x + re(834), p->x + re(962), p->x + re(450), p->x + re(194), p->x + re(706), p->y + re(264));
ifft_8_1_leaf(p->x + re(34), p->x + re(546), p->x + re(290), p->x + re(802), p->x + re(162), p->x + re(674), p->x + re(930), p->x + re(418), p->y + re(272));
ifft_8_1_leaf(p->x + re(994), p->x + re(482), p->x + re(226), p->x + re(738), p->x + re(98), p->x + re(610), p->x + re(866), p->x + re(354), p->y + re(280));
ifft_8_1_leaf(p->x + re(18), p->x + re(530), p->x + re(274), p->x + re(786), p->x + re(146), p->x + re(658), p->x + re(914), p->x + re(402), p->y + re(288));
ifft_4_2_leaf(p->x + re(82), p->x + re(594), p->x + re(338), p->x + re(850), p->x + re(978), p->x + re(466), p->x + re(210), p->x + re(722), p->y + re(296));
ifft_8_1_leaf(p->x + re(1010), p->x + re(498), p->x + re(242), p->x + re(754), p->x + re(114), p->x + re(626), p->x + re(882), p->x + re(370), p->y + re(304));
ifft_4_2_leaf(p->x + re(50), p->x + re(562), p->x + re(306), p->x + re(818), p->x + re(946), p->x + re(434), p->x + re(178), p->x + re(690), p->y + re(312));
ifft_8_1_leaf(p->x + re(10), p->x + re(522), p->x + re(266), p->x + re(778), p->x + re(138), p->x + re(650), p->x + re(906), p->x + re(394), p->y + re(320));
ifft_4_2_leaf(p->x + re(74), p->x + re(586), p->x + re(330), p->x + re(842), p->x + re(970), p->x + re(458), p->x + re(202), p->x + re(714), p->y + re(328));
ifft_8_1_leaf(p->x + re(42), p->x + re(554), p->x + re(298), p->x + re(810), p->x + re(170), p->x + re(682), p->x + re(938), p->x + re(426), p->y + re(336));
ifft_8_1_leaf(p->x + re(1002), p->x + re(490), p->x + re(234), p->x + re(746), p->x + re(106), p->x + re(618), p->x + re(874), p->x + re(362), p->y + re(344));
ifft_8_1_leaf(p->x + re(1018), p->x + re(506), p->x + re(250), p->x + re(762), p->x + re(122), p->x + re(634), p->x + re(890), p->x + re(378), p->y + re(352));
ifft_4_2_leaf(p->x + re(58), p->x + re(570), p->x + re(314), p->x + re(826), p->x + re(954), p->x + re(442), p->x + re(186), p->x + re(698), p->y + re(360));
ifft_8_1_leaf(p->x + re(26), p->x + re(538), p->x + re(282), p->x + re(794), p->x + re(154), p->x + re(666), p->x + re(922), p->x + re(410), p->y + re(368));
ifft_8_1_leaf(p->x + re(986), p->x + re(474), p->x + re(218), p->x + re(730), p->x + re(90), p->x + re(602), p->x + re(858), p->x + re(346), p->y + re(376));
ifft_8_1_leaf(p->x + re(1022), p->x + re(510), p->x + re(254), p->x + re(766), p->x + re(126), p->x + re(638), p->x + re(894), p->x + re(382), p->y + re(384));
ifft_4_2_leaf(p->x + re(62), p->x + re(574), p->x + re(318), p->x + re(830), p->x + re(958), p->x + re(446), p->x + re(190), p->x + re(702), p->y + re(392));
ifft_8_1_leaf(p->x + re(30), p->x + re(542), p->x + re(286), p->x + re(798), p->x + re(158), p->x + re(670), p->x + re(926), p->x + re(414), p->y + re(400));
ifft_8_1_leaf(p->x + re(990), p->x + re(478), p->x + re(222), p->x + re(734), p->x + re(94), p->x + re(606), p->x + re(862), p->x + re(350), p->y + re(408));
ifft_8_1_leaf(p->x + re(14), p->x + re(526), p->x + re(270), p->x + re(782), p->x + re(142), p->x + re(654), p->x + re(910), p->x + re(398), p->y + re(416));
ifft_4_2_leaf(p->x + re(78), p->x + re(590), p->x + re(334), p->x + re(846), p->x + re(974), p->x + re(462), p->x + re(206), p->x + re(718), p->y + re(424));
ifft_8_1_leaf(p->x + re(1006), p->x + re(494), p->x + re(238), p->x + re(750), p->x + re(110), p->x + re(622), p->x + re(878), p->x + re(366), p->y + re(432));
ifft_4_2_leaf(p->x + re(46), p->x + re(558), p->x + re(302), p->x + re(814), p->x + re(942), p->x + re(430), p->x + re(174), p->x + re(686), p->y + re(440));
ifft_8_1_leaf(p->x + re(6), p->x + re(518), p->x + re(262), p->x + re(774), p->x + re(134), p->x + re(646), p->x + re(902), p->x + re(390), p->y + re(448));
ifft_4_2_leaf(p->x + re(70), p->x + re(582), p->x + re(326), p->x + re(838), p->x + re(966), p->x + re(454), p->x + re(198), p->x + re(710), p->y + re(456));
ifft_8_1_leaf(p->x + re(38), p->x + re(550), p->x + re(294), p->x + re(806), p->x + re(166), p->x + re(678), p->x + re(934), p->x + re(422), p->y + re(464));
ifft_8_1_leaf(p->x + re(998), p->x + re(486), p->x + re(230), p->x + re(742), p->x + re(102), p->x + re(614), p->x + re(870), p->x + re(358), p->y + re(472));
ifft_8_1_leaf(p->x + re(1014), p->x + re(502), p->x + re(246), p->x + re(758), p->x + re(118), p->x + re(630), p->x + re(886), p->x + re(374), p->y + re(480));
ifft_4_2_leaf(p->x + re(54), p->x + re(566), p->x + re(310), p->x + re(822), p->x + re(950), p->x + re(438), p->x + re(182), p->x + re(694), p->y + re(488));
ifft_8_1_leaf(p->x + re(22), p->x + re(534), p->x + re(278), p->x + re(790), p->x + re(150), p->x + re(662), p->x + re(918), p->x + re(406), p->y + re(496));
ifft_8_1_leaf(p->x + re(982), p->x + re(470), p->x + re(214), p->x + re(726), p->x + re(86), p->x + re(598), p->x + re(854), p->x + re(342), p->y + re(504));
ifft_8_1_leaf(p->x + re(1), p->x + re(513), p->x + re(257), p->x + re(769), p->x + re(129), p->x + re(641), p->x + re(897), p->x + re(385), p->y + re(512));
ifft_4_2_leaf(p->x + re(65), p->x + re(577), p->x + re(321), p->x + re(833), p->x + re(961), p->x + re(449), p->x + re(193), p->x + re(705), p->y + re(520));
ifft_8_1_leaf(p->x + re(33), p->x + re(545), p->x + re(289), p->x + re(801), p->x + re(161), p->x + re(673), p->x + re(929), p->x + re(417), p->y + re(528));
ifft_8_1_leaf(p->x + re(993), p->x + re(481), p->x + re(225), p->x + re(737), p->x + re(97), p->x + re(609), p->x + re(865), p->x + re(353), p->y + re(536));
ifft_8_1_leaf(p->x + re(17), p->x + re(529), p->x + re(273), p->x + re(785), p->x + re(145), p->x + re(657), p->x + re(913), p->x + re(401), p->y + re(544));
ifft_4_2_leaf(p->x + re(81), p->x + re(593), p->x + re(337), p->x + re(849), p->x + re(977), p->x + re(465), p->x + re(209), p->x + re(721), p->y + re(552));
ifft_8_1_leaf(p->x + re(1009), p->x + re(497), p->x + re(241), p->x + re(753), p->x + re(113), p->x + re(625), p->x + re(881), p->x + re(369), p->y + re(560));
ifft_4_2_leaf(p->x + re(49), p->x + re(561), p->x + re(305), p->x + re(817), p->x + re(945), p->x + re(433), p->x + re(177), p->x + re(689), p->y + re(568));
ifft_8_1_leaf(p->x + re(9), p->x + re(521), p->x + re(265), p->x + re(777), p->x + re(137), p->x + re(649), p->x + re(905), p->x + re(393), p->y + re(576));
ifft_4_2_leaf(p->x + re(73), p->x + re(585), p->x + re(329), p->x + re(841), p->x + re(969), p->x + re(457), p->x + re(201), p->x + re(713), p->y + re(584));
ifft_8_1_leaf(p->x + re(41), p->x + re(553), p->x + re(297), p->x + re(809), p->x + re(169), p->x + re(681), p->x + re(937), p->x + re(425), p->y + re(592));
ifft_8_1_leaf(p->x + re(1001), p->x + re(489), p->x + re(233), p->x + re(745), p->x + re(105), p->x + re(617), p->x + re(873), p->x + re(361), p->y + re(600));
ifft_8_1_leaf(p->x + re(1017), p->x + re(505), p->x + re(249), p->x + re(761), p->x + re(121), p->x + re(633), p->x + re(889), p->x + re(377), p->y + re(608));
ifft_4_2_leaf(p->x + re(57), p->x + re(569), p->x + re(313), p->x + re(825), p->x + re(953), p->x + re(441), p->x + re(185), p->x + re(697), p->y + re(616));
ifft_8_1_leaf(p->x + re(25), p->x + re(537), p->x + re(281), p->x + re(793), p->x + re(153), p->x + re(665), p->x + re(921), p->x + re(409), p->y + re(624));
ifft_8_1_leaf(p->x + re(985), p->x + re(473), p->x + re(217), p->x + re(729), p->x + re(89), p->x + re(601), p->x + re(857), p->x + re(345), p->y + re(632));
ifft_8_1_leaf(p->x + re(5), p->x + re(517), p->x + re(261), p->x + re(773), p->x + re(133), p->x + re(645), p->x + re(901), p->x + re(389), p->y + re(640));
ifft_4_2_leaf(p->x + re(69), p->x + re(581), p->x + re(325), p->x + re(837), p->x + re(965), p->x + re(453), p->x + re(197), p->x + re(709), p->y + re(648));
ifft_8_1_leaf(p->x + re(37), p->x + re(549), p->x + re(293), p->x + re(805), p->x + re(165), p->x + re(677), p->x + re(933), p->x + re(421), p->y + re(656));
ifft_8_1_leaf(p->x + re(997), p->x + re(485), p->x + re(229), p->x + re(741), p->x + re(101), p->x + re(613), p->x + re(869), p->x + re(357), p->y + re(664));
ifft_8_1_leaf(p->x + re(21), p->x + re(533), p->x + re(277), p->x + re(789), p->x + re(149), p->x + re(661), p->x + re(917), p->x + re(405), p->y + re(672));
ifft_4_2_leaf(p->x + re(85), p->x + re(597), p->x + re(341), p->x + re(853), p->x + re(981), p->x + re(469), p->x + re(213), p->x + re(725), p->y + re(680));
ifft_8_1_leaf(p->x + re(1013), p->x + re(501), p->x + re(245), p->x + re(757), p->x + re(117), p->x + re(629), p->x + re(885), p->x + re(373), p->y + re(688));
ifft_4_2_leaf(p->x + re(53), p->x + re(565), p->x + re(309), p->x + re(821), p->x + re(949), p->x + re(437), p->x + re(181), p->x + re(693), p->y + re(696));
ifft_8_1_leaf(p->x + re(1021), p->x + re(509), p->x + re(253), p->x + re(765), p->x + re(125), p->x + re(637), p->x + re(893), p->x + re(381), p->y + re(704));
ifft_4_2_leaf(p->x + re(61), p->x + re(573), p->x + re(317), p->x + re(829), p->x + re(957), p->x + re(445), p->x + re(189), p->x + re(701), p->y + re(712));
ifft_8_1_leaf(p->x + re(29), p->x + re(541), p->x + re(285), p->x + re(797), p->x + re(157), p->x + re(669), p->x + re(925), p->x + re(413), p->y + re(720));
ifft_8_1_leaf(p->x + re(989), p->x + re(477), p->x + re(221), p->x + re(733), p->x + re(93), p->x + re(605), p->x + re(861), p->x + re(349), p->y + re(728));
ifft_8_1_leaf(p->x + re(13), p->x + re(525), p->x + re(269), p->x + re(781), p->x + re(141), p->x + re(653), p->x + re(909), p->x + re(397), p->y + re(736));
ifft_4_2_leaf(p->x + re(77), p->x + re(589), p->x + re(333), p->x + re(845), p->x + re(973), p->x + re(461), p->x + re(205), p->x + re(717), p->y + re(744));
ifft_8_1_leaf(p->x + re(1005), p->x + re(493), p->x + re(237), p->x + re(749), p->x + re(109), p->x + re(621), p->x + re(877), p->x + re(365), p->y + re(752));
ifft_4_2_leaf(p->x + re(45), p->x + re(557), p->x + re(301), p->x + re(813), p->x + re(941), p->x + re(429), p->x + re(173), p->x + re(685), p->y + re(760));
ifft_8_1_leaf(p->x + re(1023), p->x + re(511), p->x + re(255), p->x + re(767), p->x + re(127), p->x + re(639), p->x + re(895), p->x + re(383), p->y + re(768));
ifft_4_2_leaf(p->x + re(63), p->x + re(575), p->x + re(319), p->x + re(831), p->x + re(959), p->x + re(447), p->x + re(191), p->x + re(703), p->y + re(776));
ifft_8_1_leaf(p->x + re(31), p->x + re(543), p->x + re(287), p->x + re(799), p->x + re(159), p->x + re(671), p->x + re(927), p->x + re(415), p->y + re(784));
ifft_8_1_leaf(p->x + re(991), p->x + re(479), p->x + re(223), p->x + re(735), p->x + re(95), p->x + re(607), p->x + re(863), p->x + re(351), p->y + re(792));
ifft_8_1_leaf(p->x + re(15), p->x + re(527), p->x + re(271), p->x + re(783), p->x + re(143), p->x + re(655), p->x + re(911), p->x + re(399), p->y + re(800));
ifft_4_2_leaf(p->x + re(79), p->x + re(591), p->x + re(335), p->x + re(847), p->x + re(975), p->x + re(463), p->x + re(207), p->x + re(719), p->y + re(808));
ifft_8_1_leaf(p->x + re(1007), p->x + re(495), p->x + re(239), p->x + re(751), p->x + re(111), p->x + re(623), p->x + re(879), p->x + re(367), p->y + re(816));
ifft_4_2_leaf(p->x + re(47), p->x + re(559), p->x + re(303), p->x + re(815), p->x + re(943), p->x + re(431), p->x + re(175), p->x + re(687), p->y + re(824));
ifft_8_1_leaf(p->x + re(7), p->x + re(519), p->x + re(263), p->x + re(775), p->x + re(135), p->x + re(647), p->x + re(903), p->x + re(391), p->y + re(832));
ifft_4_2_leaf(p->x + re(71), p->x + re(583), p->x + re(327), p->x + re(839), p->x + re(967), p->x + re(455), p->x + re(199), p->x + re(711), p->y + re(840));
ifft_8_1_leaf(p->x + re(39), p->x + re(551), p->x + re(295), p->x + re(807), p->x + re(167), p->x + re(679), p->x + re(935), p->x + re(423), p->y + re(848));
ifft_8_1_leaf(p->x + re(999), p->x + re(487), p->x + re(231), p->x + re(743), p->x + re(103), p->x + re(615), p->x + re(871), p->x + re(359), p->y + re(856));
ifft_8_1_leaf(p->x + re(1015), p->x + re(503), p->x + re(247), p->x + re(759), p->x + re(119), p->x + re(631), p->x + re(887), p->x + re(375), p->y + re(864));
ifft_4_2_leaf(p->x + re(55), p->x + re(567), p->x + re(311), p->x + re(823), p->x + re(951), p->x + re(439), p->x + re(183), p->x + re(695), p->y + re(872));
ifft_8_1_leaf(p->x + re(23), p->x + re(535), p->x + re(279), p->x + re(791), p->x + re(151), p->x + re(663), p->x + re(919), p->x + re(407), p->y + re(880));
ifft_8_1_leaf(p->x + re(983), p->x + re(471), p->x + re(215), p->x + re(727), p->x + re(87), p->x + re(599), p->x + re(855), p->x + re(343), p->y + re(888));
ifft_8_1_leaf(p->x + re(3), p->x + re(515), p->x + re(259), p->x + re(771), p->x + re(131), p->x + re(643), p->x + re(899), p->x + re(387), p->y + re(896));
ifft_4_2_leaf(p->x + re(67), p->x + re(579), p->x + re(323), p->x + re(835), p->x + re(963), p->x + re(451), p->x + re(195), p->x + re(707), p->y + re(904));
ifft_8_1_leaf(p->x + re(35), p->x + re(547), p->x + re(291), p->x + re(803), p->x + re(163), p->x + re(675), p->x + re(931), p->x + re(419), p->y + re(912));
ifft_8_1_leaf(p->x + re(995), p->x + re(483), p->x + re(227), p->x + re(739), p->x + re(99), p->x + re(611), p->x + re(867), p->x + re(355), p->y + re(920));
ifft_8_1_leaf(p->x + re(19), p->x + re(531), p->x + re(275), p->x + re(787), p->x + re(147), p->x + re(659), p->x + re(915), p->x + re(403), p->y + re(928));
ifft_4_2_leaf(p->x + re(83), p->x + re(595), p->x + re(339), p->x + re(851), p->x + re(979), p->x + re(467), p->x + re(211), p->x + re(723), p->y + re(936));
ifft_8_1_leaf(p->x + re(1011), p->x + re(499), p->x + re(243), p->x + re(755), p->x + re(115), p->x + re(627), p->x + re(883), p->x + re(371), p->y + re(944));
ifft_4_2_leaf(p->x + re(51), p->x + re(563), p->x + re(307), p->x + re(819), p->x + re(947), p->x + re(435), p->x + re(179), p->x + re(691), p->y + re(952));
ifft_8_1_leaf(p->x + re(1019), p->x + re(507), p->x + re(251), p->x + re(763), p->x + re(123), p->x + re(635), p->x + re(891), p->x + re(379), p->y + re(960));
ifft_4_2_leaf(p->x + re(59), p->x + re(571), p->x + re(315), p->x + re(827), p->x + re(955), p->x + re(443), p->x + re(187), p->x + re(699), p->y + re(968));
ifft_8_1_leaf(p->x + re(27), p->x + re(539), p->x + re(283), p->x + re(795), p->x + re(155), p->x + re(667), p->x + re(923), p->x + re(411), p->y + re(976));
ifft_8_1_leaf(p->x + re(987), p->x + re(475), p->x + re(219), p->x + re(731), p->x + re(91), p->x + re(603), p->x + re(859), p->x + re(347), p->y + re(984));
ifft_8_1_leaf(p->x + re(11), p->x + re(523), p->x + re(267), p->x + re(779), p->x + re(139), p->x + re(651), p->x + re(907), p->x + re(395), p->y + re(992));
ifft_4_2_leaf(p->x + re(75), p->x + re(587), p->x + re(331), p->x + re(843), p->x + re(971), p->x + re(459), p->x + re(203), p->x + re(715), p->y + re(1000));
ifft_8_1_leaf(p->x + re(1003), p->x + re(491), p->x + re(235), p->x + re(747), p->x + re(107), p->x + re(619), p->x + re(875), p->x + re(363), p->y + re(1008));
ifft_4_2_leaf(p->x + re(43), p->x + re(555), p->x + re(299), p->x + re(811), p->x + re(939), p->x + re(427), p->x + re(171), p->x + re(683), p->y + re(1016));
ifft_n_(4, p->y + re(0), p->y + re(4), p->y + re(8), p->y + re(12), p->wre_table[1], p->wim_table[1]);
ifft_n_(8, p->y + re(0), p->y + re(8), p->y + re(16), p->y + re(24), p->wre_table[2], p->wim_table[2]);
ifft_n_(4, p->y + re(32), p->y + re(36), p->y + re(40), p->y + re(44), p->wre_table[1], p->wim_table[1]);
ifft_n_(4, p->y + re(48), p->y + re(52), p->y + re(56), p->y + re(60), p->wre_table[1], p->wim_table[1]);
ifft_n_(16, p->y + re(0), p->y + re(16), p->y + re(32), p->y + re(48), p->wre_table[3], p->wim_table[3]);
ifft_n_(4, p->y + re(64), p->y + re(68), p->y + re(72), p->y + re(76), p->wre_table[1], p->wim_table[1]);
ifft_n_(8, p->y + re(64), p->y + re(72), p->y + re(80), p->y + re(88), p->wre_table[2], p->wim_table[2]);
ifft_n_(4, p->y + re(96), p->y + re(100), p->y + re(104), p->y + re(108), p->wre_table[1], p->wim_table[1]);
ifft_n_(8, p->y + re(96), p->y + re(104), p->y + re(112), p->y + re(120), p->wre_table[2], p->wim_table[2]);
ifft_n_(32, p->y + re(0), p->y + re(32), p->y + re(64), p->y + re(96), p->wre_table[4], p->wim_table[4]);
ifft_n_(4, p->y + re(128), p->y + re(132), p->y + re(136), p->y + re(140), p->wre_table[1], p->wim_table[1]);
ifft_n_(8, p->y + re(128), p->y + re(136), p->y + re(144), p->y + re(152), p->wre_table[2], p->wim_table[2]);
ifft_n_(4, p->y + re(160), p->y + re(164), p->y + re(168), p->y + re(172), p->wre_table[1], p->wim_table[1]);
ifft_n_(4, p->y + re(176), p->y + re(180), p->y + re(184), p->y + re(188), p->wre_table[1], p->wim_table[1]);
ifft_n_(16, p->y + re(128), p->y + re(144), p->y + re(160), p->y + re(176), p->wre_table[3], p->wim_table[3]);
ifft_n_(4, p->y + re(192), p->y + re(196), p->y + re(200), p->y + re(204), p->wre_table[1], p->wim_table[1]);
ifft_n_(8, p->y + re(192), p->y + re(200), p->y + re(208), p->y + re(216), p->wre_table[2], p->wim_table[2]);
ifft_n_(4, p->y + re(224), p->y + re(228), p->y + re(232), p->y + re(236), p->wre_table[1], p->wim_table[1]);
ifft_n_(4, p->y + re(240), p->y + re(244), p->y + re(248), p->y + re(252), p->wre_table[1], p->wim_table[1]);
ifft_n_(16, p->y + re(192), p->y + re(208), p->y + re(224), p->y + re(240), p->wre_table[3], p->wim_table[3]);
ifft_n_(64, p->y + re(0), p->y + re(64), p->y + re(128), p->y + re(192), p->wre_table[5], p->wim_table[5]);
ifft_n_(4, p->y + re(256), p->y + re(260), p->y + re(264), p->y + re(268), p->wre_table[1], p->wim_table[1]);
ifft_n_(8, p->y + re(256), p->y + re(264), p->y + re(272), p->y + re(280), p->wre_table[2], p->wim_table[2]);
ifft_n_(4, p->y + re(288), p->y + re(292), p->y + re(296), p->y + re(300), p->wre_table[1], p->wim_table[1]);
ifft_n_(4, p->y + re(304), p->y + re(308), p->y + re(312), p->y + re(316), p->wre_table[1], p->wim_table[1]);
ifft_n_(16, p->y + re(256), p->y + re(272), p->y + re(288), p->y + re(304), p->wre_table[3], p->wim_table[3]);
ifft_n_(4, p->y + re(320), p->y + re(324), p->y + re(328), p->y + re(332), p->wre_table[1], p->wim_table[1]);
ifft_n_(8, p->y + re(320), p->y + re(328), p->y + re(336), p->y + re(344), p->wre_table[2], p->wim_table[2]);
ifft_n_(4, p->y + re(352), p->y + re(356), p->y + re(360), p->y + re(364), p->wre_table[1], p->wim_table[1]);
ifft_n_(8, p->y + re(352), p->y + re(360), p->y + re(368), p->y + re(376), p->wre_table[2], p->wim_table[2]);
ifft_n_(32, p->y + re(256), p->y + re(288), p->y + re(320), p->y + re(352), p->wre_table[4], p->wim_table[4]);
ifft_n_(4, p->y + re(384), p->y + re(388), p->y + re(392), p->y + re(396), p->wre_table[1], p->wim_table[1]);
ifft_n_(8, p->y + re(384), p->y + re(392), p->y + re(400), p->y + re(408), p->wre_table[2], p->wim_table[2]);
ifft_n_(4, p->y + re(416), p->y + re(420), p->y + re(424), p->y + re(428), p->wre_table[1], p->wim_table[1]);
ifft_n_(4, p->y + re(432), p->y + re(436), p->y + re(440), p->y + re(444), p->wre_table[1], p->wim_table[1]);
ifft_n_(16, p->y + re(384), p->y + re(400), p->y + re(416), p->y + re(432), p->wre_table[3], p->wim_table[3]);
ifft_n_(4, p->y + re(448), p->y + re(452), p->y + re(456), p->y + re(460), p->wre_table[1], p->wim_table[1]);
ifft_n_(8, p->y + re(448), p->y + re(456), p->y + re(464), p->y + re(472), p->wre_table[2], p->wim_table[2]);
ifft_n_(4, p->y + re(480), p->y + re(484), p->y + re(488), p->y + re(492), p->wre_table[1], p->wim_table[1]);
ifft_n_(8, p->y + re(480), p->y + re(488), p->y + re(496), p->y + re(504), p->wre_table[2], p->wim_table[2]);
ifft_n_(32, p->y + re(384), p->y + re(416), p->y + re(448), p->y + re(480), p->wre_table[4], p->wim_table[4]);
ifft_n_(128, p->y + re(0), p->y + re(128), p->y + re(256), p->y + re(384), p->wre_table[6], p->wim_table[6]);
ifft_n_(4, p->y + re(512), p->y + re(516), p->y + re(520), p->y + re(524), p->wre_table[1], p->wim_table[1]);
ifft_n_(8, p->y + re(512), p->y + re(520), p->y + re(528), p->y + re(536), p->wre_table[2], p->wim_table[2]);
ifft_n_(4, p->y + re(544), p->y + re(548), p->y + re(552), p->y + re(556), p->wre_table[1], p->wim_table[1]);
ifft_n_(4, p->y + re(560), p->y + re(564), p->y + re(568), p->y + re(572), p->wre_table[1], p->wim_table[1]);
ifft_n_(16, p->y + re(512), p->y + re(528), p->y + re(544), p->y + re(560), p->wre_table[3], p->wim_table[3]);
ifft_n_(4, p->y + re(576), p->y + re(580), p->y + re(584), p->y + re(588), p->wre_table[1], p->wim_table[1]);
ifft_n_(8, p->y + re(576), p->y + re(584), p->y + re(592), p->y + re(600), p->wre_table[2], p->wim_table[2]);
ifft_n_(4, p->y + re(608), p->y + re(612), p->y + re(616), p->y + re(620), p->wre_table[1], p->wim_table[1]);
ifft_n_(8, p->y + re(608), p->y + re(616), p->y + re(624), p->y + re(632), p->wre_table[2], p->wim_table[2]);
ifft_n_(32, p->y + re(512), p->y + re(544), p->y + re(576), p->y + re(608), p->wre_table[4], p->wim_table[4]);
ifft_n_(4, p->y + re(640), p->y + re(644), p->y + re(648), p->y + re(652), p->wre_table[1], p->wim_table[1]);
ifft_n_(8, p->y + re(640), p->y + re(648), p->y + re(656), p->y + re(664), p->wre_table[2], p->wim_table[2]);
ifft_n_(4, p->y + re(672), p->y + re(676), p->y + re(680), p->y + re(684), p->wre_table[1], p->wim_table[1]);
ifft_n_(4, p->y + re(688), p->y + re(692), p->y + re(696), p->y + re(700), p->wre_table[1], p->wim_table[1]);
ifft_n_(16, p->y + re(640), p->y + re(656), p->y + re(672), p->y + re(688), p->wre_table[3], p->wim_table[3]);
ifft_n_(4, p->y + re(704), p->y + re(708), p->y + re(712), p->y + re(716), p->wre_table[1], p->wim_table[1]);
ifft_n_(8, p->y + re(704), p->y + re(712), p->y + re(720), p->y + re(728), p->wre_table[2], p->wim_table[2]);
ifft_n_(4, p->y + re(736), p->y + re(740), p->y + re(744), p->y + re(748), p->wre_table[1], p->wim_table[1]);
ifft_n_(4, p->y + re(752), p->y + re(756), p->y + re(760), p->y + re(764), p->wre_table[1], p->wim_table[1]);
ifft_n_(16, p->y + re(704), p->y + re(720), p->y + re(736), p->y + re(752), p->wre_table[3], p->wim_table[3]);
ifft_n_(64, p->y + re(512), p->y + re(576), p->y + re(640), p->y + re(704), p->wre_table[5], p->wim_table[5]);
ifft_n_(4, p->y + re(768), p->y + re(772), p->y + re(776), p->y + re(780), p->wre_table[1], p->wim_table[1]);
ifft_n_(8, p->y + re(768), p->y + re(776), p->y + re(784), p->y + re(792), p->wre_table[2], p->wim_table[2]);
ifft_n_(4, p->y + re(800), p->y + re(804), p->y + re(808), p->y + re(812), p->wre_table[1], p->wim_table[1]);
ifft_n_(4, p->y + re(816), p->y + re(820), p->y + re(824), p->y + re(828), p->wre_table[1], p->wim_table[1]);
ifft_n_(16, p->y + re(768), p->y + re(784), p->y + re(800), p->y + re(816), p->wre_table[3], p->wim_table[3]);
ifft_n_(4, p->y + re(832), p->y + re(836), p->y + re(840), p->y + re(844), p->wre_table[1], p->wim_table[1]);
ifft_n_(8, p->y + re(832), p->y + re(840), p->y + re(848), p->y + re(856), p->wre_table[2], p->wim_table[2]);
ifft_n_(4, p->y + re(864), p->y + re(868), p->y + re(872), p->y + re(876), p->wre_table[1], p->wim_table[1]);
ifft_n_(8, p->y + re(864), p->y + re(872), p->y + re(880), p->y + re(888), p->wre_table[2], p->wim_table[2]);
ifft_n_(32, p->y + re(768), p->y + re(800), p->y + re(832), p->y + re(864), p->wre_table[4], p->wim_table[4]);
ifft_n_(4, p->y + re(896), p->y + re(900), p->y + re(904), p->y + re(908), p->wre_table[1], p->wim_table[1]);
ifft_n_(8, p->y + re(896), p->y + re(904), p->y + re(912), p->y + re(920), p->wre_table[2], p->wim_table[2]);
ifft_n_(4, p->y + re(928), p->y + re(932), p->y + re(936), p->y + re(940), p->wre_table[1], p->wim_table[1]);
ifft_n_(4, p->y + re(944), p->y + re(948), p->y + re(952), p->y + re(956), p->wre_table[1], p->wim_table[1]);
ifft_n_(16, p->y + re(896), p->y + re(912), p->y + re(928), p->y + re(944), p->wre_table[3], p->wim_table[3]);
ifft_n_(4, p->y + re(960), p->y + re(964), p->y + re(968), p->y + re(972), p->wre_table[1], p->wim_table[1]);
ifft_n_(8, p->y + re(960), p->y + re(968), p->y + re(976), p->y + re(984), p->wre_table[2], p->wim_table[2]);
ifft_n_(4, p->y + re(992), p->y + re(996), p->y + re(1000), p->y + re(1004), p->wre_table[1], p->wim_table[1]);
ifft_n_(4, p->y + re(1008), p->y + re(1012), p->y + re(1016), p->y + re(1020), p->wre_table[1], p->wim_table[1]);
ifft_n_(16, p->y + re(960), p->y + re(976), p->y + re(992), p->y + re(1008), p->wre_table[3], p->wim_table[3]);
ifft_n_(64, p->y + re(768), p->y + re(832), p->y + re(896), p->y + re(960), p->wre_table[5], p->wim_table[5]);
ifft_n_(256, p->y + re(0), p->y + re(256), p->y + re(512), p->y + re(768), p->wre_table[7], p->wim_table[7]);
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
	fft_init(&p, 1024, 1);

    srand((unsigned int)time(0));
    for (size_t i = 0; i < p.n; i++) {
        p.x[re(i)] = (float)((double)rand() / (double)RAND_MAX);
        p.x[im(i)] = (float)((double)rand() / (double)RAND_MAX);
    }
    //FILE *fid = fopen("./input1024.dat", "w");
    //for (int i = 0; i < N; i++)
    //    fprintf(fid, "%3.16f, %3.16f\n", x[re(i)], x[im(i)]);
    //fclose(fid);


















	fftwf_complex * fftIn = (fftwf_complex*)fftwf_malloc(sizeof(fftwf_complex) * 1024);
    fftwf_complex * fftOut = (fftwf_complex*)fftwf_malloc(sizeof(fftwf_complex) * 1024);

    float * cast = (float *)fftIn;
	for (size_t i = 0; i < 1024; ++i) {
	    cast[re(i)] = p.x[re(i)];
	    cast[im(i)] = p.x[im(i)];
    }

    fftwf_plan plan = fftwf_plan_dft_1d(1024, fftIn, fftOut, FFTW_BACKWARD, FFTW_ESTIMATE);


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
	for (int i = 0; i < 1024; ++i) {
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




