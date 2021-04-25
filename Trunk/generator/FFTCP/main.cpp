// My FFT offers ~ 75% - 90% MIPS of FFTW in single precision float
// So I have almost the same performance without link to any third-
// party libs! Tested under both Visual Studio 2015(V140) and GCC 
// 4.8.4
// Lixun Xia, 2016.
//
// Linux: install libfftw3-dev
//        gcc fft.c -o ./Release/fft-linux -march=native -O3 -std=gnu99 -lrt -lfftw3f -lm
// TDM-GCC: 
//        g++ fft.c macros-sse.h -o tdmfft.exe -O3 -std=c++11 -march=native -DWIN32
// Turn on AVX_ENABLE flag in macros-sse.h for avx support.
//
//
#include <Windows.h>
#include <map>
#include <cmath>
#include <vector>
#include <complex>
#include <iostream>
#include <iomanip>
#include "macros-sse.h"
#include "cpuid.h"





#define INVERSE_FFT
#define N 8192






#define PI 3.1415926535897932384626433832795028841971693993751058209  /* pi */

#ifdef INVERSE_FFT
int direction = 1;
#else
int direction = -1;
#endif



std::complex<float> complex_i(0.0f, 1.0f);
std::vector<std::complex<float>> Wnk(N / 4);
std::vector<std::complex<float>> temp(N);

std::map<std::complex<float> *, std::string> lut_re;
std::map<std::complex<float> *, std::string> lut_im;



void fft_init()
{
	for (size_t k = 0; k < N / 4; ++k) {
		std::complex<double> phi(0, -2.0 * PI * (double)k / (double)N);
		auto temp = exp(phi);
		Wnk[k].real((float)temp.real());
		Wnk[k].imag((float)temp.imag());
	}
}


static bool is_mid_child = true;
static int _cnt4 = 0;
static int _ncat = 0;

std::vector<std::complex<float>>
fft(
	bool leaf,
	std::vector<int>& leaf_categ,
	std::vector<std::string>& leaf_param,
	cpu_features_t& _cpuid,
	std::vector<std::complex<float> *>& px,
	std::vector<std::vector<std::complex<float>*>>& input_tree,
	size_t trace,
	size_t input_offset,
	std::vector<std::complex<float>>& output,
	std::map<std::complex<float> *, std::string>& lut_r,
	std::map<std::complex<float> *, std::string>& lut_i)
{
	size_t n = px.size();
	std::vector<std::complex<float>> y(n);

#if(0)
	//node trace the input tree
	if (!input_tree[trace].empty()) {
		std::cout << "\n";
		//input offset info
		std::cout << input_offset << " ";
		std::cout << input_offset + n / 2 << " ";
		std::cout << input_offset + n / 2 + n / 4 << " ";
		std::cout << "| ";
		for (const auto& nt : input_tree[trace])
			std::cout << nt << " ";
		std::cout << "\n";
	}
#endif


	//data crunching begins
	if (n == 1) {

		output[input_offset] = *input_tree[trace][0];
		//std::cout
		//	<< "output[re(" + std::to_string(input_offset) + ")] = " + lut_r[input_tree[trace][0]] + ";"
		//	<< "output[im(" + std::to_string(input_offset) + ")] = " + lut_i[input_tree[trace][0]] + ";"
		//	<< std::endl;

		//if (print_node == n) {
		if(leaf) {
			if (is_mid_child) {
				is_mid_child = false;
				//std::cout << "fft_1(p->y + re(" + std::to_string(input_offset) + "), " + lut_r[input_tree[trace][0]] + ",";
				//std::cout << "db n1: " << lut_r[input_tree[trace][0]] << " ";
				
				leaf_param.push_back(lut_r[input_tree[trace][0]]);
			}
			else {
				is_mid_child = true;
				//std::cout << lut_r[input_tree[trace][0]] + ");" << std::endl;
				//std::cout << lut_r[input_tree[trace][0]] << std::endl;

				leaf_param.push_back(lut_r[input_tree[trace][0]]);
				_ncat += 1; _cnt4 += 1; if (_cnt4 == 4) { _cnt4 = 0; leaf_categ.push_back(_ncat); _ncat = 0; }
			}
		}

		y[0] = *px[0];
		return y;
	}

	else if (n == 2) {

		output[input_offset] = *input_tree[trace][0] + *input_tree[trace][1];
		output[input_offset + 1] = *input_tree[trace][0] - *input_tree[trace][1];
		//std::cout 
		//	<< "output[" + std::to_string(input_offset) + "] = " + lut[input_tree[trace][0]] + " + "+ lut[input_tree[trace][1]] + ";" 
		//	<< std::endl;
		//std::cout 
		//	<< "output[" + std::to_string(input_offset + 1) + "] = " + lut[input_tree[trace][0]] + " - "+ lut[input_tree[trace][1]] + ";" 
		//	<< std::endl;

		//if (print_node == n) {
		if(leaf) {
			//std::cout << "fft_2(" + lut_r[input_tree[trace][0]] + ", " + lut_r[input_tree[trace][1]] + ", p->y + re(" + std::to_string(input_offset) + "));" << std::endl;
			//std::cout << "db n2: " << lut_r[input_tree[trace][0]] << " " << lut_r[input_tree[trace][1]] << std::endl;

			leaf_param.push_back(lut_r[input_tree[trace][0]]);
			leaf_param.push_back(lut_r[input_tree[trace][1]]);
			_ncat += 2; _cnt4 += 1; if (_cnt4 == 4) { _cnt4 = 0; leaf_categ.push_back(_ncat); _ncat = 0; }
		}

		y[0] = *px[0] + *px[1];
		y[1] = *px[0] - *px[1];
		return y;
	}

	else {

		std::vector<std::complex<float> *> pxu(n / 2);
		std::vector<std::complex<float> *> pxz(n / 4);
		std::vector<std::complex<float> *> pxz_(n / 4);

		for (size_t j = 0; j < n / 2; ++j)
			pxu[j] = px[j * 2];
		for (size_t j = 0; j < n / 4; ++j)
			pxz[j] = px[j * 4 + 1];
		for (size_t j = 1; j < n / 4; ++j)
			pxz_[j] = px[j * 4 - 1];
		pxz_[0] = px[n - 1];

		std::vector<std::complex<float>> u = fft(leaf, leaf_categ, leaf_param, _cpuid, pxu, input_tree, trace * 3 + 1, input_offset, output, lut_r, lut_i);
		std::vector<std::complex<float>> z = fft(leaf, leaf_categ, leaf_param, _cpuid, pxz, input_tree, trace * 3 + 2, input_offset + n / 2, output, lut_r, lut_i);
		std::vector<std::complex<float>> z_ = fft(leaf, leaf_categ, leaf_param, _cpuid, pxz_, input_tree, trace * 3 + 3, input_offset + n / 2 + n / 4, output, lut_r, lut_i);

		size_t stride = N / n;
		std::complex<float> wnk, wnk_;

		for (size_t k = 0; k < n / 4; ++k) {

			//std::complex<float> phi(0, -2.0f * (float)M_PI * (float)k / (float)n);
			//auto Wnk = exp(phi);
			//auto Wnk_ = exp(-phi);
			//std::complex<float> i(0.0f, 1.0f);
			wnk = Wnk[k * stride];
			wnk_ = std::conj(wnk);

			temp[k] = output[input_offset + k] + (wnk * output[input_offset + n / 2 + k] + wnk_ * output[input_offset + n / 2 + n / 4 + k]);
			temp[k + n / 2] = output[input_offset + k] - (wnk * output[input_offset + n / 2 + k] + wnk_ * output[input_offset + n / 2 + n / 4 + k]);

			temp[k + n / 4] = output[input_offset + n / 4 + k] - complex_i * (wnk * output[input_offset + n / 2 + k] - wnk_ * output[input_offset + n / 2 + n / 4 + k]);
			temp[k + n / 4 + n / 2] = output[input_offset + n / 4 + k] + complex_i * (wnk * output[input_offset + n / 2 + k] - wnk_ * output[input_offset + n / 2 + n / 4 + k]);

			output[input_offset + k] = temp[k];
			output[input_offset + n / 2 + k] = temp[k + n / 2];

			output[input_offset + n / 4 + k] = temp[k + n / 4];
			output[input_offset + n / 4 + n / 2 + k] = temp[k + n / 4 + n / 2];
		}




		if (n == 4) {
			//if (print_node >= 4) {
#ifdef INVERSE_FFT
				//std::cout << "ifft_4(p->y + re(" + std::to_string(input_offset) + "), p->y + re(" + std::to_string(input_offset) + "));" << std::endl;
#else
				//std::cout << "fft_4(p->y + re(" + std::to_string(input_offset) + "), p->y + re(" + std::to_string(input_offset) + "));" << std::endl;
#endif
			//}
		}


		else if (n == 8) {
			//if (print_node >= 4) {
#ifdef INVERSE_FFT
				//std::cout
				//	<< "ifft_8(p->y + re("
				//	+ std::to_string(input_offset)
				//	+ "), p->y + re(" + std::to_string(input_offset + n / 4)
				//	+ "), p->y + re(" + std::to_string(input_offset + n / 2)
				//	+ "), p->y + re(" + std::to_string(input_offset + n / 2 + n / 4)
				//	+ "), p->wre_table[0], p->wim_table[0]);"
				//	<< std::endl;
#else
				//std::cout
				//	<< "fft_8(p->y + re("
				//	+ std::to_string(input_offset)
				//	+ "), p->y + re(" + std::to_string(input_offset + n / 4)
				//	+ "), p->y + re(" + std::to_string(input_offset + n / 2)
				//	+ "), p->y + re(" + std::to_string(input_offset + n / 2 + n / 4)
				//	+ "), p->wre_table[0], p->wim_table[0]);"
				//	<< std::endl;
#endif
			//}
		}


		else {
			//if (print_node >= 4) {
			if(!leaf) {

				unsigned long index;
				_BitScanForward(&index, (unsigned long)n / 4);
				index -= 1;

				if (_cpuid.AVX) {
#ifdef INVERSE_FFT
					std::cout
						<< "ifft_n_("
						+ std::to_string(n / 4)
						+ ", p->y + re(" + std::to_string(input_offset)
						+ "), p->y + re(" + std::to_string(input_offset + n / 4)
						+ "), p->y + re(" + std::to_string(input_offset + n / 2)
						+ "), p->y + re(" + std::to_string(input_offset + n / 2 + n / 4)
						+ "), p->wre_table[" + std::to_string(index) + "], p->wim_table[" + std::to_string(index) + "]);"
						<< std::endl;
#else
					std::cout
						<< "fft_n_("
						+ std::to_string(n / 4)
						+ ", p->y + re(" + std::to_string(input_offset)
						+ "), p->y + re(" + std::to_string(input_offset + n / 4)
						+ "), p->y + re(" + std::to_string(input_offset + n / 2)
						+ "), p->y + re(" + std::to_string(input_offset + n / 2 + n / 4)
						+ "), p->wre_table[" + std::to_string(index) + "], p->wim_table[" + std::to_string(index) + "]);"
						<< std::endl;
#endif
				}
				else { //SSE only
#ifdef INVERSE_FFT
					std::cout
						<< "ifft_n("
						+ std::to_string(n / 4)
						+ ", p->y + re(" + std::to_string(input_offset)
						+ "), p->y + re(" + std::to_string(input_offset + n / 4)
						+ "), p->y + re(" + std::to_string(input_offset + n / 2)
						+ "), p->y + re(" + std::to_string(input_offset + n / 2 + n / 4)
						+ "), p->wre_table[" + std::to_string(index) + "], p->wim_table[" + std::to_string(index) + "]);"
						<< std::endl;
#else
					   std::cout
					   	<< "fft_n("
					   + std::to_string(n / 4)
					   + ", p->y + re(" + std::to_string(input_offset)
					   + "), p->y + re(" + std::to_string(input_offset + n / 4)
					   + "), p->y + re(" + std::to_string(input_offset + n / 2)
					   + "), p->y + re(" + std::to_string(input_offset + n / 2 + n / 4)
					   + "), p->wre_table[" + std::to_string(index) + "], p->wim_table[" + std::to_string(index) + "]);"
					   << std::endl;
#endif
				}
			}
		}



		for (size_t k = 0; k < n / 4; ++k) {

			std::complex<float> phi(0, -2.0f * (float)PI * (float)k / (float)n);
			auto wnk = exp(phi);
			auto wnk_ = exp(-phi);
			std::complex<float> i(0.0f, 1.0f);

			y[k] = u[k] + (wnk * z[k] + wnk_ * z_[k]);
			y[k + n / 2] = u[k] - (wnk * z[k] + wnk_ * z_[k]);
			y[k + n / 4] = u[k + n / 4] - i * (wnk * z[k] - wnk_ * z_[k]);
			y[k + 3 * n / 4] = u[k + n / 4] + i * (wnk * z[k] - wnk_ * z_[k]);
		}
		return y;
	}
}









//parent must be prepared
//tree must be grow in sequence
void
generate_child(
	std::vector<std::vector<std::complex<float>*>>& heap,
	size_t parent_node)
{
	size_t parent_n = heap[parent_node].size();

	//children, only parent larger than or equal to 4 has living children
	std::vector<std::complex<float>*> child_l;
	std::vector<std::complex<float>*> child_m;
	std::vector<std::complex<float>*> child_r;

	if (parent_n >= 4) {
		for (size_t j = 0; j < parent_n / 2; ++j)
			child_l.push_back(heap[parent_node][j * 2]);
		for (size_t j = 0; j < parent_n / 4; ++j)
			child_m.push_back(heap[parent_node][j * 4 + 1]);

		child_r.push_back(heap[parent_node][parent_n - 1]);
		for (size_t j = 1; j < parent_n / 4; ++j)
			child_r.push_back(heap[parent_node][j * 4 - 1]);
	}

	//otherwise insert zombie children
	heap.push_back(child_l);
	heap.push_back(child_m);
	heap.push_back(child_r);
}




//binary heap that contains the input array and sub-ffts
std::vector<std::vector<std::complex<float>*>>
generate_tree(std::vector<std::complex<float>*>& px)
{
	size_t logn = (size_t)log2(px.size());
	std::vector<std::vector<std::complex<float>*>> tree;
	size_t node_n = tree.size();

	tree.push_back(px);
	size_t delta_node_n = tree.size() - node_n;
	node_n = tree.size();

	size_t parent_node = 0;
	for (size_t i = 0; i < logn - 1; ++i) {

		for (size_t j = 0; j < delta_node_n; ++j) {
			generate_child(tree, parent_node + j);
		}
		parent_node += delta_node_n;
		delta_node_n = tree.size() - node_n;
		node_n = tree.size();

	}
	return tree;
}






















int main(int argc, char * argv[])
{
	//find out the cpu features
	cpu_features_t _id;
	fetch_cpu_features(&_id);


	//        -i 2Pi k/N
	//       e              k=[0..N/2-1]
	//int K = 4;
	//std::complex<float> phi(0, -2.0f * (float)M_PI * (float)K / (float)N);
	//auto a = exp(phi);
	//std::cout << a.real() << " + " << a.imag() << std::endl;



	// n is the length of the array intput x
	// px is the array of the points of the input x

	std::complex<float> x[N];
	std::vector<std::complex<float> *> ppx(N);

	for (int i = 0; i < N; ++i) {
		x[i].real((float)(i + 1));
		x[i].imag((float)(i + 1));
		ppx[i] = &x[i];
		//std::cout << x[i].real() << " + " << x[i].imag() << " i" << std::endl;

		lut_re[ppx[i]] = "p->x + re(" + std::to_string(i) + ")";
		lut_im[ppx[i]] = "p->x + im(" + std::to_string(i) + ")";
	}


#if(0)
	for (const auto& t : ppx)
		std::cout << t << " ";
	std::cout << "\n\n" << std::endl;
#endif

	std::vector<std::vector<std::complex<float>*>> z = generate_tree(ppx);

#if(0)
	for (const auto& t : z) {
		std::cout << "level: ";
		for (const auto& tt : t)
			std::cout << tt << " ";
		std::cout << "\n";
	}
	std::cout << "-------------" << std::endl;
#endif

	std::cout
		<< "#include <time.h>\n"
		<< "#include <math.h>\n"
		<< "#include <stdio.h>\n"
		<< "#include <stdlib.h>\n"
		<< "#include <immintrin.h>\n"
		<< "#include \"fftw3.h\"\n"
		<< "#include \"macros-sse.h\"\n\n"
		<< "#ifdef WIN32\n"
		<< "#include <Windows.h>\n"
		<< "#else\n"
		<< "#include <malloc.h>\n"
		<< "#endif\n"
		<< "\n\n\n\n"
		<< std::endl;




	std::cout
		<< "void fft_init(plan_t * p, size_t n, int direct)\n"
		<< "{\n"
		<< "	p->n = n;\n"

		<< "    float * W_REAL = (float *) malloc(n / 4 * 2 * sizeof(float));\n"
		<< "    float * W_IMAG = (float *) malloc(n / 4 * 2 * sizeof(float));\n\n"
		<< "\n"
		<< "    //W_REAL[0] = 1.0f;  W_REAL[1] = 1.0f;\n"
		<< "    //W_IMAG[0] = 0.0f;  W_IMAG[1] = -0.0f;\n"
		<< "\n"
		<< "    for (size_t k = 0; k < n / 4; ++k) {\n"
		<< "        W_REAL[2*k+0] = -(float)tan((double)(direct * 2) * PI * (double)k / (double)n);\n"
		<< "        W_REAL[2*k+1] = -W_REAL[2*k+0];\n"
		<< "        W_IMAG[2*k+0] = (float)cos((double)(direct * 2) * PI * (double)k / (double)n);\n"
		<< "        W_IMAG[2*k+1] = W_IMAG[2*k+0];\n"
		<< "    }\n\n"
		<< "    p->wre_table = (float **)malloc((int)log2(n / 4) * sizeof(float *));\n"
		<< "    p->wim_table = (float **)malloc((int)log2(n / 4) * sizeof(float *));\n"
		<< "\n"
		<< "   for (size_t n_4 = 2; n_4 < n / 2; n_4 = n_4 * 2) {\n"
		<< "\n"
		<< "       //stride is constant for fixed n_4 value\n"
		<< "       size_t stride = (n / 4) / n_4;\n"
		<< "       size_t index = (int)log2((double)n_4) - 1;\n"
		<< "\n"
		<< "#ifdef WIN32\n"
		<< "       p->wre_table[index] = (float *)_mm_malloc(n_4 / 2 * 4 * sizeof(float), 32);\n"
		<< "       p->wim_table[index] = (float *)_mm_malloc(n_4 / 2 * 4 * sizeof(float), 32);\n"
		<< "#else\n"
		<< "		p->wre_table[index] = (float *)memalign(32, n_4 / 2 * 4 * sizeof(float));\n"
		<< "		p->wim_table[index] = (float *)memalign(32, n_4 / 2 * 4 * sizeof(float));\n"
		<< "#endif\n"
		<< "\n"
		<< "       float * pre = p->wre_table[index];\n"
		<< "       float * pim = p->wim_table[index];\n"
		<< "\n"
		<< "       size_t i = 0;\n"
		<< "       size_t j = 0;\n"
		<< "       for (size_t k = 0; k < n_4; k += 2) {\n"
		<< "\n"
		<< "	        pre[i++] = W_REAL[k*stride * 2];\n"
		<< "	        pre[i++] = W_REAL[k*stride * 2 + 1];\n"
		<< "	        pre[i++] = W_REAL[(k + 1)*stride * 2];\n"
		<< "	        pre[i++] = W_REAL[(k + 1)*stride * 2 + 1];\n"
		<< "\n"
		<< "	        pim[j++] = W_IMAG[k*stride * 2];\n"
		<< "	        pim[j++] = W_IMAG[k*stride * 2 + 1];\n"
		<< "	        pim[j++] = W_IMAG[(k + 1)*stride * 2];\n"
		<< "	        pim[j++] = W_IMAG[(k + 1)*stride * 2 + 1];\n"
		<< "        }\n"
		<< "    }\n\n"
		<< "    if (direct == -1) {\n"
		<< "		// code below prepares constant for fft_8() and ifft_8() conversions. wre is actual tan() and wim is actual cos()\n"
		<< "		p->wre_table[0][0] = -0.0f; p->wre_table[0][1] = 0.0f; p->wre_table[0][2] = 1.0f; p->wre_table[0][3] = -1.0f; \n"
		<< "		p->wim_table[0][0] = 1.0f; p->wim_table[0][1] = 1.0f; p->wim_table[0][2] = (float)(sqrt(2.0)*0.5); p->wim_table[0][3] = (float)(sqrt(2.0)*0.5); \n"
		<< "    } else {\n"
		<< "        p->wre_table[0][0] = 0.0f; p->wre_table[0][1] = -0.0f; p->wre_table[0][2] = -1.0f; p->wre_table[0][3] = 1.0f;\n"
	    << "        p->wim_table[0][0] = 1.0f; p->wim_table[0][1] = 1.0f; p->wim_table[0][2] = (float)(sqrt(2.0)*0.5); p->wim_table[0][3] = (float)(sqrt(2.0)*0.5);\n"
		<< "    }\n\n"
		<< "    free(W_REAL);\n"
		<< "    free(W_IMAG);\n\n"
		<< "#ifdef WIN32\n"
		<< "	p->x = (float *)_mm_malloc(n * 2 * sizeof(float), 32);\n"
		<< "	p->y = (float *)_mm_malloc(n * 2 * sizeof(float), 32);\n"
		<< "#else\n"
		<< "	p->x = (float *)memalign(32, n * 2 * sizeof(float));\n"
		<< "	p->y = (float *)memalign(32, n * 2 * sizeof(float));\n"
		<< "#endif\n\n"
		<< "	_MM_SET_FLUSH_ZERO_MODE(_MM_FLUSH_ZERO_ON);\n"
		<< "    _MM_SET_DENORMALS_ZERO_MODE(_MM_DENORMALS_ZERO_ON);\n\n"
		<< "}\n"
		<< "\n\n\n\n"
		<< std::endl;



	std::cout
		<< "void fft_clean(plan_t * p)\n"
		<< "{\n"
		<< "   for (size_t n_4 = 2; n_4 < p->n / 2; n_4 = n_4 * 2) {\n"
		<< "       size_t index = (int)log2((double)n_4) - 1;\n"
		<< "#ifdef WIN32\n"
		<< "       _mm_free(p->wre_table[index]);\n"
		<< "       _mm_free(p->wim_table[index]);\n"
		<< "#else\n"
		<< "       free(p->wre_table[index]);\n"
		<< "       free(p->wim_table[index]);\n"
		<< "#endif\n"
		<< "   }\n"
		<< "   free(p->wre_table);\n"
		<< "   free(p->wim_table);\n\n"
		<< "#ifdef WIN32\n"
		<< "	_mm_free(p->x);\n"
		<< "	_mm_free(p->y);\n"
		<< "#else\n"
		<< "	free(p->x);\n"
		<< "	free(p->y);\n"
		<< "#endif\n"
		<< "}\n\n\n\n"
		<< std::endl;







	std::cout
		<< "void fft(plan_t * p)\n"
		<< "{\n"
		<< std::endl;

	fft_init();
	std::vector<std::complex<float>> yy(N);

	//explore the leaf cases
	std::vector<int> leaf_category_code;
	std::vector<std::string> leaf_parameter;
	fft(true, leaf_category_code, leaf_parameter, _id, ppx, z, 0, 0, yy, lut_re, lut_im);

	//iterate all leaf category codes for base cases
	size_t offset = 0;
	for (const auto& lcc : leaf_category_code) {
		if (lcc == 7)
#ifdef INVERSE_FFT
			std::cout << "ifft_8_1_leaf("
#else
			std::cout << "fft_8_1_leaf(" 
#endif
			+ leaf_parameter[offset + 0] + ", "
			+ leaf_parameter[offset + 1] + ", "
			+ leaf_parameter[offset + 2] + ", "
			+ leaf_parameter[offset + 3] + ", "
			+ leaf_parameter[offset + 4] + ", "
			+ leaf_parameter[offset + 5] + ", "
			+ leaf_parameter[offset + 6] + ", "
			+ leaf_parameter[offset + 7] + ", "
			"p->y + re(" + std::to_string(offset) + "));" << std::endl;
		else
#ifdef INVERSE_FFT
			std::cout << "ifft_4_2_leaf("
#else
			std::cout << "fft_4_2_leaf("
#endif
			+ leaf_parameter[offset + 0] + ", "
			+ leaf_parameter[offset + 1] + ", "
			+ leaf_parameter[offset + 2] + ", "
			+ leaf_parameter[offset + 3] + ", "
			+ leaf_parameter[offset + 4] + ", "
			+ leaf_parameter[offset + 5] + ", "
			+ leaf_parameter[offset + 6] + ", "
			+ leaf_parameter[offset + 7] + ", "
			"p->y + re(" + std::to_string(offset) + "));" << std::endl;
		offset += 8;
	}

	//generate non-base cases
	fft(false, leaf_category_code, leaf_parameter, _id, ppx, z, 0, 0, yy, lut_re, lut_im);

	std::cout
		<< "}\n\n\n\n"
		<< std::endl;




#ifdef INVERSE_FFT
	std::string fftw_flag("FFTW_BACKWARD");
#else
	std::string fftw_flag("FFTW_FORWARD");
#endif

	std::cout
		<< "#ifdef WIN32\n"
		<< "LARGE_INTEGER StartingTime, EndingTime, ElapsedMicroseconds;\n"
		<< "LARGE_INTEGER Frequency;\n\n"
		<< "//__declspec(align(32)) float x[N*2];\n"
		<< "//__declspec(align(32)) float y[N*2];\n"
		<< "#else\n"
		<< "struct timespec t0, t1;\n\n"
		<< "//float __attribute__((aligned(32))) x[N * 2];\n"
		<< "//float __attribute__((aligned(32))) y[N * 2];\n"
		<< "#endif\n\n"
		<< "\n\n\n\n"
		<< "int main(int argc, char * argv[])\n"
		<< "{\n"
		<< "	plan_t p;\n"
		<< "	fft_init(&p, " + std::to_string(N) + ", " + std::to_string(direction) + ");\n\n"
		<< "    srand((unsigned int)time(0));\n"
		<< "    for (size_t i = 0; i < p.n; i++) {\n"
		<< "        p.x[re(i)] = (float)((double)rand() / (double)RAND_MAX);\n"
		<< "        p.x[im(i)] = (float)((double)rand() / (double)RAND_MAX);\n"
		<< "    }\n"
		<< "    //FILE *fid = fopen(\"./input1024.dat\", \"w\");\n"
		<< "    //for (int i = 0; i < N; i++)\n"
		<< "    //    fprintf(fid, \"%3.16f, %3.16f\\n\", x[re(i)], x[im(i)]);\n"
		<< "    //fclose(fid);\n"
		<< "\n\n\n\n\n\n\n\n\n"
		<< "\n\n\n\n\n\n\n\n\n"
		<< "	fftwf_complex * fftIn = (fftwf_complex*)fftwf_malloc(sizeof(fftwf_complex) * " + std::to_string(N)  + ");\n"
		<< "    fftwf_complex * fftOut = (fftwf_complex*)fftwf_malloc(sizeof(fftwf_complex) * " + std::to_string(N) + ");\n"
		<< "\n"
		<< "    float * cast = (float *)fftIn;\n"
		<< "	for (size_t i = 0; i < " + std::to_string(N) + "; ++i) {\n"
		<< "	    cast[re(i)] = p.x[re(i)];\n"
		<< "	    cast[im(i)] = p.x[im(i)];\n"
		<< "    }\n"
		<< "\n"
		<< "    fftwf_plan plan = fftwf_plan_dft_1d(" + std::to_string(N) + ", fftIn, fftOut, " + fftw_flag + ", FFTW_ESTIMATE);\n"
		<< "\n"
		<< "\n"
		<< "#ifdef WIN32\n"
		<< "	QueryPerformanceFrequency(&Frequency);\n"
		<< "    QueryPerformanceCounter(&StartingTime);\n\n"
		<< "#else\n"
		<< "	clock_gettime(CLOCK_MONOTONIC, &t0);\n"
		<< "#endif\n"
		<< "\n"
		<< "    for (int i = 0; i < 10000; ++i)\n"
		<< "        fftwf_execute(plan);\n"
		<< "\n"
		<< "#ifdef WIN32\n"
		<< "	QueryPerformanceCounter(&EndingTime);\n"
		<< "    ElapsedMicroseconds.QuadPart = EndingTime.QuadPart - StartingTime.QuadPart;\n"
		<< "    ElapsedMicroseconds.QuadPart *= 1000000;\n"
		<< "    ElapsedMicroseconds.QuadPart /= Frequency.QuadPart;\n"
		<< "    printf(\"fftw : %lld usec \\n\", ElapsedMicroseconds.QuadPart);\n\n\n"
		<< "#else\n"
		<< "	clock_gettime(CLOCK_MONOTONIC, &t1);\n"
		<< "	{struct timespec temp;\n"
		<< "	if ((t1.tv_nsec - t0.tv_nsec)<0) {\n"
		<< "		temp.tv_sec = t1.tv_sec - t0.tv_sec - 1;\n"
		<< "		temp.tv_nsec = 1000000000 + t1.tv_nsec - t0.tv_nsec;\n"
		<< "	}\n"
		<< "	else {\n"
		<< "		temp.tv_sec = t1.tv_sec - t0.tv_sec;\n"
		<< "		temp.tv_nsec = t1.tv_nsec - t0.tv_nsec;\n"
		<< "	}\n"
		<< "	printf(\"fftw : %ld s, %ld ns\\n\", temp.tv_sec, temp.tv_nsec);}\n"
		<< "#endif\n"
		<< "    fftwf_destroy_plan(plan);\n"
		<< "\n\n\n\n\n\n\n\n\n"
		<< "\n\n\n\n\n\n\n\n\n"
		<< std::endl;


	std::cout
		<< "#ifdef WIN32\n"
		<< "	QueryPerformanceFrequency(&Frequency);\n"
		<< "    QueryPerformanceCounter(&StartingTime);\n\n"
		<< "#else\n"
		<< "	clock_gettime(CLOCK_MONOTONIC, &t0);\n"
		<< "#endif\n"
		<< "    for (int i = 0; i < 10000; ++i)"
		<< std::endl;

	std::cout
		<< "        fft(&p);\n\n"
		<< std::endl;

	std::cout
		<< "#ifdef WIN32\n"
		<< "	QueryPerformanceCounter(&EndingTime);\n"
		<< "    ElapsedMicroseconds.QuadPart = EndingTime.QuadPart - StartingTime.QuadPart;\n"
		<< "    ElapsedMicroseconds.QuadPart *= 1000000;\n"
		<< "    ElapsedMicroseconds.QuadPart /= Frequency.QuadPart;\n"
		<< "    printf(\"fft : %lld usec \\n\", ElapsedMicroseconds.QuadPart);\n\n\n"
		<< "#else\n"
		<< "	clock_gettime(CLOCK_MONOTONIC, &t1);\n"
		<< "	{struct timespec temp;\n"
		<< "	if ((t1.tv_nsec - t0.tv_nsec)<0) {\n"
		<< "		temp.tv_sec = t1.tv_sec - t0.tv_sec - 1;\n"
		<< "		temp.tv_nsec = 1000000000 + t1.tv_nsec - t0.tv_nsec;\n"
		<< "	}\n"
		<< "	else {\n"
		<< "		temp.tv_sec = t1.tv_sec - t0.tv_sec;\n"
		<< "		temp.tv_nsec = t1.tv_nsec - t0.tv_nsec;\n"
		<< "	}\n"
		<< "	printf(\"fft : %ld s, %ld ns\\n\", temp.tv_sec, temp.tv_nsec);}\n"
		<< "#endif\n"
		<< std::endl;

		std::cout
		<< "    fft_clean(&p);\n"
		<< std::endl;

	std::cout
		<< "\n\n\n\n\n\n\n\n\n"
		<< "\n\n\n\n\n\n\n\n\n"
		<< "    size_t maxreidx, maximidx;\n"
		<< "    double err_max_re = 0.0f;\n"
		<< "    double err_max_im = 0.0f;\n"
		<< "	cast = (float *)fftOut;\n"
		<< "	for (int i = 0; i < " + std::to_string(N) + "; ++i) {\n"
		<< "	    if ( fabs(p.y[re(i)] - cast[re(i)]) / fabs(cast[re(i)]) > err_max_re ) {\n"
		<< "            err_max_re = fabs(p.y[re(i)] - cast[re(i)]) / fabs(cast[re(i)]);\n"
		<< "            maxreidx = i; }\n"
		<< "        if ( fabs(p.y[im(i)] - cast[im(i)]) / fabs(cast[im(i)]) > err_max_im ) {\n"
		<< "            err_max_im = fabs(p.y[im(i)] - cast[im(i)]) / fabs(cast[im(i)]);\n"
		<< "            maximidx = i; }\n"
		<< "    }\n"
		<< "    printf(\"max error (real, imag) : (%3.16f, %3.16f)\\n\", err_max_re, err_max_im);\n"
		<< "    printf(\"real fft vs fftw: (%3.16f, %3.16f * I) (%3.16f, %3.16f * I)\\n\", p.y[re(maxreidx)], p.y[im(maxreidx)], cast[re(maxreidx)], cast[im(maxreidx)]);\n"
		<< "    printf(\"imag fft vs fftw: (%3.16f, %3.16f * I) (%3.16f, %3.16f * I)\\n\", p.y[re(maximidx)], p.y[im(maximidx)], cast[re(maximidx)], cast[im(maximidx)]);\n"
		<< "    return 0;\n"
		<< "}\n\n\n\n"
		<< std::endl;
	return 0;


}


