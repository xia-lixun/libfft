#pragma once


///howto: build windows x86 dll from TDM-GCC
///       TDM-GCC build x64 exe/dll by default!
///
/// gcc -c -DBUILD_DLL fft.c -o fft.o -O3 -march=corei7-avx -std=c11 -DWIN32 -m32
/// gcc -shared -o libfft.dll fft.o -Wl,--output-def,libfft.def,--out-implib,libfft.a -m32
///
/// then use lib tool of visual studio to generate the import lib
/// lib /machine:x86 /def:libfft.def



///export functions for c
#ifdef __cplusplus
extern "C" {
#endif


//#define BUILD_DLL 1


//#ifdef BUILD_DLL
//#define LIBFFT_DLL __declspec(dllexport)
//#else
//#define LIBFFT_DLL __declspec(dllimport)
//#endif



typedef struct plan {

	int direct;
	size_t n;  //fft size

	float * x;  //input buffer, complex number interleaved format
	float * y;  //output buffer, complex number interleaved format

	float* ze;  //even part for real-optimized fft
	float* zo;  //odd part for real-optimized fft

				//n_4 counts from 8/4 to N/4, which is 2..N/4
	float ** wre_table;
	float ** wim_table;

	float* wre_ropt;
	float* wim_ropt;
} plan_t;









//LIBFFT_DLL void fft_init(plan_t *p, size_t n, int direct);
//LIBFFT_DLL void fft(plan_t * p);
//LIBFFT_DLL void ropt(plan_t* p, float * output);
//LIBFFT_DLL void fft_clean(plan_t * p);
void fft_init(plan_t* p, size_t n, int direct);
void fft(plan_t* p);
void fft_clean(plan_t* p);


#ifdef __cplusplus
}
#endif



///export for c++
///export for class
