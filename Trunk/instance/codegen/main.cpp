#include <Windows.h>
#include <iostream>
#include <stdio.h>
#include "../fft.h"
#include "../fftw3.h"

#include <immintrin.h>


LARGE_INTEGER StartingTime, EndingTime, ElapsedMicroseconds;
LARGE_INTEGER Frequency;




int main(int argc, char * argv[])
{

	std::vector<std::complex<float>> u(N);
	std::vector<std::complex<float>> y(N);


	for (size_t i = 0; i < N; ++i) {
		u[i].real((float)(i + 1));
		u[i].imag((float)(i + 1));
	}

	fft_init();


	QueryPerformanceFrequency(&Frequency);
	QueryPerformanceCounter(&StartingTime);
	
	for (int i = 0; i < 100; ++i)
		fft(u, y);

#if(0)
		for (const auto& i : y)
			std::cout << i.real() << "  " << i.imag() << " i" << std::endl;
#endif

	QueryPerformanceCounter(&EndingTime);
	ElapsedMicroseconds.QuadPart = EndingTime.QuadPart - StartingTime.QuadPart;
	ElapsedMicroseconds.QuadPart *= 1000000;
	ElapsedMicroseconds.QuadPart /= Frequency.QuadPart;

	printf("%lld usec \n\n\n", ElapsedMicroseconds.QuadPart);




	fftwf_complex * fftIn = (fftwf_complex*)fftwf_malloc(sizeof(fftwf_complex) * N);
	fftwf_complex * fftOut = (fftwf_complex*)fftwf_malloc(sizeof(fftwf_complex) * N);
	
	float * cast = (float *)fftIn;
	for (size_t i = 0; i < 2*N; ++i) {
		cast[i] = (float)i;
	}

    fftwf_plan plan = fftwf_plan_dft_1d(N, fftIn, fftOut, FFTW_FORWARD,  FFTW_ESTIMATE);


	QueryPerformanceFrequency(&Frequency);
	QueryPerformanceCounter(&StartingTime);

	for (int i = 0; i < 100; ++i)
		fftwf_execute(plan);
	
	QueryPerformanceCounter(&EndingTime);
	ElapsedMicroseconds.QuadPart = EndingTime.QuadPart - StartingTime.QuadPart;
	ElapsedMicroseconds.QuadPart *= 1000000;
	ElapsedMicroseconds.QuadPart /= Frequency.QuadPart;

	printf("%lld usec \n\n\n", ElapsedMicroseconds.QuadPart);
	return 0;
}