/*
sampling 된 신호이므로 신호는 이산적일 것이다.
window를 적용한다는 것은 각 sample에 Hamming window식을 곱하는 것이다.
이 곱한 sample들에 대해 각각 DFT를 적용한다.
그럼 X[bin]=magnitude 이러한 값이 나온다.

sample의 개수가 500개 있을 때 overlap없이 10point windowing한 뒤 DFT를 한다고 하면 Bin은 총 50개 나올것이다.
*/
#include <stdio.h>
#include <math.h>


#define WINDOW_S 500	//window_size
#define PI 3.1415926535897	//double형에 맞게 그리고 오류가 없도록 13자리까지만함
#define	SAMPLE_FRE 441000	//sample frequency는 8kHz이다.


int main(void)
{
	double result[WINDOW_S];
	unsigned char buffer[WINDOW_S];
	double signal[WINDOW_S];	//cos과 sin의 반환값이 double이므로
	double max = -1.0;
	size_t read;
	int i, max_fre;


	FILE* sam_f = fopen("input.snd", "rb");

	if (!sam_f)
	{
		perror("파일 열기 실패");
		return 1;
	}

	read = fread(buffer, sizeof(signed char), WINDOW_S, sam_f);

	//dft과정 X[k], k=0,...,WINDOW_S
	//x[k]의 절대값은 루트(cos(2π*k*n/WINDOW_S)^2+sincos(2π*k*n/WINDOW_S)^2)이다.
	for (i = 0; i < WINDOW_S; i++)
	{
		double real = 0, imag = 0;//실수부, 허수부
		int j;
		for (j = 0; j < WINDOW_S; j++)	//sigma
		{
			//buffer는 unsigned로 8bit를 받았으므로 일단 signed char로 변환해주고
			//double형으로 변환하여 cos나 sin과 곱했을 때 값의 손실을 없앰
			real += (double)(signed char)buffer[j] * cos((2 * PI * i * j) / WINDOW_S);	//X[k]에서 x[n]*cos값의 합
			imag -= (double)(signed char)buffer[j] * sin((2 * PI * i * j) / WINDOW_S);	//X[k]에서 x[n]*sin값의 합
		}
		result[i] = sqrt(real * real + imag * imag);
		if (i < WINDOW_S / 2 && max < result[i])
		{
			max = result[i];
			max_fre = i * SAMPLE_FRE / WINDOW_S;
		}
	}

	//c언어여서 그래프는 그리지 못함 그래서 0이 아닌 값만 출력함
	printf("window size는 %d인데, 이때 유의미한 주파수와 그 크기는 다음과 같다.\n", WINDOW_S);
	for (i = 0; i < WINDOW_S / 2; i++)
	{
		if ((long long int)result[i] > 0)
		{
			printf("bin은 %d이고 그때의 frequency는 %dHz일 때의 magnitude는 %lf다\n",i, i * SAMPLE_FRE / WINDOW_S, result[i]);
		}
	}

	printf("최대 크기를 가지는 주파수값은 %dHz이며,그 크기는 %lf이다.", max_fre, max);

	fclose(sam_f);
	return 0;
}