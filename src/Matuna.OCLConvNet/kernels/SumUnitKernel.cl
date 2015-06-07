#include "RealType.h"

#ifndef INPUT_STRIDE
#define INPUT_STRIDE -1
#endif

#ifndef INPUT_WIDTH_OFFSET
#define INPUT_WIDTH_OFFSET -1
#endif

#ifndef WIDTH_LIMIT
#define WIDTH_LIMIT -1
#endif

#ifndef HEIGHT_LIMIT
#define HEIGHT_LIMIT -1
#endif

#ifndef INPUT_HEIGHT_OFFSET
#define INPUT_HEIGHT_OFFSET -1
#endif

#ifndef INPUT_UNIT_OFFSET
#define INPUT_UNIT_OFFSET -1
#endif

//Width * Height
#ifndef INPUT_UNIT_ELEMENT_COUNT_INC_PADDING 
#define INPUT_UNIT_ELEMENT_COUNT_INC_PADDING -1
#endif

#ifndef OUTPUT_OFFSET
#define OUTPUT_OFFSET -1
#endif

__kernel void SumUnitKernel(
#ifdef CONSTANT_INPUT
		__constant real_t* input,
#else
		__global const real_t* input,
#endif

		__global real_t* output
)
{
	const int index = get_global_id(0);
	real_t sum = 0;
	const int tempIndex = INPUT_UNIT_ELEMENT_COUNT_INC_PADDING * (index + INPUT_UNIT_OFFSET);
	int temp2;
	for (int i = INPUT_HEIGHT_OFFSET; i < HEIGHT_LIMIT; i++)
	{
		temp2 = INPUT_STRIDE * i + tempIndex;
		for (int j = INPUT_WIDTH_OFFSET; j < WIDTH_LIMIT; j++)
		{
			sum += input[j + temp2];
		}
	}

	output[index + OUTPUT_OFFSET] = sum;
}
