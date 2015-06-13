#include "RealType.h"

//<!@
#define INPUT_STRIDE -1
#define INPUT_WIDTH_OFFSET -1
#define WIDTH_LIMIT -1
#define HEIGHT_LIMIT -1
#define INPUT_HEIGHT_OFFSET -1
#define INPUT_UNIT_OFFSET -1
#define INPUT_UNIT_ELEMENT_COUNT_INC_PADDING -1
#define OUTPUT_OFFSET -1
//#define CONSTANT_INPUT
//!@>

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
