#include "RealType.h"

//<!@
#define INPUT_DELTA_UNIT_MEMORY_WIDTH -1
#define OUTPUT_UNIT_MEMORY_WIDTH -1
#define INPUT_UNIT_MEMORY_WIDTH -1
#define INPUT_UNIT_MEMORY_WIDTH_OFFSET -1
#define INPUT_UNIT_MEMORY_HEIGHT_OFFSET -1
#define INPUT_DELTA_UNIT_MEMORY_WIDTH_OFFSET -1
#define INPUT_DELTA_UNIT_MEMORY_HEIGHT_OFFSET -1
#define INPUT_DELTA_UNIT_OFFSET -1
#define WIDTH_LIMIT -1
#define HEIGHT_LIMIT -1
#define OUTPUT_UNIT_MEMORY_ELEMENTS -1
#define INPUT_DELTA_UNIT_MEMORY_ELEMENTS -1
#define OUTPUT_UNIT_MEMORY_WIDTH_OFFSET -1
#define OUTPUT_UNIT_MEMORY_HEIGHT_OFFSET -1
#define OUTPUT_UNIT_OFFSET -1
//#define CONSTANT_INPUT
//#define CONSTANT_INPUT_DELTA
//!@>

__kernel void MultiplyWithOffsetKernel(
#ifdef CONSTANT_INPUT
		__constant real_t* input,
#else
		__global const real_t* input,
#endif
#ifdef CONSTANT_INPUT_DELTA
		__constant real_t* inputDelta,
#else
		__global const real_t* inputDelta,
#endif

		__global real_t* output
)
{

	const int xIndex = get_global_id(0);
	const int yIndex = get_global_id(1);
	const int zIndex = get_global_id(2);

	const int outputIndex = xIndex + OUTPUT_UNIT_MEMORY_WIDTH_OFFSET + (yIndex + OUTPUT_UNIT_MEMORY_HEIGHT_OFFSET) * OUTPUT_UNIT_MEMORY_WIDTH + (zIndex + OUTPUT_UNIT_OFFSET) * OUTPUT_UNIT_MEMORY_ELEMENTS;

	real_t sum = 0;

	const int tempDeltaIndex = INPUT_DELTA_UNIT_MEMORY_ELEMENTS * (zIndex + INPUT_DELTA_UNIT_OFFSET);
	int temp1;
	int temp2;
	for (int y = 0; y < HEIGHT_LIMIT; y++)
	{
		temp1 = tempDeltaIndex + INPUT_DELTA_UNIT_MEMORY_WIDTH * (y + INPUT_DELTA_UNIT_MEMORY_HEIGHT_OFFSET);
		temp2 = INPUT_UNIT_MEMORY_WIDTH * (y + INPUT_UNIT_MEMORY_HEIGHT_OFFSET + yIndex);
		for (int x = 0; x < WIDTH_LIMIT; x++)
		{
			//printf("input delta: %f \n", inputDelta[temp1 + INPUT_DELTA_UNIT_MEMORY_WIDTH_OFFSET + x]);
			//printf("input: %f \n", input[temp2 + INPUT_UNIT_MEMORY_WIDTH_OFFSET + x + xIndex]);
			sum += inputDelta[temp1 + INPUT_DELTA_UNIT_MEMORY_WIDTH_OFFSET + x] * input[temp2 + INPUT_UNIT_MEMORY_WIDTH_OFFSET + x + xIndex];
		}
	}

	//printf("(%i) : %f \n", outputIndex, sum);
	output[outputIndex] = sum;
}
