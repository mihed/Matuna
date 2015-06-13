#include "RealType.h"

//<!@
#define INPUT_DELTA_STRIDE -1
#define OUTPUT_STRIDE -1
#define INPUT_STRIDE -1
#define INPUT_WIDTH_OFFSET -1
#define INPUT_HEIGHT_OFFSET -1
#define INPUT_DELTA_WIDTH_OFFSET -1
#define INPUT_DELTA_HEIGHT_OFFSET -1
#define INPUT_DELTA_UNIT_OFFSET -1
#define WIDTH_LIMIT -1
#define HEIGHT_LIMIT -1
#define OUTPUT_UNIT_ELEMENT_COUNT_INC_PADDING -1
#define INPUT_DELTA_UNIT_ELEMENT_COUNT_INC_PADDING -1
#define OUTPUT_WIDTH_OFFSET -1
#define OUTPUT_HEIGHT_OFFSET -1
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

	/*
	 printf("xIndex: %i \n", xIndex);
	 printf("yIndex: %i \n", yIndex);
	 printf("zIndex: %i \n", zIndex);

	 if (xIndex == 0 && yIndex == 0 && zIndex == 0)
	 {
	 printf("OUTPUT_WIDTH_OFFSET: %i \n", OUTPUT_WIDTH_OFFSET);
	 printf("OUTPUT_HEIGHT_OFFSET: %i \n", OUTPUT_HEIGHT_OFFSET);
	 printf("OUTPUT_UNIT_OFFSET: %i \n", OUTPUT_UNIT_OFFSET);
	 printf("OUTPUT_STRIDE: %i \n", OUTPUT_STRIDE);
	 printf("OUTPUT_UNIT_ELEMENT_COUNT_INC_PADDING: %i \n", OUTPUT_UNIT_ELEMENT_COUNT_INC_PADDING);
	 printf("INPUT_DELTA_WIDTH_OFFSET: %i \n", INPUT_DELTA_WIDTH_OFFSET);
	 printf("INPUT_DELTA_HEIGHT_OFFSET: %i \n", INPUT_DELTA_HEIGHT_OFFSET);
	 printf("INPUT_DELTA_UNIT_OFFSET: %i \n", INPUT_DELTA_UNIT_OFFSET);
	 printf("INPUT_DELTA_STRIDE: %i \n", INPUT_DELTA_STRIDE);
	 printf("INPUT_DELTA_UNIT_ELEMENT_COUNT_INC_PADDING: %i \n", INPUT_DELTA_UNIT_ELEMENT_COUNT_INC_PADDING);
	 printf("HEIGHT_LIMIT: %i \n", HEIGHT_LIMIT);
	 printf("WIDTH_LIMIT: %i \n", WIDTH_LIMIT);
	 printf("INPUT_WIDTH_OFFSET: %i \n", INPUT_WIDTH_OFFSET);
	 printf("INPUT_STRIDE: %i \n", INPUT_STRIDE);
	 printf("INPUT_HEIGHT_OFFSET: %i \n", INPUT_HEIGHT_OFFSET);
	 printf("zIndex: %i \n", zIndex);
	 }
	 */

	const int outputIndex = xIndex + OUTPUT_WIDTH_OFFSET + (yIndex + OUTPUT_HEIGHT_OFFSET) * OUTPUT_STRIDE + (zIndex + OUTPUT_UNIT_OFFSET) * OUTPUT_UNIT_ELEMENT_COUNT_INC_PADDING;

	real_t sum = 0;

	const int tempDeltaIndex = INPUT_DELTA_UNIT_ELEMENT_COUNT_INC_PADDING * (zIndex + INPUT_DELTA_UNIT_OFFSET);
	int temp1;
	int temp2;
	for (int y = 0; y < HEIGHT_LIMIT; y++)
	{
		temp1 = tempDeltaIndex + INPUT_DELTA_STRIDE * (y + INPUT_DELTA_HEIGHT_OFFSET);
		temp2 = INPUT_STRIDE * (y + INPUT_HEIGHT_OFFSET + yIndex);
		for (int x = 0; x < WIDTH_LIMIT; x++)
		{
			//printf("input delta: %f \n", inputDelta[temp1 + INPUT_DELTA_WIDTH_OFFSET + x]);
			//printf("input: %f \n", input[temp2 + INPUT_WIDTH_OFFSET + x + xIndex]);
			sum += inputDelta[temp1 + INPUT_DELTA_WIDTH_OFFSET + x] * input[temp2 + INPUT_WIDTH_OFFSET + x + xIndex];
		}
	}

	//printf("(%i) : %f \n", outputIndex, sum);
	output[outputIndex] = sum;
}
