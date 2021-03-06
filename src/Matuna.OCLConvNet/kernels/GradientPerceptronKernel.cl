//Since we don't support perceptrons that are located as an image at the moment
//we only allow for INPUT_DELTA_UNIT_OFFSET 


/**
 *Macros to define:
 * - CONSTANT_INPUT: If we may put the inputs into the constant memory space
 * - CONSTANT_INPUT_DELTA: If the delta is to be put into constant memory space
 * - INPUT_OFFSET: The unit offset of the input
 * - INPUT_DELTA_UNIT_OFFSET: The unit offset of the input delta
 * - WEIGHT_COLUMN_COUNT: The column dimension of the matrix
 * - DOUBLE_PRECISION: if double precision is to be used
 */

#include "RealType.h"

//<!@
#define INPUT_UNIT_WIDTH -1
#define INPUT_UNIT_ELEMENTS -1
#define INPUT_UNIT_MEMORY_WIDTH_OFFSET -1
#define INPUT_UNIT_MEMORY_HEIGHT_OFFSET -1
#define INPUT_UNIT_OFFSET -1
#define INPUT_UNIT_MEMORY_WIDTH -1
#define INPUT_UNIT_MEMORY_ELEMENTS -1
#define INPUT_DELTA_UNIT_OFFSET 0
#define WEIGHT_COLUMN_COUNT -1
//#define CONSTANT_INPUT
//#define CONSTANT_INPUT_DELTA
//!@>

__kernel void ImageGradientPerceptronKernel(
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

		__global real_t* outputGradient
)
{
	const int xIndex = get_global_id(0);
	const int yIndex = get_global_id(1);
	const real_t realValue = (real_t)xIndex;

	const int zIndexInputData = (int)(floor(realValue / INPUT_UNIT_ELEMENTS));
	const int temp = zIndexInputData * INPUT_UNIT_ELEMENTS;
	const int yIndexInputData = (int)(floor((realValue - temp) / INPUT_UNIT_WIDTH));
	const int xIndexInputData = xIndex - temp - yIndexInputData * INPUT_UNIT_WIDTH;

	const int inputIndex = xIndexInputData + INPUT_UNIT_MEMORY_WIDTH_OFFSET + INPUT_UNIT_MEMORY_WIDTH * (INPUT_UNIT_MEMORY_HEIGHT_OFFSET + yIndexInputData) + INPUT_UNIT_MEMORY_ELEMENTS * (INPUT_UNIT_OFFSET + zIndexInputData);

	outputGradient[yIndex * WEIGHT_COLUMN_COUNT + xIndex] = inputDelta[yIndex + INPUT_DELTA_UNIT_OFFSET] * input[inputIndex];
}
