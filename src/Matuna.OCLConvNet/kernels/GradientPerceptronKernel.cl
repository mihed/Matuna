/**
 *Macros to define:
 * - CONSTANT_INPUT: If we may put the inputs into the constant memory space
 * - CONSTANT_INPUT_DELTA: If the delta is to be put into constant memory space
 * - INPUT_OFFSET: The unit offset of the input
 * - INPUT_DELTA_OFFSET: The unit offset of the input delta
 * - WEIGHT_COLUMN_COUNT: The column dimension of the matrix
 * - DOUBLE_PRECISION: if double precision is to be used
 */

#include "RealType.h"

#ifndef INPUT_OFFSET
#define INPUT_OFFSET 0
#endif

#ifndef INPUT_DELTA_OFFSET
#define INPUT_DELTA_OFFSET 0
#endif

#ifndef WEIGHT_COLUMN_COUNT
#define WEIGHT_COLUMN_COUNT -1
#endif

__kernel void GradientPerceptronKernel(
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

	outputGradient[yIndex * WEIGHT_COLUMN_COUNT + xIndex] = inputDelta[yIndex + INPUT_DELTA_OFFSET] * input[xIndex + INPUT_OFFSET];
}
