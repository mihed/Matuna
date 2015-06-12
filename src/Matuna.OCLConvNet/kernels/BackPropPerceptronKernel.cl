/**
 *Macros to define:
 * - INPUT_DELTA_COUNT: The amount of units of the input delta
 * - DOUBLE_PRECISION: If the kernel is to be executed with double precision
 * - CONSTANT_INPUT: If we may put the inputs into the constant memory space
 * - CONSTANT_WEIGHTS: If we may put the weights into the constant memory space
 * - CONSTANT_INPUT_DELTA: If the delta is to be put into constant memory space
 * - WEIGHT_COLUMN_COUNT: The column dimension of the matrix
 * - INPUT_OFFSET: The unit offset of the input
 * - OUTPUT_DELTA_OFFSET: The unit offset of the delta output
 * - INPUT_DELTA_OFFSET: The unit offset of the input delta
 * - SIGMOID: If we are using sigmoid activation
 * - TANH: If we are using tanh activation
 */

#include "RealType.h"
#include "ActivationFunction.h"

//<!@
#define INPUT_OFFSET 0
#define OUTPUT_DELTA_OFFSET 0
#define INPUT_DELTA_OFFSET 0
#define INPUT_DELTA_COUNT -1
#define WEIGHT_COLUMN_COUNT -1
//#define CONSTANT_INPUT
//#define CONSTANT_INPUT_DELTA
//#define CONSTANT_WEIGHTS
//!@>

__kernel void BackPerceptronKernel(
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

		__global real_t* outputDelta,

#ifdef CONSTANT_WEIGHTS
		__constant real_t* weights
#else
		__global const real_t* weights
#endif
)
{
	const int columnIndex = get_global_id(0);

	real_t sum = 0;
	for (int y = INPUT_DELTA_OFFSET; y < INPUT_DELTA_COUNT; y++)
	{
		sum += inputDelta[y] * weights[columnIndex + WEIGHT_COLUMN_COUNT * y];
	}

	const real_t tempInput = input[columnIndex + INPUT_DELTA_OFFSET];
	outputDelta[columnIndex + OUTPUT_DELTA_OFFSET] = ACTIVATION_DERIVATIVE(sum, tempInput);
}
