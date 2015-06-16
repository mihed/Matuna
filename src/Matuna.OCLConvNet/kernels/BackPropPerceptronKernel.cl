//Since we don't allow for perceptrons that look like images
//We only have offset in the unit direction for the deltas


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
 * - INPUT_DELTA_UNIT_OFFSET: The unit offset of the input delta
 * - SIGMOID: If we are using sigmoid activation
 * - TANH: If we are using tanh activation
 */

#include "RealType.h"
#include "ActivationFunction.h"

//<!@
#define OUTPUT_UNIT_MEMORY_WIDTH_OFFSET -1
#define OUTPUT_UNIT_MEMORY_HEIGHT_OFFSET -1
#define OUTPUT_UNIT_OFFSET -1
#define OUTPUT_UNIT_MEMORY_WIDTH -1
#define OUTPUT_UNIT_MEMORY_ELEMENTS -1
#define INPUT_UNIT_MEMORY_WIDTH_OFFSET -1
#define INPUT_UNIT_MEMORY_HEIGHT_OFFSET -1
#define INPUT_UNIT_OFFSET -1
#define INPUT_UNIT_MEMORY_WIDTH -1
#define INPUT_UNIT_MEMORY_ELEMENTS -1
#define INPUT_DELTA_UNIT_OFFSET -1
#define INPUT_DELTA_LIMIT -1
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

	const int xIndex = get_global_id(0);
	const int yIndex = get_global_id(1);
	const int zIndex = get_global_id(2);
	const int columnIndex = xIndex + get_global_size(0) * yIndex + get_global_size(0) * get_global_size(1) * zIndex;

	const int outputDeltaIndex = OUTPUT_UNIT_MEMORY_WIDTH_OFFSET + xIndex + OUTPUT_UNIT_MEMORY_WIDTH * (OUTPUT_UNIT_MEMORY_HEIGHT_OFFSET + yIndex) + OUTPUT_UNIT_MEMORY_ELEMENTS * (OUTPUT_UNIT_OFFSET + zIndex);
	const int inputIndex = INPUT_UNIT_MEMORY_WIDTH_OFFSET + xIndex + INPUT_UNIT_MEMORY_WIDTH * (INPUT_UNIT_MEMORY_HEIGHT_OFFSET + yIndex) + INPUT_UNIT_MEMORY_ELEMENTS * (INPUT_UNIT_OFFSET + zIndex);

	real_t sum = 0;
	for (int y = INPUT_DELTA_UNIT_OFFSET; y < INPUT_DELTA_LIMIT; y++)
	{
		sum += inputDelta[y] * weights[columnIndex + WEIGHT_COLUMN_COUNT * y];
	}

	const real_t tempInput = input[inputIndex];
	outputDelta[outputDeltaIndex] = ACTIVATION_DERIVATIVE(sum, tempInput);
}
