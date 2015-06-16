/**
 *Macros to define:
 * - INPUT_UNITS_LIMIT: The limit in the input units
 * - INPUT_WIDTH: The input width
 * - INPUT_HEIGHT: The input height
 * - INPUT_UNITS_OFFSET: The offset in the units direction
 * - INPUT_WIDTH_OFFSET: The offset in the width direction
 * - INPUT_HEIGHT_OFFSET: The offset in the height direction
 * - COLUMN_COUNT: The amount of columns in the weight matrix
 * - INPUT_UNIT_ELEMENT_COUNT_INC_PADDING: The amount of elements inside one unit with padding
 * - INPUT_MEMORY_WIDTH: The width of the memory
 * - DOUBLE_PRECISION: If the kernel is to be executed with double precision
 * - CONSTANT_INPUT: If we may put the inputs into the constant memory space
 * - CONSTANT_WEIGHTS: If we may put the weights into the constant memory space
 * - CONSTANT_BIASES: If we may put the biases into the constant memory space
 * - SIGMOID: If we are using sigmoid activation
 * - TANH: If we are using tanh activation
 * - SOFTMAX: If we are using the softmax activation
 * - HALF_MATH: If we use half precision math
 * - NATIVE_MATH: If we use native precision math
 */

#include "RealType.h"
#include "ActivationFunction.h"

//<!@
#define INPUT_UNITS_LIMIT -1
#define INPUT_WIDTH_LIMIT -1
#define INPUT_HEIGHT_LIMIT -1
#define INPUT_UNITS_OFFSET -1
#define INPUT_WIDTH_OFFSET -1
#define INPUT_HEIGHT_OFFSET -1
#define COLUMN_COUNT -1
#define OUTPUT_UNIT_OFFSET -1
#define INPUT_UNIT_ELEMENT_COUNT_INC_PADDING -1
#define INPUT_MEMORY_WIDTH -1
//#define CONSTANT_INPUT
//#define CONSTANT_WEIGHTS
//#define CONSTANT_BIASES
//!@>

__kernel void ForwardPerceptronKernel(
#ifdef CONSTANT_INPUT
		__constant real_t* input,
#else
		__global const real_t* input,
#endif

		__global real_t* output,

#ifdef CONSTANT_WEIGHTS
		__constant real_t* weights,
#else
		__global const real_t* weights,
#endif

#ifdef CONSTANT_BIASES
		__constant real_t* biases
#else
		__global const real_t* biases
#endif
)
{
	const int outputIndex = get_global_id(0);
	const int rowIndex = COLUMN_COUNT * outputIndex;

	real_t sum = 0;
	int columnIndex = 0;
	int tempZIndex = 0;
	int tempYIndex = 0;
	for (int unit = INPUT_UNITS_OFFSET; unit < INPUT_UNITS_LIMIT; unit++)
	{
		tempZIndex = unit * INPUT_UNIT_ELEMENT_COUNT_INC_PADDING;
		for (int row = INPUT_HEIGHT_OFFSET; row < INPUT_HEIGHT_LIMIT; row++)
		{
			tempYIndex = row * INPUT_MEMORY_WIDTH + tempZIndex;
			for(int column = INPUT_WIDTH_OFFSET; column < INPUT_WIDTH_LIMIT; column++)
			{
				sum += input[tempYIndex + column] * weights[rowIndex + columnIndex];
				columnIndex++;
			}
		}
	}

	const real_t sumBias = sum + biases[outputIndex];
	output[outputIndex + OUTPUT_UNIT_OFFSET] = ACTIVATION(sumBias);
}
