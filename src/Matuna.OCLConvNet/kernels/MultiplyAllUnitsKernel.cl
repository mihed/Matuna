#include "RealType.h"
#include "ActivationFunction.h"

//<!@
#define INPUT_DELTA_STRIDE -1
#define OUTPUT_STRIDE -1
#define INPUT_STRIDE -1
#define INPUT_DELTA_WIDTH_OFFSET -1
#define INPUT_DELTA_HEIGHT_OFFSET -1
#define OUTPUT_WIDTH_OFFSET -1
#define OUTPUT_HEIGHT_OFFSET -1
#define OUTPUT_UNIT_OFFSET -1
#define INPUT_WIDTH_OFFSET -1
#define INPUT_HEIGHT_OFFSET -1
#define INPUT_UNIT_OFFSET -1
#define OUTPUT_UNIT_ELEMENT_COUNT_INC_PADDING -1
#define INPUT_UNIT_ELEMENT_COUNT_INC_PADDING -1
//#define CONSTANT_INPUT
//#define CONSTANT_INPUT_DELTA
//!@>

__kernel void MultiplyAllUnitsKernel(
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

	const int outputIndex = xIndex + OUTPUT_WIDTH_OFFSET + OUTPUT_STRIDE * (yIndex + OUTPUT_HEIGHT_OFFSET) + OUTPUT_UNIT_ELEMENT_COUNT_INC_PADDING * (zIndex + OUTPUT_UNIT_OFFSET);

	const real_t tempInputDelta = inputDelta[xIndex + INPUT_DELTA_WIDTH_OFFSET + INPUT_DELTA_STRIDE * (yIndex + INPUT_DELTA_HEIGHT_OFFSET)];

	const int inputIndex = xIndex + INPUT_WIDTH_OFFSET + INPUT_STRIDE * (yIndex + INPUT_HEIGHT_OFFSET) + INPUT_UNIT_ELEMENT_COUNT_INC_PADDING * (zIndex + INPUT_UNIT_OFFSET);
	const real_t tempInput = input[inputIndex];

	output[outputIndex] = ACTIVATION_DERIVATIVE(tempInputDelta, tempInput);
}
