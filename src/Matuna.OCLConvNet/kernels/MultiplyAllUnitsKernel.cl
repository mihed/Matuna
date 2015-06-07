#include "RealType.h"
#include "ActivationFunction.h"

#ifndef INPUT_DELTA_STRIDE
#define INPUT_DELTA_STRIDE -1
#endif

#ifndef OUTPUT_STRIDE
#define OUTPUT_STRIDE -1
#endif

#ifndef INPUT_STRIDE
#define INPUT_STRIDE -1
#endif

//Observe that this offset is offset to the global id
#ifndef INPUT_DELTA_WIDTH_OFFSET
#define INPUT_DELTA_WIDTH_OFFSET -1
#endif

//Observe that this offset is offset to the global id
#ifndef INPUT_DELTA_HEIGHT_OFFSET
#define INPUT_DELTA_HEIGHT_OFFSET -1
#endif

#ifndef OUTPUT_WIDTH_OFFSET
#define OUTPUT_WIDTH_OFFSET -1
#endif

#ifndef OUTPUT_HEIGHT_OFFSET
#define OUTPUT_HEIGHT_OFFSET -1
#endif

#ifndef OUTPUT_UNIT_OFFSET
#define OUTPUT_UNIT_OFFSET -1
#endif

#ifndef INPUT_WIDTH_OFFSET
#define INPUT_WIDTH_OFFSET -1
#endif

#ifndef INPUT_HEIGHT_OFFSET
#define INPUT_HEIGHT_OFFSET -1
#endif

#ifndef INPUT_UNIT_OFFSET
#define INPUT_UNIT_OFFSET -1
#endif

//Width * Height
#ifndef OUTPUT_UNIT_ELEMENT_COUNT_INC_PADDING 
#define OUTPUT_UNIT_ELEMENT_COUNT_INC_PADDING -1
#endif

//Width * Height
#ifndef INPUT_UNIT_ELEMENT_COUNT_INC_PADDING 
#define INPUT_UNIT_ELEMENT_COUNT_INC_PADDING -1
#endif

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
