
#include "RealType.h"
#include "ActivationFunction.h"

//<!@
#define MAX_INPUT_DELTA_X_INDEX -1
#define MAX_INPUT_DELTA_Y_INDEX -1
#define INPUT_DELTA_UNIT_WIDTH -1
#define INPUT_DELTA_UNIT_ELEMENTS -1
#define SAMPLING_SIZE_WIDTH -1
#define SAMPLING_SIZE_HEIGHT -1
#define INPUT_DELTA_UNIT_MEMORY_WIDTH_OFFSET -1
#define INPUT_DELTA_UNIT_MEMORY_HEIGHT_OFFSET -1
#define OUTPUT_UNIT_MEMORY_WIDTH_OFFSET -1
#define OUTPUT_UNIT_MEMORY_HEIGHT_OFFSET -1
#define INPUT_UNIT_MEMORY_WIDTH_OFFSET -1
#define INPUT_UNIT_MEMORY_HEIGHT_OFFSET -1
#define OUTPUT_UNIT_OFFSET -1
#define INPUT_DELTA_UNIT_OFFSET -1
#define INPUT_UNIT_OFFSET -1
#define OUTPUT_UNIT_MEMORY_WIDTH -1
#define INPUT_DELTA_UNIT_MEMORY_WIDTH -1
#define INPUT_UNIT_MEMORY_WIDTH -1
#define OUTPUT_UNIT_MEMORY_ELEMENTS -1
#define INPUT_DELTA_UNIT_MEMORY_ELEMENTS -1
#define INPUT_UNIT_MEMORY_ELEMENTS -1
//#define CONSTANT_INPUT_DELTA
//#define CONSTANT_INPUT
//#define CONSTANT_X_MAX_INDICES
//#define CONSTANT_Y_MAX_INDICES
//!@>

__kernel void MaxPoolingUpSamplingKernel(
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

__global real_t* output,

#ifdef CONSTANT_X_MAX_INDICES
__constant int* xMaxIndices,
#else
__global const int* xMaxIndices,
#endif

#ifdef CONSTANT_Y_MAX_INDICES
__constant int* yMaxIndices
#else
__global const int* yMaxIndices
#endif
)
{
	const int xIndex = get_global_id(0);
	const int yIndex = get_global_id(1);
	const int zIndex = get_global_id(2);

	//Determine the index of the bucket we are in
	int xBucketIndex = (int)floor(xIndex / (float)SAMPLING_SIZE_WIDTH);
	int yBucketIndex = (int)floor(yIndex / (float)SAMPLING_SIZE_HEIGHT);

	//These to checks are neccessarliy since we can support non-divisable sampling layers
	if (MAX_INPUT_DELTA_X_INDEX <  xBucketIndex)
	{
		xBucketIndex = MAX_INPUT_DELTA_X_INDEX;
	}

	if (MAX_INPUT_DELTA_Y_INDEX <  yBucketIndex)
	{
		yBucketIndex = MAX_INPUT_DELTA_Y_INDEX; 
	}

	const int tempIndex = xBucketIndex + INPUT_DELTA_UNIT_WIDTH * yBucketIndex + zIndex * INPUT_DELTA_UNIT_ELEMENTS; // Simply the index of the succeeding layer

	const int xMaxIndex = xMaxIndices[tempIndex];
	const int yMaxIndex = yMaxIndices[tempIndex];

	const int outputIndex = OUTPUT_UNIT_MEMORY_WIDTH_OFFSET + xIndex + OUTPUT_UNIT_MEMORY_WIDTH * 
	(OUTPUT_UNIT_MEMORY_HEIGHT_OFFSET + yIndex) + OUTPUT_UNIT_MEMORY_ELEMENTS * (OUTPUT_UNIT_OFFSET + zIndex);


	if (xIndex == xMaxIndex && yIndex == yMaxIndex)
	{
		const int inputDeltaIndex = INPUT_DELTA_UNIT_MEMORY_WIDTH_OFFSET + xBucketIndex + INPUT_DELTA_UNIT_MEMORY_WIDTH * 
			(INPUT_DELTA_UNIT_MEMORY_HEIGHT_OFFSET + yBucketIndex) + INPUT_DELTA_UNIT_MEMORY_ELEMENTS * (INPUT_DELTA_UNIT_OFFSET + zIndex);

		const int inputIndex = INPUT_UNIT_MEMORY_WIDTH_OFFSET + xIndex + INPUT_UNIT_MEMORY_WIDTH * 
			(INPUT_UNIT_MEMORY_HEIGHT_OFFSET + yIndex) + INPUT_UNIT_MEMORY_ELEMENTS * (INPUT_UNIT_OFFSET + zIndex);

		const real_t tempInputDelta = inputDelta[inputDeltaIndex];
		const real_t tempInput = input[inputIndex];

		output[outputIndex] = ACTIVATION_DERIVATIVE(tempInputDelta, tempInput);
	}
	else
	{
		output[outputIndex] = 0;
	}
}