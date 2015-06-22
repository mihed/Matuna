
#include "RealType.h"
#include "ActivationFunction.h"

//<!@
#define MAX_INPUT_DELTA_X_INDEX -1
#define MAX_INPUT_DELTA_Y_INDEX -1
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
//!@>

__kernel void VanillaUpSamplingKernel(
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

	const int outputIndex = OUTPUT_UNIT_MEMORY_WIDTH_OFFSET + xIndex + OUTPUT_UNIT_MEMORY_WIDTH * 
	(OUTPUT_UNIT_MEMORY_HEIGHT_OFFSET + yIndex) + OUTPUT_UNIT_MEMORY_ELEMENTS * (OUTPUT_UNIT_OFFSET + zIndex);

	//If this turns out to be a perfomance bottleneck somehwere in the future, it's worth chaning the modulus and branching
	if (((xIndex % SAMPLING_SIZE_WIDTH) == 0) && ((yIndex % SAMPLING_SIZE_HEIGHT) == 0))
	{
		const int xInputIndex = xIndex / SAMPLING_SIZE_WIDTH;
		const int yInputIndex = yIndex / SAMPLING_SIZE_HEIGHT;

		if ((xInputIndex < MAX_INPUT_DELTA_X_INDEX) && (yInputIndex < MAX_INPUT_DELTA_Y_INDEX))
		{
			const int inputDeltaIndex = INPUT_DELTA_UNIT_MEMORY_WIDTH_OFFSET + xInputIndex + INPUT_DELTA_UNIT_MEMORY_WIDTH * 
			(INPUT_DELTA_UNIT_MEMORY_HEIGHT_OFFSET + yInputIndex) + INPUT_DELTA_UNIT_MEMORY_ELEMENTS * (INPUT_DELTA_UNIT_OFFSET + zIndex);
			
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
	else
	{
		output[outputIndex] = 0;
	}
}