

#include "RealType.h"

//<!@
#define SAMPLING_SIZE_WIDTH -1
#define SAMPLING_SIZE_HEIGHT -1
#define INPUT_UNIT_MEMORY_WIDTH_OFFSET -1
#define INPUT_UNIT_MEMORY_HEIGHT_OFFSET -1
#define OUTPUT_UNIT_MEMORY_WIDTH_OFFSET -1
#define OUTPUT_UNIT_MEMORY_HEIGHT_OFFSET -1
#define OUTPUT_UNIT_OFFSET -1
#define INPUT_UNIT_OFFSET -1
#define OUTPUT_UNIT_MEMORY_WIDTH -1
#define INPUT_UNIT_MEMORY_WIDTH -1
#define OUTPUT_UNIT_MEMORY_ELEMENTS -1
#define INPUT_UNIT_MEMORY_ELEMENTS -1
#define REMAINDER_SIZE_WIDTH -1
#define REMAINDER_SIZE_HEIGHT -1
#define X_MAX_INDEX -1
#define Y_MAX_INDEX -1
//#define CONSTANT_INPUT
//#define USE_REMAINDER
//!@>


__kernel void MaxPoolingSamplingKernel(
#ifdef CONSTANT_INPUT
__constant real_t* input,
#else
__global const real_t* input,
#endif
__global real_t* output,
__global int* xMaxIndices,
__global int* yMaxIndices
)
{
	const int xIndex = get_global_id(0);
	const int yIndex = get_global_id(1);
	const int zIndex = get_global_id(2);

	real_t maxValue = FLT_MIN;
	int maxXIndex;
	int maxYIndex;

	const int xStart = xIndex * SAMPLING_SIZE_WIDTH + INPUT_UNIT_MEMORY_WIDTH_OFFSET;
	const int xStop = xStart + SAMPLING_SIZE_WIDTH;

	const int yStart = yIndex * SAMPLING_SIZE_HEIGHT + INPUT_UNIT_MEMORY_HEIGHT_OFFSET;
	const int yStop = yStart + SAMPLING_SIZE_HEIGHT;


	int tempIndex1 = INPUT_UNIT_MEMORY_ELEMENTS * (INPUT_UNIT_OFFSET + zIndex);
	int tempIndex2;
	real_t tempValue;

	#ifdef USE_REMAINDER
	
	if (xIndex == X_MAX_INDEX && yIndex == Y_MAX_INDEX)
	{
		const int xMaxInputIndex = xStart + REMAINDER_SIZE_WIDTH;
		const int yMaxInputIndex = yStart + REMAINDER_SIZE_HEIGHT;

		for (int y = yStart; y < xMaxInputIndex; y++)
		{
			tempIndex2 = tempIndex1 + INPUT_UNIT_MEMORY_WIDTH * y;
			for (int x = xStart; x < yMaxInputIndex; x++)
			{
				tempValue = input[x + tempIndex2];
				if (tempValue > maxValue)
				{
					maxXIndex = x - INPUT_UNIT_MEMORY_WIDTH_OFFSET; //This needs to be memory neutral for back-prop
					maxYIndex = y - INPUT_UNIT_MEMORY_HEIGHT_OFFSET; //This needs to be memory neutral for back-prop
					maxValue = tempValue;
				}
			}
		}
	}
	else if (xIndex == X_MAX_INDEX)
	{
		const int xMaxInputIndex = xStart + REMAINDER_SIZE_WIDTH;
		for (int y = yStart; y < yStop; y++)
		{
			tempIndex2 = tempIndex1 + INPUT_UNIT_MEMORY_WIDTH * y;
			for (int x = xStart; x < xMaxInputIndex; x++)
			{
				tempValue = input[x + tempIndex2];
				if (tempValue > maxValue)
				{
					maxXIndex = x - INPUT_UNIT_MEMORY_WIDTH_OFFSET;	//This needs to be memory neutral for back-prop
					maxYIndex = y - INPUT_UNIT_MEMORY_HEIGHT_OFFSET; //This needs to be memory neutral for back-prop
					maxValue = tempValue;
				}
			}
		}
	}
	else if (yIndex == Y_MAX_INDEX)
	{
		const int yMaxInputIndex = yStart + REMAINDER_SIZE_HEIGHT;
		for (int y = yStart; y < yMaxInputIndex; y++)
		{
			tempIndex2 = tempIndex1 + INPUT_UNIT_MEMORY_WIDTH * y;
			for (int x = xStart; x < xStop; x++)
			{
				tempValue = input[x + tempIndex2];
				if (tempValue > maxValue)
				{
					maxXIndex = x - INPUT_UNIT_MEMORY_WIDTH_OFFSET;	//This needs to be memory neutral for back-prop
					maxYIndex = y - INPUT_UNIT_MEMORY_HEIGHT_OFFSET; //This needs to be memory neutral for back-prop
					maxValue = tempValue;
				}
			}
		}
	}
	else
	#endif
	{
		for (int y = yStart; y < yStop; y++)
		{
			tempIndex2 = tempIndex1 + INPUT_UNIT_MEMORY_WIDTH * y;
			for (int x = xStart; x < xStop; x++)
			{
				tempValue = input[x + tempIndex2];
				if (tempValue > maxValue)
				{
					maxXIndex = x - INPUT_UNIT_MEMORY_WIDTH_OFFSET;	//This needs to be memory neutral for back-prop
					maxYIndex = y - INPUT_UNIT_MEMORY_HEIGHT_OFFSET; //This needs to be memory neutral for back-prop
					maxValue = tempValue;
				}
			}
		}
	}

	const int outputIndex = xIndex + OUTPUT_UNIT_MEMORY_WIDTH_OFFSET + OUTPUT_UNIT_MEMORY_WIDTH * (yIndex + OUTPUT_UNIT_MEMORY_HEIGHT_OFFSET) + 
		OUTPUT_UNIT_MEMORY_ELEMENTS * (OUTPUT_UNIT_OFFSET + zIndex);

	output[outputIndex] = maxValue;
	xMaxIndices[xIndex] = maxXIndex;
	yMaxIndices[yIndex] = maxYIndex;

}