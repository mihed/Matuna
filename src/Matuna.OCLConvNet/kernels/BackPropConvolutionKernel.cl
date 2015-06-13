//This kernel is back propagation for a convolution layer that is fully connected.
//Futhermore, it is assumed that the inputDelta has zero padding / offset around itself with FILTER_WIDTH - 1 / FILTER_HEIGHT -1 size.


#include "RealType.h"

//TEST------------
//#pragma OPENCL EXTENSION cl_intel_printf : enable
//TEST------------

//<!@
#define FILTER_WIDTH -1
#define FILTER_HEIGHT -1
#define INPUT_UNIT_OFFSET -1
#define INPUT_UNIT_LIMIT -1
#define INPUT_UNIT_COUNT -1
#define INPUT_STRIDE -1
#define OUTPUT_STRIDE -1
#define INPUT_WIDTH_OFFSET -1
#define INPUT_HEIGHT_OFFSET -1
#define OUTPUT_WIDTH_OFFSET -1
#define OUTPUT_HEIGHT_OFFSET -1
#define INPUT_UNIT_ELEMENT_COUNT_INC_PADDING -1
#define FILTER_UNIT_ELEMENT_COUNT_INC_PADDING -1
//#define CONSTANT_INPUT
//#define USE_LOCAL_MEMORY
//#define CONSTANT_FILTERS
//!@>

__kernel void BackPropConvolutionKernel(
#ifdef CONSTANT_INPUT
		__constant real_t* inputDelta,
#else
		__global const real_t* inputDelta,
#endif

		__global real_t* output,

#ifdef USE_LOCAL_MEMORY
#ifdef CONSTANT_FILTERS
		__constant real_t* filters,
#else
		__global const real_t* filters,
#endif
		__local real_t* cache
#else
#ifdef CONSTANT_FILTERS
		__constant real_t* filters
#else
		__global const real_t* filters
#endif
#endif
)
{
	const int xIndex = get_global_id(0);
	const int yIndex = get_global_id(1);

	const int tempYIndex = (yIndex + INPUT_HEIGHT_OFFSET) * INPUT_STRIDE + xIndex + INPUT_WIDTH_OFFSET;
	const int maxFilterHeightIndex = FILTER_HEIGHT - 1;
	const int maxFilterWidthIndex = FILTER_WIDTH - 1;

	//TEST------------
	//printf("xIndex: %i \n", xIndex);
	//printf("yIndex: %i \n", yIndex);
	//TEST------------

#ifdef USE_LOCAL_MEMORY
	const int xIndexLocal = get_local_id(0);
	const int yIndexLocal = get_local_id(1);
	const int xMaxIndexLocal = get_local_size(0) - 1;
	const int yMaxIndexLocal = get_local_size(1) - 1;
	const int localCacheWidth = xMaxIndexLocal + FILTER_WIDTH;
	const int localIndex = xIndexLocal + localCacheWidth * yIndexLocal;
	const int localElementCount = localCacheWidth * (yMaxIndexLocal + FILTER_HEIGHT);

	//TEST------------
	//printf("xIndex local: %i \n", xIndexLocal);
	//printf("yIndex local: %i \n", yIndexLocal);
	//TEST------------

	int localTemp1;
	int localTemp2;
	int globalTemp1;
	int globalTemp2;
	int localIndexCounter = 0;
	for (int i = INPUT_UNIT_OFFSET; i < INPUT_UNIT_LIMIT; i++)
	{
		globalTemp1 = INPUT_UNIT_ELEMENT_COUNT_INC_PADDING * i + tempYIndex;
		localTemp1 = localElementCount * localIndexCounter + localIndex;
		localIndexCounter++;
		if (xIndexLocal == xMaxIndexLocal && yIndexLocal == yMaxIndexLocal)
		{
			for (int u = 0; u < FILTER_HEIGHT; u++)
			{
				localTemp2 = localTemp1 + u * localCacheWidth;
				globalTemp2 = globalTemp1 + u * INPUT_STRIDE;
				for (int v = 0; v < FILTER_WIDTH; v++)
				{
					cache[localTemp2 + v] = inputDelta[globalTemp2 + v];
					//TEST------------
					//printf("Cache double loop: %f \n", cache[localTemp2 + v]);
					//TEST------------
				}
			}
		}
		else if (xIndexLocal == xMaxIndexLocal)
		{
			for (int v = 0; v < FILTER_WIDTH; v++)
			{
				cache[localTemp1 + v] = inputDelta[globalTemp1 + v];
				//TEST------------
				//printf("Cache width loop: %f \n", cache[localTemp1 + v]);
				//TEST------------
			}
		}
		else if (yIndexLocal == yMaxIndexLocal)
		{
			for (int u = 0; u < FILTER_HEIGHT; u++)
			{
				cache[localTemp1 + u * localCacheWidth] = inputDelta[globalTemp1 + u * INPUT_STRIDE];
				//TEST------------
				//printf("Cache height loop: %f \n", cache[localTemp1 + u * localCacheWidth]);
				//TEST------------
			}
		}
		else
		{
			cache[localTemp1] = inputDelta[globalTemp1];
			//TEST------------
			//printf("Cache assignment: %f \n", cache[localTemp1]);
			//TEST------------
		}
	}

	barrier(CLK_LOCAL_MEM_FENCE);

#endif

	int tempIndex;
	int tempIndex2;
	int tempIndex3;
	int tempIndex4;
	real_t sum = 0;

#ifdef USE_LOCAL_MEMORY
	for (int i = 0; i < INPUT_UNIT_COUNT; i++)
	{
		tempIndex = localElementCount * i + localIndex;
		tempIndex4 = FILTER_UNIT_ELEMENT_COUNT_INC_PADDING * i;
		for (int u = 0; u < FILTER_HEIGHT; u++)
		{
			tempIndex2 = tempIndex + localCacheWidth * u;
			tempIndex3 = FILTER_WIDTH * (maxFilterHeightIndex - u) + tempIndex4;
			for (int v = 0; v < FILTER_WIDTH; v++)
			{
				sum += filters[tempIndex3 + maxFilterWidthIndex - v] * cache[tempIndex2 + v];
				//TEST------------
				//printf("Cache: %f \n", cache[tempIndex2 + v]);
				//TEST------------
			}
		}
	}
#else
	int filterCounter = 0;
	for (int i = INPUT_UNIT_OFFSET; i < INPUT_UNIT_LIMIT; i++)
	{
		tempIndex = INPUT_UNIT_ELEMENT_COUNT_INC_PADDING * i + tempYIndex;
		tempIndex4 = FILTER_UNIT_ELEMENT_COUNT_INC_PADDING * filterCounter;
		filterCounter++;
		for (int u = 0; u < FILTER_HEIGHT; u++)
		{
			tempIndex2 = tempIndex + INPUT_STRIDE * u;
			tempIndex3 = FILTER_WIDTH * (maxFilterHeightIndex - u) + tempIndex4;
			for (int v = 0; v < FILTER_WIDTH; v++)
			{
				sum += filters[tempIndex3 + maxFilterWidthIndex - v] * inputDelta[tempIndex2 + v];
			}
		}
	}
#endif

	output[xIndex + OUTPUT_WIDTH_OFFSET + OUTPUT_STRIDE * (yIndex + OUTPUT_HEIGHT_OFFSET)] = sum;
}
