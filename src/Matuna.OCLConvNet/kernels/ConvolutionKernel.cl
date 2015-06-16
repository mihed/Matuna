//This kernel performs convolution on a single image with multiple filters

/**
 *Macros to define:
 * - DOUBLE_PRECISION: If the kernel is to be executed with double precision
 * - CONSTANT_INPUT: If we put the input into __constant space
 * - CONSTANT_FILTERS: If we put the filters into __constant space
 * - CONSTANT_BIAS: If we put the bias into __constant space
 * - USE_LOCAL_MEMORY: If we want to send input into the local cache.
 * Useless for CPU devices but may be very good for GPU if the input doesn't fit inside inside __constant.
 * - MAX_LOCAL_WIDTH_INDEX: The maximum x index of the local work group
 * - MAX_LOCAL_HEIGHT_INDEX: The maximum y index of the local work group
 * - LOCAL_CACHE_WIDTH: The stride of the local cache when USE_LOCAL_MEMORY is used
 * - FILTER_WIDTH: The stride of the filter buffer. We assume that there's no padding in the filters.
 * - FILTER_HEIGHT: The height of a filter.
 * - INPUT_UNIT_MEMORY_WIDTH_OFFSET: The width offset of the input buffer
 * - INPUT_UNIT_MEMORY_HEIGHT_OFFSET: The height offset of the input buffer
 * - OUTPUT_UNIT_MEMORY_WIDTH_OFFSET: The width offset of the output buffer
 * - OUTPUT_UNIT_MEMORY_HEIGHT_OFFSET: The height offset of the output buffer
 * - OUTPUT_UNIT_OFFSET: The unit offset of the output buffer
 * - OUTPUT_UNIT_MEMORY_WIDTH: The stride of the output buffer
 * - INPUT_UNIT_MEMORY_WIDTH: The stride of the input buffer (Observe that there's no unit offset for this buffer, since we only support a single image for input)
 * - OUTPUT_UNIT_MEMORY_ELEMENTS: Stride * Height of the buffer
 * - FILTER_UNIT_ELEMENTS: Stride * Height of the buffer. Observe that there's no actual padding atm for the filters.
 * - SIGMOID: If we are using sigmoid activation
 * - TANH: If we are using tanh activation
 * - HALF_MATH: If we use half precision math
 * - NATIVE_MATH: If we use native precision math
 */

#include "RealType.h"
#include "ActivationFunction.h"

//TEST---------------------
//#pragma OPENCL EXTENSION cl_intel_printf : enable
//END TEST-----------------

//<!@
#define FILTER_WIDTH -1
#define FILTER_HEIGHT -1
#define INPUT_UNIT_MEMORY_WIDTH_OFFSET -1
#define INPUT_UNIT_MEMORY_HEIGHT_OFFSET -1
#define OUTPUT_UNIT_MEMORY_WIDTH_OFFSET -1
#define OUTPUT_UNIT_MEMORY_HEIGHT_OFFSET -1
#define OUTPUT_UNIT_OFFSET -1
#define OUTPUT_UNIT_MEMORY_WIDTH -1
#define INPUT_UNIT_MEMORY_WIDTH -1
#define OUTPUT_UNIT_MEMORY_ELEMENTS -1
#define FILTER_UNIT_ELEMENTS -1
//#define CONSTANT_INPUT
//#define CONSTANT_FILTERS
//#define USE_LOCAL_MEMORY
//#define CONSTANT_BIAS
//!@>

__kernel void ConvolutionKernel(
#ifdef CONSTANT_INPUT
		__constant real_t* input,
#else
		__global const real_t* input,
#endif

		__global real_t* output,

#ifdef CONSTANT_FILTERS
		__constant real_t* filters,
#else
		__global const real_t* filters,
#endif
#ifdef USE_LOCAL_MEMORY
#ifdef CONSTANT_BIAS
		__constant real_t* biases,
#else
		__global const real_t* biases,
#endif
		__local real_t* cache
#else
#ifdef CONSTANT_BIAS
		__constant real_t* biases
#else
		__global const real_t* biases
#endif
#endif
)
{
	const int xIndex = get_global_id(0);
	const int yIndex = get_global_id(1);
	const int zIndex = get_global_id(2);

	const int globalInputIndex = xIndex + INPUT_UNIT_MEMORY_WIDTH_OFFSET + INPUT_UNIT_MEMORY_WIDTH * (yIndex + INPUT_UNIT_MEMORY_HEIGHT_OFFSET);

#ifdef USE_LOCAL_MEMORY
	const int xIndexLocal = get_local_id(0);
	const int yIndexLocal = get_local_id(1);

	//The below three variables cannot be set by macros since we need to create the kernel
	//before we can query the available local work size.
	const int xMaxIndexLocal = get_local_size(0) - 1;
	const int yMaxIndexLocal = get_local_size(1) - 1;
	const int localCacheWidth = get_local_size(0) + FILTER_WIDTH - 1;

	const int localIndex = xIndexLocal + localCacheWidth * yIndexLocal;

	int localTemp1;
	int globalTemp1;
	if (xIndexLocal == xMaxIndexLocal && yIndexLocal == yMaxIndexLocal)
	{
		for (int i = 0; i < FILTER_HEIGHT; i++)
		{
			localTemp1 = localIndex + i * localCacheWidth;
			globalTemp1 = globalInputIndex + i * INPUT_UNIT_MEMORY_WIDTH;
			for (int j = 0; j < FILTER_WIDTH; j++)
			{
				cache[localTemp1 + j] = input[globalTemp1 + j];
			}
		}
	}
	else if (xIndexLocal == xMaxIndexLocal)
	{
		for (int i = 0; i < FILTER_WIDTH; i++)
		{
			cache[localIndex + i] = input[globalInputIndex + i];
		}
	}
	else if (yIndexLocal == yMaxIndexLocal)
	{
		for (int i = 0; i < FILTER_HEIGHT; i++)
		{
			cache[localIndex + i * localCacheWidth] = input[globalInputIndex + i * INPUT_UNIT_MEMORY_WIDTH];
		}
	}
	else
	{
		cache[localIndex] = input[globalInputIndex];
	}

	barrier(CLK_LOCAL_MEM_FENCE);

#endif

	//TEST------------------------------------------------
	//printf("Input: %f \n", input[globalInputIndex]);
	//END TEST------------------------------------------------
	const int filterZCache = FILTER_UNIT_ELEMENTS * zIndex;

	real_t sum = 0;

#ifdef USE_LOCAL_MEMORY
	int localTemp;
	int filterTemp;
	for(int i = 0; i < FILTER_HEIGHT; i++)
	{
		filterTemp = filterZCache + FILTER_WIDTH * i;
		localTemp = localIndex + localCacheWidth * i;
		for (int j = 0; j < FILTER_WIDTH; j++)
		{
			sum += cache[localTemp + j] * filters[filterTemp + j];
		}
	}
#else
	int inputTemp;
	int filterTemp;
	for(int i = 0; i < FILTER_HEIGHT; i++)
	{
		filterTemp = filterZCache + FILTER_WIDTH * i;
		inputTemp = globalInputIndex + INPUT_UNIT_MEMORY_WIDTH * i;
		for (int j = 0; j < FILTER_WIDTH; j++)
		{
			sum += input[inputTemp + j] * filters[filterTemp + j];
			//TEST------------------------------------------------
			//printf("Filter: %f \n", filters[filterTemp + j]);
			//END TEST------------------------------------------------
		}
	}
#endif

	const int outputIndex = xIndex + OUTPUT_UNIT_MEMORY_WIDTH_OFFSET + OUTPUT_UNIT_MEMORY_WIDTH * (OUTPUT_UNIT_MEMORY_HEIGHT_OFFSET + yIndex) + OUTPUT_UNIT_MEMORY_ELEMENTS * (zIndex + OUTPUT_UNIT_OFFSET);
	//TEST------------------------------------------------
	//printf("Bias: %f \n", biases[zIndex]);
	//END TEST------------------------------------------------

	const real_t biasSum = sum + biases[zIndex];
	output[outputIndex] = ACTIVATION(biasSum);

	//TEST---------------------
	//printf("Output(%i, %i, %i): %f \n", xIndex, yIndex, zIndex, output[outputIndex]);
	//END TEST-----------------

}
