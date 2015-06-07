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
 * - INPUT_OFFSET_WIDTH: The width offset of the input buffer
 * - INPUT_OFFSET_HEIGHT: The height offset of the input buffer
 * - OUTPUT_OFFSET_WIDTH: The width offset of the output buffer
 * - OUTPUT_OFFSET_HEIGHT: The height offset of the output buffer
 * - OUTPUT_OFFSET_UNIT: The unit offset of the output buffer
 * - OUTPUT_WIDTH: The stride of the output buffer
 * - INPUT_WIDTH: The stride of the input buffer (Observe that there's no unit offset for this buffer, since we only support a single image for input)
 * - OUTPUT_UNIT_ELEMENT_COUNT_INC_PADDING: Stride * Height of the buffer
 * - FILTER_UNIT_ELEMENT_COUNT_INC_PADDING: Stride * Height of the buffer. Observe that there's no actual padding atm for the filters.
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

#ifndef FILTER_WIDTH
#define FILTER_WIDTH -1
#endif

#ifndef FILTER_HEIGHT
#define FILTER_HEIGHT -1
#endif

#ifndef INPUT_OFFSET_WIDTH
#define INPUT_OFFSET_WIDTH -1
#endif

#ifndef INPUT_OFFSET_HEIGHT
#define INPUT_OFFSET_HEIGHT -1
#endif

#ifndef OUTPUT_OFFSET_WIDTH
#define OUTPUT_OFFSET_WIDTH -1
#endif

#ifndef OUTPUT_OFFSET_HEIGHT
#define OUTPUT_OFFSET_HEIGHT -1
#endif

#ifndef OUTPUT_OFFSET_UNIT
#define OUTPUT_OFFSET_UNIT -1
#endif

#ifndef OUTPUT_WIDTH
#define OUTPUT_WIDTH -1
#endif

#ifndef INPUT_WIDTH
#define INPUT_WIDTH -1
#endif

//Width * Height
#ifndef OUTPUT_UNIT_ELEMENT_COUNT_INC_PADDING 
#define OUTPUT_UNIT_ELEMENT_COUNT_INC_PADDING -1
#endif

//Width * Height
#ifndef FILTER_UNIT_ELEMENT_COUNT_INC_PADDING 
#define FILTER_UNIT_ELEMENT_COUNT_INC_PADDING -1
#endif

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

	//TEST------------------------------------------------
	/*
	 if (get_global_id(0) == 0 && get_global_id(1) == 0 && get_global_id(2) == 0)
	 {
	 printf("FILTER_WIDTH: %i \n", FILTER_WIDTH);
	 printf("FILTER_HEIGHT: %i \n", FILTER_HEIGHT);
	 printf("INPUT_OFFSET_WIDTH: %i \n", INPUT_OFFSET_WIDTH);
	 printf("INPUT_OFFSET_HEIGHT: %i \n", INPUT_OFFSET_HEIGHT);
	 printf("OUTPUT_OFFSET_WIDTH: %i \n", OUTPUT_OFFSET_WIDTH);
	 printf("OUTPUT_OFFSET_HEIGHT: %i \n", OUTPUT_OFFSET_HEIGHT);
	 printf("OUTPUT_OFFSET_UNIT: %i \n", OUTPUT_OFFSET_UNIT);
	 printf("OUTPUT_WIDTH: %i \n", OUTPUT_WIDTH);
	 printf("INPUT_WIDTH: %i \n", INPUT_WIDTH);
	 printf("OUTPUT_UNIT_ELEMENT_COUNT_INC_PADDING : %i \n", OUTPUT_UNIT_ELEMENT_COUNT_INC_PADDING );
	 printf("FILTER_UNIT_ELEMENT_COUNT_INC_PADDING : %i \n", FILTER_UNIT_ELEMENT_COUNT_INC_PADDING );
	 #if defined(SIGMOID)
	 printf("Using sigmoid \n");
	 #elif defined(TANH)
	 printf("Using tanh \n");
	 #else
	 printf("Using linear \n");
	 #endif
	 #if defined(HALF_MATH)
	 printf("Using half math \n");
	 #elif defined(NATIVE_MATH)
	 printf("Using native math \n");
	 #else
	 printf("Using normal math \n");
	 #endif

	 #ifdef CONSTANT_INPUT
	 printf("Constant input \n");
	 #endif
	 #ifdef CONSTANT_BIAS
	 printf("Constant bias \n");
	 #endif
	 #ifdef CONSTANT_FILTERS
	 printf("Constant filters \n");
	 #endif
	 #ifdef USE_LOCAL_MEMORY
	 printf("Using local memory \n");
	 #endif
	 }
	 */
	//END TEST------------------------------------------------
	const int globalInputIndex = xIndex + INPUT_OFFSET_WIDTH + INPUT_WIDTH * (yIndex + INPUT_OFFSET_HEIGHT);

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
			globalTemp1 = globalInputIndex + i * INPUT_WIDTH;
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
			cache[localIndex + i * localCacheWidth] = input[globalInputIndex + i * INPUT_WIDTH];
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
	const int filterZCache = FILTER_UNIT_ELEMENT_COUNT_INC_PADDING * zIndex;

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
		inputTemp = globalInputIndex + INPUT_WIDTH * i;
		for (int j = 0; j < FILTER_WIDTH; j++)
		{
			sum += input[inputTemp + j] * filters[filterTemp + j];
			//TEST------------------------------------------------
			//printf("Filter: %f \n", filters[filterTemp + j]);
			//END TEST------------------------------------------------
		}
	}
#endif

	const int outputIndex = xIndex + OUTPUT_OFFSET_WIDTH + OUTPUT_WIDTH * (OUTPUT_OFFSET_HEIGHT + yIndex) + OUTPUT_UNIT_ELEMENT_COUNT_INC_PADDING * (zIndex + OUTPUT_OFFSET_UNIT);
	//TEST------------------------------------------------
	//printf("Bias: %f \n", biases[zIndex]);
	//END TEST------------------------------------------------

	const real_t biasSum = sum + biases[zIndex];
	output[outputIndex] = ACTIVATION(biasSum);

	//TEST---------------------
	//printf("Output(%i, %i, %i): %f \n", xIndex, yIndex, zIndex, output[outputIndex]);
	//END TEST-----------------

}
