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

#ifndef MAX_LOCAL_WIDTH_INDEX
#define MAX_LOCAL_WIDTH_INDEX -1
#endif

#ifndef MAX_LOCAL_HEIGHT_INDEX
#define MAX_LOCAL_HEIGHT_INDEX -1
#endif

#ifndef LOCAL_CACHE_WIDTH
#define LOCAL_CACHE_WIDTH -1
#endif

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

#ifdef DOUBLE_PRECISION

#if defined(cl_khr_fp64)  // Khronos extension available?
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
#elif defined(cl_amd_fp64)  // AMD extension available?
#pragma OPENCL EXTENSION cl_amd_fp64 : enable
#else
#error "Double precision floating point not supported by OpenCL implementation."
#endif

#define ONE 1.0
#define TANH_OUTER 1.7159
#define TANH_INNER 0.666666666666666
typedef double TYPE;
#else
#define ONE 1.0f
#define TANH_OUTER 1.7159f
#define TANH_INNER 0.6666666f
typedef float TYPE;
#endif

__kernel void ConvolutionKernel(
    #ifdef USE_LOCAL_MEMORY
    __local TYPE* cache,
    #endif
	#ifdef CONSTANT_INPUT
	    __constant TYPE* input,
    #else
	    __global const TYPE* input,
    #endif
    
	__global TYPE* output,

    #ifdef CONSTANT_FILTERS
        __constant TYPE* filters,    
    #else
        __global const TYPE* filters,
    #endif
    #ifdef CONSTANT_BIAS
        __constant TYPE* biases    
    #else
        __global const TYPE* biases
    #endif
    )
{
    const int xIndex = get_global_id(0);
    const int yIndex = get_global_id(1);
    const int zIndex = get_global_id(2);
    
    const int globalInputIndex = xIndex + INPUT_OFFSET_WIDTH + INPUT_WIDTH * (yIndex + INPUT_OFFSET_HEIGHT);
    
    #ifdef USE_LOCAL_MEMORY
        const int xIndexLocal = get_local_id(0);
        const int yIndexLocal = get_local_id(1);
        
        const int localIndex = xIndexLocal + LOCAL_CACHE_WIDTH * yIndexLocal; 

        if (xIndexLocal == MAX_LOCAL_WIDTH_INDEX && yIndexLocal == MAX_LOCAL_HEIGHT_INDEX)
        {
            for (int i = 0; i < FILTER_HEIGHT; i++)
            {
                for (int j = 0; j < FILTER_WIDTH; j++)
                {
                    cache[localIndex + i * LOCAL_CACHE_WIDTH + j] = input[globalInputIndex + i * INPUT_WIDTH + j];
                }
            }
        }
        else if (xIndexLocal == MAX_LOCAL_WIDTH_INDEX)
        {
            for (int i = 0; i < FILTER_WIDTH; i++)
            {
                cache[localIndex + i] = input[globalInputIndex + i];
            }
        }
        else if (yIndexLocal == MAX_LOCAL_HEIGHT_INDEX)
        {
            for (int i = 0; i < FILTER_HEIGHT; i++)
            {
                cache[localIndex + i * LOCAL_CACHE_WIDTH] = input[globalInputIndex + i * INPUT_WIDTH];
            }
        }
        else
        {
            cache[localIndex] = input[globalInputIndex];
        }
        
        barrier(CLK_LOCAL_MEM_FENCE);
        
    #endif
    
    const int filterZCache = FILTER_UNIT_ELEMENT_COUNT_INC_PADDING * zIndex;
    
    TYPE sum = 0;
    
    #ifdef USE_LOCAL_MEMORY
    int localTemp;
    int filterTemp;
    for(int i = 0; i < FILTER_HEIGHT; i++)
    {
        filterTemp = filterZCache + FILTER_WIDTH * i;
        localTemp = localIndex + LOCAL_CACHE_WIDTH * i;
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
        }
    }
    #endif
    
    const int outputIndex = xIndex + OUTPUT_OFFSET_WIDTH + OUTPUT_WIDTH * (OUTPUT_OFFSET_HEIGHT + yIndex) + OUTPUT_UNIT_ELEMENT_COUNT_INC_PADDING * (zIndex + OUTPUT_OFFSET_UNIT);
    
#if defined(SIGMOID)
    #ifndef DOUBLE_PRECISION
        #if defined(HALF_MATH)
	        output[outputIndex] = ONE / (ONE + half_exp(-(sum + biases[zIndex])));
        #elif defined(NATIVE_MATH)
	        output[outputIndex] = ONE / (ONE + native_exp(-(sum + biases[zIndex])));
        #else
	        output[outputIndex] = ONE / (ONE + exp(-(sum + biases[zIndex])));
        #endif
    #else
	    output[outputIndex] = ONE / (ONE + exp(-(sum + biases[zIndex])));
    #endif
#elif defined(TANH)
	output[outputIndex] = TANH_OUTER * tanh(TANH_INNER * (sum + biases[zIndex]));
#else
	output[outputIndex] = sum + biases[zIndex];
#endif
    
}