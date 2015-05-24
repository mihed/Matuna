//This kernel is back propagation for a convolution layer that is fully connected.
//Futhermore, it is assumed that the inputDelta has zero padding / offset around itself with FILTER_WIDTH - 1 / FILTER_HEIGHT -1 size.

#ifndef FILTER_WIDTH
#define FILTER_WIDTH -1
#endif

#ifndef FILTER_HEIGHT
#define FILTER_HEIGHT -1
#endif

#ifndef INPUT_UNIT_OFFSET
#define INPUT_UNIT_OFFSET -1
#endif

#ifndef INPUT_UNIT_LIMIT
#define INPUT_UNIT_LIMIT -1
#endif

#ifndef INPUT_UNIT_COUNT
#define INPUT_UNIT_COUNT -1
#endif

#ifndef INPUT_STRIDE
#define INPUT_STRIDE -1
#endif

#ifndef OUTPUT_STRIDE
#define OUTPUT_STRIDE -1
#endif

//Observe that this offset is offset to the global id
#ifndef INPUT_WIDTH_OFFSET
#define INPUT_WIDTH_OFFSET -1
#endif

//Observe that this offset is offset to the global id
#ifndef INPUT_HEIGHT_OFFSET
#define INPUT_HEIGHT_OFFSET -1
#endif

#ifndef OUTPUT_WIDTH_OFFSET
#define OUTPUT_WIDTH_OFFSET -1
#endif

#ifndef OUTPUT_HEIGHT_OFFSET
#define OUTPUT_HEIGHT_OFFSET -1
#endif

//Width * Height
#ifndef INPUT_UNIT_ELEMENT_COUNT_INC_PADDING 
#define INPUT_UNIT_ELEMENT_COUNT_INC_PADDING -1
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

typedef double TYPE;
#else

typedef float TYPE;
#endif

__kernel void BackPropConvolutionKernel(
	#ifdef CONSTANT_INPUT
	    __constant TYPE* inputDelta,
    #else
	    __global const TYPE* inputDelta,
    #endif
    
	__global TYPE* output,

    #ifdef USE_LOCAL_MEMORY
        #ifdef CONSTANT_FILTERS
            __constant TYPE* filters,    
        #else
            __global const TYPE* filters,
        #endif
        __local TYPE* cache
    #else
        #ifdef CONSTANT_FILTERS
            __constant TYPE* filters    
        #else
            __global const TYPE* filters
        #endif
    #endif
    )
{
    const int xIndex = get_global_id(0);
    const int yIndex = get_global_id(1);
    
    const int tempYIndex = (yIndex + INPUT_HEIGHT_OFFSET) * INPUT_STRIDE + xIndex + INPUT_WIDTH_OFFSET;
    const int maxFilterHeightIndex = FILTER_HEIGHT - 1;
    const int maxFilterWidthIndex = FILTER_WIDTH - 1;
   
    
    #ifdef USE_LOCAL_MEMORY
    const int xIndexLocal = get_local_id(0);
    const int yIndexLocal = get_local_id(1);
    const int xMaxIndexLocal = get_local_size(0) - 1;
    const int yMaxIndexLocal = get_local_size(1) - 1;
    const int localCacheWidth =  get_local_size(0) + FILTER_WIDTH - 1;
    const int localIndex = xIndexLocal + localCacheWidth * yIndexLocal;
    const int localElementCount = localCacheWidth * (get_local_size(1)  + FILTER_HEIGHT - 1);

    int localTemp1;
    int globalTemp1;
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
                localTemp1 = localTemp1 + localIndex + u * localCacheWidth;
                globalTemp1 = globalTemp1 + u * INPUT_STRIDE;
                for (int v = 0; v < FILTER_WIDTH; v++)
                {
                    cache[localTemp1 + v] = inputDelta[globalTemp1 + v];
                }
            }
        }
        else if (xIndexLocal == xMaxIndexLocal)
        {
            for (int v = 0; v < FILTER_WIDTH; v++)
            {
                cache[localTemp1 + v] = inputDelta[globalTemp1 + v];
            }
        }
        else if (yIndexLocal == yMaxIndexLocal)
        {
            for (int u = 0; u < FILTER_HEIGHT; u++)
            {
                cache[localTemp1 + u * localCacheWidth] = inputDelta[globalTemp1 + u * INPUT_STRIDE];
            }
        }
        else
        {
            cache[localTemp1] = inputDelta[globalTemp1];
        }  
    }
    
     barrier(CLK_LOCAL_MEM_FENCE);
     
    #endif
    
    int tempIndex;
    int tempIndex2;
    int tempIndex3;
    int tempIndex4;
    int sum = 0;
    
    
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