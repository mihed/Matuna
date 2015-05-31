#ifndef INPUT_DELTA_STRIDE
#define INPUT_DELTA_STRIDE -1
#endif

#ifndef OUTPUT_STRIDE
#define OUTPUT_STRIDE -1
#endif

#ifndef INPUT_STRIDE
#define INPUT_STRIDE -1
#endif

#ifndef INPUT_WIDTH_OFFSET
#define INPUT_WIDTH_OFFSET -1
#endif

#ifndef INPUT_HEIGHT_OFFSET
#define INPUT_HEIGHT_OFFSET -1
#endif

#ifndef INPUT_DELTA_WIDTH_OFFSET
#define INPUT_DELTA_WIDTH_OFFSET -1
#endif

#ifndef INPUT_DELTA_HEIGHT_OFFSET
#define INPUT_DELTA_HEIGHT_OFFSET -1
#endif

#ifndef INPUT_DELTA_UNIT_OFFSET
#define INPUT_DELTA_UNIT_OFFSET -1
#endif

#ifndef WIDTH_LIMIT
#define WIDTH_LIMIT -1
#endif

#ifndef HEIGHT_LIMIT
#define HEIGHT_LIMIT -1
#endif

//Width * Height
#ifndef OUTPUT_UNIT_ELEMENT_COUNT_INC_PADDING 
#define OUTPUT_UNIT_ELEMENT_COUNT_INC_PADDING -1
#endif

//Width * Height
#ifndef INPUT_DELTA_UNIT_ELEMENT_COUNT_INC_PADDING 
#define INPUT_DELTA_UNIT_ELEMENT_COUNT_INC_PADDING -1
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

__kernel void MultiplyWithOffsetKernel(
	#ifdef CONSTANT_INPUT
	    __constant TYPE* input,
    #else
	    __global const TYPE* input,
    #endif
    #ifdef CONSTANT_INPUT_DELTA
	    __constant TYPE* inputDelta,
    #else
	    __global const TYPE* inputDelta,
    #endif
  
    __global TYPE* output
)
{
    
    const int xIndex = get_global_id(0);
    const int yIndex = get_global_id(1);
    const int zIndex = get_global_id(2);
    
    const int outputIndex = xIndex + OUTPUT_WIDTH_OFFSET + (yIndex + OUTPUT_HEIGHT_OFFSET) * OUTPUT_STRIDE + (zIndex + OUTPUT_UNIT_OFFSET) *  OUTPUT_UNIT_ELEMENT_COUNT_INC_PADDING;
    
    TYPE sum = 0;
        
    const int tempDeltaIndex = INPUT_DELTA_UNIT_ELEMENT_COUNT_INC_PADDING * (zIndex + INPUT_DELTA_UNIT_OFFSET);
    int temp1;
    int temp2;
    for (int y = 0; y < HEIGHT_LIMIT; y++)
    {
        temp1 = tempDeltaIndex + INPUT_DELTA_STRIDE * (y + INPUT_DELTA_HEIGHT_OFFSET);
        temp2 = INPUT_STRIDE * (y + INPUT_HEIGHT_OFFSET);
        for (int x = 0; x < WIDTH_LIMIT; x++)
        {
            sum += inputDelta[temp1 + INPUT_DELTA_WIDTH_OFFSET + x] * input[temp2 + INPUT_WIDTH_OFFSET + y];
        }
    }
    
    output[outputIndex] = sum;
}