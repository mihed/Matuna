#ifndef INPUT_STRIDE
#define INPUT_STRIDE -1
#endif

#ifndef INPUT_WIDTH_OFFSET
#define INPUT_WIDTH_OFFSET -1
#endif

#ifndef WIDTH_LIMIT
#define WIDTH_LIMIT -1
#endif

#ifndef HEIGHT_LIMIT
#define HEIGHT_LIMIT -1
#endif

#ifndef INPUT_HEIGHT_OFFSET
#define INPUT_HEIGHT_OFFSET -1
#endif

#ifndef INPUT_UNIT_OFFSET
#define INPUT_UNIT_OFFSET -1
#endif

//Width * Height
#ifndef INPUT_UNIT_ELEMENT_COUNT_INC_PADDING 
#define INPUT_UNIT_ELEMENT_COUNT_INC_PADDING -1
#endif

#ifndef OUTPUT_OFFSET
#define OUTPUT_OFFSET -1
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

__kernel void SumUnitKernel(
	#ifdef CONSTANT_INPUT
	    __constant TYPE* input,
    #else
	    __global const TYPE* input,
    #endif
    
    __global TYPE* output
)
{
    const int index = get_global_id(0);
    TYPE sum = 0;
    const int tempIndex = INPUT_UNIT_ELEMENT_COUNT_INC_PADDING * (index + INPUT_UNIT_OFFSET);
    int temp2;
    for (int i = INPUT_HEIGHT_OFFSET; i < HEIGHT_LIMIT; i++)
    {
        temp2 = INPUT_STRIDE * i + tempIndex;
        for (int j = INPUT_WIDTH_OFFSET; j < WIDTH_LIMIT; j++)
        {
            sum += input[j + temp2];
        }
    }
    
    output[index + OUTPUT_OFFSET] = sum;
}