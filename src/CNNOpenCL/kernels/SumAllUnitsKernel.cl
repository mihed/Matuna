/**
*Macros to define:
* - DOUBLE_PRECISION: If the kernel is to be executed with double precision
* - UNIT_COUNT_INC_PADDING: This is the upper limit when summarizing the input. This must equal the number of data units + padding
* - UNIT_INPUT_OFFSET: The memory offset of the input units
* - WIDTH_INPUT_OFFSET: The memory width offset of the input units
* - HEIGHT_INPUT_OFFSET: The memory width offset of the input units
* - WIDTH_OUTPUT_OFFSET: The memory width offset of the output units
* - HEIGHT_OUTPUT_OFFSET: The memory height offset of the output units
* - WIDTH_INPUT: The memory width of the input units
* - INPUT_UNIT_ELEMENT_COUNT_INC_PADDING: The memory count of one unit in the input
* - WIDTH_OUTPUT: The memory width of the output units.
* - CONSTANT_INPUT: If we put the input into __constant space
*/

//Offset + the data unit count.
#ifndef UNIT_COUNT_INC_PADDING
#define UNIT_COUNT_INC_PADDING -1
#endif

#ifndef UNIT_INPUT_OFFSET
#define UNIT_INPUT_OFFSET -1
#endif

#ifndef WIDTH_INPUT_OFFSET
#define WIDTH_INPUT_OFFSET -1
#endif

#ifndef HEIGHT_INPUT_OFFSET
#define HEIGHT_INPUT_OFFSET -1
#endif

#ifndef WIDTH_OUTPUT_OFFSET
#define WIDTH_OUTPUT_OFFSET -1
#endif

#ifndef HEIGHT_OUTPUT_OFFSET
#define HEIGHT_OUTPUT_OFFSET -1
#endif

#ifndef WIDTH_INPUT
#define WIDTH_INPUT -1
#endif

//Width * Height
#ifndef INPUT_UNIT_ELEMENT_COUNT_INC_PADDING 
#define INPUT_UNIT_ELEMENT_COUNT_INC_PADDING -1
#endif

#ifndef WIDTH_OUTPUT
#define WIDTH_OUTPUT -1
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


__kernel void SumAllUnitsKernel(
	#ifdef CONSTANT_INPUT
	    __constant TYPE* input,
    #else
	    __global const TYPE* input,
    #endif
	__global TYPE* output)

{
    const int xIndex = get_global_id(0);
    const int yIndex = get_global_id(1);
    
    const int temp = xIndex + WIDTH_INPUT_OFFSET + WIDTH_INPUT * (HEIGHT_INPUT_OFFSET + yIndex);
	TYPE sum = 0;
	for (int z = UNIT_INPUT_OFFSET; z < UNIT_COUNT_INC_PADDING; z++)
	{
	    sum += input[temp + INPUT_UNIT_ELEMENT_COUNT_INC_PADDING * z];
	}
    
    //We have a single unit as output
    output[WIDTH_OUTPUT_OFFSET + xIndex + WIDTH_OUTPUT * (HEIGHT_OUTPUT_OFFSET + yIndex)] = sum;
}