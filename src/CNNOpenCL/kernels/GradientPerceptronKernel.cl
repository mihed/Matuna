/**
*Macros to define:
* - CONSTANT_INPUT: If we may put the inputs into the constant memory space
* - CONSTANT_INPUT_DELTA: If the delta is to be put into constant memory space
* - INPUT_OFFSET: The unit offset of the input
* - INPUT_DELTA_OFFSET: The unit offset of the input delta
* - WEIGHT_COLUMN_COUNT: The column dimension of the matrix
* - DOUBLE_PRECISION: if double precision is to be used
*/

#ifndef INPUT_OFFSET
#define INPUT_OFFSET 0
#endif

#ifndef INPUT_DELTA_OFFSET
#define INPUT_DELTA_OFFSET 0
#endif

#ifndef WEIGHT_COLUMN_COUNT
#define WEIGHT_COLUMN_COUNT -1
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

__kernel void GradientPerceptronKernel(
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

	__global TYPE* outputGradient
)
{
    const int xIndex = get_global_id(0);
    const int yIndex = get_global_id(1);
    
    outputGradient[yIndex * WEIGHT_COLUMN_COUNT + xIndex] = inputDelta[yIndex + INPUT_DELTA_OFFSET] * input[xIndex + INPUT_OFFSET];
}