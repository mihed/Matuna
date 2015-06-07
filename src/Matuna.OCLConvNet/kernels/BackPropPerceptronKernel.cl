/**
*Macros to define:
* - INPUT_DELTA_COUNT: The amount of units of the input delta
* - DOUBLE_PRECISION: If the kernel is to be executed with double precision
* - CONSTANT_INPUT: If we may put the inputs into the constant memory space
* - CONSTANT_WEIGHTS: If we may put the weights into the constant memory space
* - CONSTANT_INPUT_DELTA: If the delta is to be put into constant memory space
* - WEIGHT_COLUMN_COUNT: The column dimension of the matrix
* - INPUT_OFFSET: The unit offset of the input
* - OUTPUT_DELTA_OFFSET: The unit offset of the delta output
* - INPUT_DELTA_OFFSET: The unit offset of the input delta
* - SIGMOID: If we are using sigmoid activation
* - TANH: If we are using tanh activation
*/

#ifndef INPUT_OFFSET
#define INPUT_OFFSET 0
#endif

#ifndef OUTPUT_DELTA_OFFSET
#define OUTPUT_DELTA_OFFSET 0
#endif

#ifndef INPUT_DELTA_OFFSET
#define INPUT_DELTA_OFFSET 0
#endif

#ifndef INPUT_DELTA_COUNT
#define INPUT_DELTA_COUNT -1
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

__kernel void BackPerceptronKernel(
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

	__global TYPE* outputDelta,

#ifdef CONSTANT_WEIGHTS
	__constant TYPE* weights
#else
	__global const TYPE* weights
#endif
	)
{
    const int columnIndex = get_global_id(0);
    
	TYPE sum = 0;
	for (int y = INPUT_DELTA_OFFSET; y < INPUT_DELTA_COUNT; y++)
	{
		sum += inputDelta[y] * weights[columnIndex + WEIGHT_COLUMN_COUNT * y];
	}

#if defined(SIGMOID)
	const TYPE tempInput = input[columnIndex + INPUT_DELTA_OFFSET];
	outputDelta[columnIndex + OUTPUT_DELTA_OFFSET] = sum  * tempInput * (ONE - tempInput);
#elif defined(TANH)
	const TYPE tempInput = input[columnIndex + INPUT_DELTA_OFFSET];
	outputDelta[columnIndex + OUTPUT_DELTA_OFFSET] = sum * TANH_INNER * (TANH_OUTER - (tempInput * tempInput) / TANH_OUTER);
#else
	outputDelta[columnIndex + OUTPUT_DELTA_OFFSET] = sum;
#endif
}