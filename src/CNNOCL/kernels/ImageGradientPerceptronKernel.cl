/**
*Macros to define:
* - CONSTANT_INPUT: If we may put the inputs into the constant memory space
* - CONSTANT_INPUT_DELTA: If the delta is to be put into constant memory space
* - INPUT_OFFSET: The unit offset of the input
* - INPUT_DELTA_OFFSET: The unit offset of the input delta
* - WEIGHT_COLUMN_COUNT: The column dimension of the matrix
* - DOUBLE_PRECISION: if double precision is to be used
*/

#ifndef INPUT_DATA_WIDTH
#define INPUT_DATA_WIDTH -1
#endif

#ifndef INPUT_UNIT_ELEMENT_COUNT 
#define INPUT_UNIT_ELEMENT_COUNT -1
#endif

#ifndef INPUT_WIDTH_OFFSET
#define INPUT_WIDTH_OFFSET -1
#endif

#ifndef INPUT_HEIGHT_OFFSET
#define INPUT_HEIGHT_OFFSET -1
#endif

#ifndef INPUT_UNIT_OFFSET
#define INPUT_UNIT_OFFSET -1
#endif

#ifndef INPUT_STRIDE
#define INPUT_STRIDE -1
#endif

#ifndef INPUT_UNIT_ELEMENT_COUNT_INC_PADDING 
#define INPUT_UNIT_ELEMENT_COUNT_INC_PADDING -1
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

__kernel void ImageGradientPerceptronKernel(
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
    const TYPE realValue = (TYPE)xIndex;
    
    const int zIndexInputData = (int)(floor(realValue /  INPUT_UNIT_ELEMENT_COUNT));
    const int temp = zIndexInputData * INPUT_UNIT_ELEMENT_COUNT;
    const int yIndexInputData = (int)(floor((realValue - temp) / INPUT_DATA_WIDTH));
    const int xIndexInputData = xIndex - temp - yIndexInputData * INPUT_DATA_WIDTH;
    
    const int inputIndex = xIndexInputData + INPUT_WIDTH_OFFSET + INPUT_STRIDE * (INPUT_HEIGHT_OFFSET + yIndexInputData) + INPUT_UNIT_ELEMENT_COUNT_INC_PADDING * (INPUT_UNIT_OFFSET + zIndexInputData);
    
    outputGradient[yIndex * WEIGHT_COLUMN_COUNT + xIndex] = inputDelta[yIndex + INPUT_DELTA_OFFSET] * input[inputIndex];
}