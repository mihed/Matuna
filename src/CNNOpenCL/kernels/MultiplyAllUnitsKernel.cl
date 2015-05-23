
#ifndef INPUT_DELTA_STRIDE
#define INPUT_DELTA_STRIDE -1
#endif

#ifndef OUTPUT_STRIDE
#define OUTPUT_STRIDE -1
#endif

#ifndef INPUT_STRIDE
#define INPUT_STRIDE -1
#endif

//Observe that this offset is offset to the global id
#ifndef INPUT_DELTA_WIDTH_OFFSET
#define INPUT_DELTA_WIDTH_OFFSET -1
#endif

//Observe that this offset is offset to the global id
#ifndef INPUT_DELTA_HEIGHT_OFFSET
#define INPUT_DELTA_HEIGHT_OFFSET -1
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

#ifndef INPUT_WIDTH_OFFSET
#define INPUT_WIDTH_OFFSET -1
#endif

#ifndef INPUT_HEIGHT_OFFSET
#define INPUT_HEIGHT_OFFSET -1
#endif

#ifndef INPUT_UNIT_OFFSET
#define INPUT_UNIT_OFFSET -1
#endif

//Width * Height
#ifndef OUTPUT_UNIT_ELEMENT_COUNT_INC_PADDING 
#define OUTPUT_UNIT_ELEMENT_COUNT_INC_PADDING -1
#endif

//Width * Height
#ifndef INPUT_UNIT_ELEMENT_COUNT_INC_PADDING 
#define INPUT_UNIT_ELEMENT_COUNT_INC_PADDING -1
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

__kernel void MultiplyAllUnitsKernel(
	#ifdef CONSTANT_INPUT
	    __constant TYPE* input,
    #else
	    __global const TYPE* input,
    #endif
    #ifdef CONSTANT_INPUT
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
   
   const int outputIndex = xIndex + OUTPUT_WIDTH_OFFSET + OUTPUT_STRIDE * (yIndex + OUTPUT_HEIGHT_OFFSET) + OUTPUT_UNIT_ELEMENT_COUNT_INC_PADDING * (zIndex + OUTPUT_UNIT_OFFSET);
   
   const TYPE tempInputDelta = inputDelta[xIndex + INPUT_DELTA_WIDTH_OFFSET + INPUT_DELTA_STRIDE * (yIndex + INPUT_DELTA_HEIGHT_OFFSET)];
    
#if defined(SIGMOID)
    const int inputIndex = xIndex + INPUT_WIDTH_OFFSET + INPUT_STRIDE * (yIndex + INPUT_HEIGHT_OFFSET) + INPUT_UNIT_ELEMENT_COUNT_INC_PADDING * (zIndex + INPUT_UNIT_OFFSET);
    const TYPE tempInput = input[inputIndex];
	output[outputIndex] = tempInputDelta  * tempInput * (ONE - tempInput);
#elif defined(TANH)
    const int inputIndex = xIndex + INPUT_WIDTH_OFFSET + INPUT_STRIDE * (yIndex + INPUT_HEIGHT_OFFSET) + INPUT_UNIT_ELEMENT_COUNT_INC_PADDING * (zIndex + INPUT_UNIT_OFFSET);
    const TYPE tempInput = input[inputIndex];
	output[outputIndex] = tempInputDelta * TANH_INNER * (TANH_OUTER - (tempInput * tempInput) / TANH_OUTER);
#else
	output[outputIndex] = tempInputDelta;
#endif 
}