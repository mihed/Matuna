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

#ifndef OUTPUT_WIDTH_OFFSET
#define OUTPUT_WIDTH_OFFSET -1
#endif

#ifndef OUTPUT_HEIGHT_OFFSET
#define OUTPUT_HEIGHT_OFFSET -1
#endif

#ifndef OUTPUT_UNIT_OFFSET
#define OUTPUT_UNIT_OFFSET -1
#endif

#ifndef OUTPUT_STRIDE
#define OUTPUT_STRIDE -1
#endif

#ifndef OUTPUT_UNIT_ELEMENT_COUNT_INC_PADDING 
#define OUTPUT_UNIT_ELEMENT_COUNT_INC_PADDING -1
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
#define INPUT_DELTA_OFFSET -1
#endif

#ifndef INPUT_DELTA_LIMIT
#define INPUT_DELTA_LIMIT -1
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
    
    const int xIndex = get_global_id(0);
    const int yIndex = get_global_id(1);
    const int zIndex = get_global_id(2);
    const int columnIndex = xIndex + get_global_size(0) * yIndex + get_global_size(0) * get_global_size(1) * zIndex;
    
    /*
    if (xIndex == 0 && yIndex == 0 && zIndex == 0)
    {
        printf("Global size 0: %i \n", get_global_size(0));
         printf("Global size 1: %i \n", get_global_size(1));
          printf("Global size 2: %i \n", get_global_size(2));
          printf(" OUTPUT_WIDTH_OFFSET: %i \n", OUTPUT_WIDTH_OFFSET);
          printf(" OUTPUT_HEIGHT_OFFSET: %i \n", OUTPUT_HEIGHT_OFFSET);
           printf(" OUTPUT_UNIT_OFFSET: %i \n", OUTPUT_UNIT_OFFSET);
            printf(" OUTPUT_STRIDE: %i \n", OUTPUT_STRIDE);
             printf(" OUTPUT_UNIT_ELEMENT_COUNT_INC_PADDING: %i \n", OUTPUT_UNIT_ELEMENT_COUNT_INC_PADDING);
              printf(" INPUT_WIDTH_OFFSET: %i \n", INPUT_WIDTH_OFFSET);
               printf(" INPUT_HEIGHT_OFFSET: %i \n", INPUT_HEIGHT_OFFSET);
                printf(" INPUT_UNIT_OFFSET: %i \n", INPUT_UNIT_OFFSET);
                 printf(" INPUT_STRIDE: %i \n", INPUT_STRIDE);
                  printf(" INPUT_UNIT_ELEMENT_COUNT_INC_PADDING: %i \n", INPUT_UNIT_ELEMENT_COUNT_INC_PADDING);
                  printf(" INPUT_DELTA_OFFSET: %i \n", INPUT_DELTA_OFFSET);
                  printf(" INPUT_DELTA_LIMIT: %i \n", INPUT_DELTA_LIMIT);
                  printf(" WEIGHT_COLUMN_COUNT: %i \n", WEIGHT_COLUMN_COUNT);
    }
    */
    
    const int outputDeltaIndex = OUTPUT_WIDTH_OFFSET + xIndex + OUTPUT_STRIDE * (OUTPUT_HEIGHT_OFFSET + yIndex) + OUTPUT_UNIT_ELEMENT_COUNT_INC_PADDING * (OUTPUT_UNIT_OFFSET + zIndex);
    const int inputIndex = INPUT_WIDTH_OFFSET + xIndex + INPUT_STRIDE * (INPUT_HEIGHT_OFFSET + yIndex) + INPUT_UNIT_ELEMENT_COUNT_INC_PADDING * (INPUT_UNIT_OFFSET + zIndex); 
    
	TYPE sum = 0;
	for (int y = INPUT_DELTA_OFFSET; y < INPUT_DELTA_LIMIT; y++)
	{
		sum += inputDelta[y] * weights[columnIndex + WEIGHT_COLUMN_COUNT * y];
	}

#if defined(SIGMOID)
	const TYPE tempInput = input[inputIndex];
	outputDelta[outputDeltaIndex] = sum  * tempInput * (ONE - tempInput);
#elif defined(TANH)
	const TYPE tempInput = input[inputIndex];
	outputDelta[outputDeltaIndex] = sum * TANH_INNER * (TANH_OUTER - (tempInput * tempInput) / TANH_OUTER);
#else
	outputDelta[outputDeltaIndex] = sum;
#endif
}