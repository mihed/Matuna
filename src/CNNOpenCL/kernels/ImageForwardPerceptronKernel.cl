/**
*Macros to define:
* - INPUT_UNITS: The amount of input units
* - INPUT_WIDTH: The input width
* - INPUT_HEIGHT: The input height
* - INPUT_UNITS_OFFSET: The offset in the units direction
* - INPUT_WIDTH_OFFSET: The offset in the width direction
* - INPUT_HEIGHT_OFFSET: The offset in the height direction
* - COLUMN_COUNT: The amount of columns in the weight matrix
* - INPUT_UNIT_ELEMENT_COUNT_INC_PADDING: The amount of elements inside one unit with padding
* - INPUT_MEMORY_WIDTH: The width of the memory
* - DOUBLE_PRECISION: If the kernel is to be executed with double precision
* - CONSTANT_INPUT: If we may put the inputs into the constant memory space
* - CONSTANT_WEIGHTS: If we may put the weights into the constant memory space
* - CONSTANT_BIASES: If we may put the biases into the constant memory space
* - SIGMOID: If we are using sigmoid activation
* - TANH: If we are using tanh activation
* - SOFTMAX: If we are using the softmax activation
* - HALF_MATH: If we use half precision math
* - NATIVE_MATH: If we use native precision math
*/

#ifndef INPUT_UNITS
#define INPUT_UNITS -1
#endif

#ifndef INPUT_WIDTH
#define INPUT_WIDTH -1
#endif

#ifndef INPUT_HEIGHT
#define INPUT_HEIGHT -1
#endif

#ifndef INPUT_UNITS_OFFSET
#define INPUT_UNITS_OFFSET -1
#endif

#ifndef INPUT_WIDTH_OFFSET
#define INPUT_WIDTH_OFFSET -1
#endif

#ifndef INPUT_HEIGHT_OFFSET
#define INPUT_HEIGHT_OFFSET -1
#endif

#ifndef COLUMN_COUNT
#define COLUMN_COUNT -1
#endif

#ifndef INPUT_UNIT_ELEMENT_COUNT_INC_PADDING
#define INPUT_UNIT_ELEMENT_COUNT_INC_PADDING -1
#endif

#ifndef INPUT_MEMORY_WIDTH
#define INPUT_MEMORY_WIDTH -1
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

__kernel void ForwardPerceptronKernel(
#ifdef CONSTANT_INPUT
    __constant TYPE* input,
#else
    __global const TYPE* input,
#endif

    __global TYPE* output,
    
#ifdef CONSTANT_WEIGHTS
    __constant TYPE* weights,
#else
    __global const TYPE* weights,
#endif

#ifdef CONSTANT_BIASES
    __constant TYPE* biases
#else
    __global const TYPE* biases
#endif
    )
{
    const int outputIndex = get_global_id(0);
    const int rowIndex = COLUMN_COUNT * outputIndex;
    
    TYPE sum = 0;
    int columnIndex = 0;
    int tempZIndex = 0;
    int tempYIndex = 0;
    for (int unit = INPUT_UNITS_OFFSET; unit < INPUT_UNITS; unit++)
    {
        tempZIndex = unit * INPUT_UNIT_ELEMENT_COUNT_INC_PADDING;
        for (int row = INPUT_HEIGHT_OFFSET; row < INPUT_HEIGHT; row++)
        {
            tempYIndex = row * INPUT_MEMORY_WIDTH + tempZIndex;
            for(int column = INPUT_WIDTH_OFFSET; column < INPUT_WIDTH; column++)
            {
               sum += input[tempYIndex + column] * weights[rowIndex + columnIndex];
				columnIndex++;
            }
        }
    }

#if defined(SIGMOID)
    #ifndef DOUBLE_PRECISION
        #if defined(HALF_MATH)
            output[outputIndex] = ONE / (ONE + half_exp(-(sum + biases[outputIndex])));
        #elif defined(NATIVE_MATH)
            output[outputIndex] = ONE / (ONE + native_exp(-(sum + biases[outputIndex])));
        #else
            output[outputIndex] = ONE / (ONE + exp(-(sum + biases[outputIndex])));
        #endif
    #else
        output[outputIndex] = ONE / (ONE + exp(-(sum + biases[outputIndex])));
    #endif
#elif defined(TANH)
    output[outputIndex] = TANH_OUTER * tanh(TANH_INNER * (sum + biases[outputIndex]));
#elif defined(SOFTMAX)
    #ifndef DOUBLE_PRECISION
        #if defined(HALF_MATH)
	        output[outputIndex] = half_exp(sum + biases[outputIndex]);
        #elif defined(NATIVE_MATH)
	        output[outputIndex] = native_exp(sum + biases[outputIndex]);
        #else
	        output[outputIndex] = exp(sum + biases[outputIndex]);
        #endif
    #else
	    output[outputIndex] = exp(sum + biases[outputIndex]);
    #endif
#else
    output[outputIndex] = sum + biases[outputIndex];
#endif

}