/**
*Macros to define:
* - INPUT_COUNT: The amount of input units
* - DOUBLE_PRECISION: If the kernel is to be executed with double precision
* - CONSTANT_INPUT: If we may put the inputs into the constant memory space
* - CONSTANT_WEIGHTS: If we may put the weights into the constant memory space
* - CONSTANT_BIASES: If we may put the biases into the constant memory space
* - SIGMOID: If we are using sigmoid activation
* - TANH: If we are using tanh activation
* - HALF_MATH: If we use half precision math
* - NATIVE_MATH: If we use native precision math
*/

#ifndef INPUT_COUNT
#define INPUT_COUNT -1
#endif

#ifdef DOUBLE_PRECISION
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
    __constant float* input,
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
    const int rowIndex = INPUT_COUNT * outputIndex;
    
    TYPE sum = 0;
    for (int i = 0; i < INPUT_COUNT; i++)
    {
        sum += input[i] * weights[i + rowIndex];    
    }

#if defined(SIGMOID)
    #if defined(HALF_MATH)
        output[outputIndex] = ONE / (ONE + half_exp(-(sum + biases[outputIndex])));
    #elif defined(NATIVE_MATH)
        output[outputIndex] = ONE / (ONE + native_exp(-(sum + biases[outputIndex])));
    #else
        output[outputIndex] = ONE / (ONE + exp(-(sum + biases[outputIndex])));
    #endif
#elif defined(TANH)
    output[outputIndex] = TANH_OUTER * tanh(TANH_INNER * (sum + biases[outputIndex]));
#else
    output[outputIndex] = sum + biases[outputIndex];
#endif

}