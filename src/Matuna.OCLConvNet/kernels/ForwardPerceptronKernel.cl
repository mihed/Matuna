/**
 *Macros to define:
 * - INPUT_COUNT: The amount of input units
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

#include "RealType.h"
#include "ActivationFunction.h"

#ifndef INPUT_COUNT
#define INPUT_COUNT -1
#endif

__kernel void ForwardPerceptronKernel(
#ifdef CONSTANT_INPUT
		__constant real_t* input,
#else
		__global const real_t* input,
#endif

		__global real_t* output,

#ifdef CONSTANT_WEIGHTS
		__constant real_t* weights,
#else
		__global const real_t* weights,
#endif

#ifdef CONSTANT_BIASES
		__constant real_t* biases
#else
		__global const real_t* biases
#endif
)
{

	//TEST----------------------
	/*printf("Input count: %i \n", INPUT_COUNT);
	 #ifdef CONSTANT_INPUT
	 printf("Using constant input\n");
	 #else
	 printf("Using global input\n");
	 #endif

	 #ifdef CONSTANT_WEIGHTS
	 printf("Using constant weights\n");
	 #else
	 printf("Using global weights\n");
	 #endif

	 #ifdef CONSTANT_BIASES
	 printf("Using constant biases\n");
	 #else
	 printf("Using global biases\n");
	 #endif

	 #if defined(HALF_MATH)
	 printf("Using half math\n");
	 #elif defined(NATIVE_MATH)
	 printf("Using native math\n");
	 #else
	 printf("Using standard math\n");
	 #endif

	 #ifdef DOUBLE_PRECISION
	 printf("Using double\n");
	 #else
	 printf("Using single\n");
	 #endif

	 #if defined(SIGMOID)
	 printf("Using sigmoid\n");
	 #elif defined(TANH)
	 printf("Using tanh\n");
	 #else
	 printf("Using linear\n");
	 #endif*/
	//END-----------------------
	const int outputIndex = get_global_id(0);
	const int rowIndex = INPUT_COUNT * outputIndex;

	real_t sum = 0;
	for (int i = 0; i < INPUT_COUNT; i++)
	{
		//TEST-----------------------
		//printf("Weight(%i,%i): %f \n", outputIndex, i, weights[i + rowIndex]);
		//printf("Input(%i): %f \n", i, input[i]);
		//END-----------------------
		sum = sum + input[i] * weights[i + rowIndex];
	}

	//TEST-----------------------
	//printf("Sum before activation (index : %i): %f \n", outputIndex, sum);
	//printf("Bias(%i): %f \n", outputIndex, biases[outputIndex]);
	//END-----------------------

	const real_t biasSum = sum + biases[outputIndex];
	output[outputIndex] = ACTIVATION(biasSum);
	//TEST-----------------------
	//printf("Result (index %i): %f \n", outputIndex, output[outputIndex]);
	//END-----------------------
}
