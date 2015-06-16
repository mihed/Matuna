// Furthermore it's important to note that the targets and input have the same memory description by definition of the network.

/**
 * Macros to define:
 * - MATUNA_DIFFERENCE:         The back prop will use a simple difference between the target and the input
 * - MATUNA_MSE_ANY:            The back prop will use the MATUNA_MSE together with any back-prop activation function.
 * - MATUNA_CE_ANY:             The back prop will use the MATUNA_CE together with any back-prop activation function.
 * - MATUNA_CE_BINARY_ANY:      The back prop will use the MATUNA_CE binary together with any back-prop activation function.
 * - DOUBLE_PRECISION:   If the kernel is to be executed with double precision.
 * - CONSTANT_INPUT:     If we may put the inputs into the constant memory space.
 * - CONSTANT_TARGET:    If we may put the targets into the constant memory space.
 * - SIGMOID:            If we are using sigmoid back-prop activation
 * - TANH:               If we are using tanh back-prop activation
 * - INPUT_UNIT_OFFSET:  The offset of the input / target memory
 * - OUTPUT_UNIT_OFFSET: The offset of the output memory
 */

#include "RealType.h"
#include "ActivationFunction.h"

//<!@
#define INPUT_UNIT_MEMORY_WIDTH_OFFSET -1
#define INPUT_UNIT_MEMORY_HEIGHT_OFFSET -1
#define OUTPUT_UNIT_MEMORY_WIDTH_OFFSET -1
#define OUTPUT_UNIT_MEMORY_HEIGHT_OFFSET -1
#define INPUT_UNIT_OFFSET -1
#define OUTPUT_UNIT_OFFSET -1
#define INPUT_UNIT_MEMORY_WIDTH -1
#define OUTPUT_UNIT_MEMORY_WIDTH -1
#define OUTPUT_UNIT_MEMORY_ELEMENTS -1
#define INPUT_UNIT_MEMORY_ELEMENTS -1
//#define MATUNA_DIFFERENCE
//#define MATUNA_MSE_ANY
//#define MATUNA_CE_ANY
//#define MATUNA_CE_BINARY_ANY
//#define CONSTANT_INPUT
//#define CONSTANT_TARGET
//!@>

#if defined(MATUNA_DIFFERENCE)

__kernel void BackPropagation(
#ifdef CONSTANT_INPUT
		__constant real_t* input,
#else
		__global const real_t* input,
#endif
#ifdef CONSTANT_TARGET
		__constant real_t* target,
#else
		__global const real_t* target,
#endif
		__global real_t* output)
{
	const int xIndex = get_global_id(0);
	const int yIndex = get_global_id(1);
	const int zIndex = get_global_id(2);

	const int inputIndex = (zIndex + INPUT_UNIT_OFFSET)* INPUT_UNIT_MEMORY_ELEMENTS + (yIndex + INPUT_UNIT_MEMORY_HEIGHT_OFFSET) * INPUT_UNIT_MEMORY_WIDTH + xIndex + INPUT_UNIT_MEMORY_WIDTH_OFFSET;
	const int outputIndex = (zIndex + OUTPUT_UNIT_OFFSET) * OUTPUT_UNIT_MEMORY_ELEMENTS + (yIndex + OUTPUT_UNIT_MEMORY_HEIGHT_OFFSET) * OUTPUT_UNIT_MEMORY_WIDTH + xIndex + OUTPUT_UNIT_MEMORY_WIDTH_OFFSET;

	output[outputIndex] = input[inputIndex] - target[inputIndex];
}

#elif defined(MATUNA_MSE_ANY)

__kernel void BackPropagation(
#ifdef CONSTANT_INPUT
		__constant real_t* input,
#else
		__global const real_t* input,
#endif
#ifdef CONSTANT_TARGET
		__constant real_t* target,
#else
		__global const real_t* target,
#endif
		__global real_t* output)
{
	const int xIndex = get_global_id(0);
	const int yIndex = get_global_id(1);
	const int zIndex = get_global_id(2);

	const int inputIndex = (zIndex + INPUT_UNIT_OFFSET)* INPUT_UNIT_MEMORY_ELEMENTS + (yIndex + INPUT_UNIT_MEMORY_HEIGHT_OFFSET) * INPUT_UNIT_MEMORY_WIDTH + xIndex + INPUT_UNIT_MEMORY_WIDTH_OFFSET;
	const int outputIndex = (zIndex + OUTPUT_UNIT_OFFSET) * OUTPUT_UNIT_MEMORY_ELEMENTS + (yIndex + OUTPUT_UNIT_MEMORY_HEIGHT_OFFSET) * OUTPUT_UNIT_MEMORY_WIDTH + xIndex + OUTPUT_UNIT_MEMORY_WIDTH_OFFSET;

	const real_t tempInput = input[inputIndex];
	const real_t temp2 = tempInput - target[inputIndex];
	output[outputIndex] = ACTIVATION_DERIVATIVE(temp2, tempInput);

}

#elif defined(MATUNA_CE_BINARY_ANY)

__kernel void BackPropagation(
#ifdef CONSTANT_INPUT
		__constant real_t* inputs,
#else
		__global const real_t* inputs,
#endif
#ifdef CONSTANT_TARGET
		__constant real_t* target,
#else
		__global const real_t* target,
#endif
		__global real_t* output)
{
	const int xIndex = get_global_id(0);
	const int yIndex = get_global_id(1);
	const int zIndex = get_global_id(2);

	const int inputIndex = (zIndex + INPUT_UNIT_OFFSET)* INPUT_UNIT_MEMORY_ELEMENTS + (yIndex + INPUT_UNIT_MEMORY_HEIGHT_OFFSET) * INPUT_UNIT_MEMORY_WIDTH + xIndex + INPUT_UNIT_MEMORY_WIDTH_OFFSET;
	const int outputIndex = (zIndex + OUTPUT_UNIT_OFFSET) * OUTPUT_UNIT_MEMORY_ELEMENTS + (yIndex + OUTPUT_UNIT_MEMORY_HEIGHT_OFFSET) * OUTPUT_UNIT_MEMORY_WIDTH + xIndex + OUTPUT_UNIT_MEMORY_WIDTH_OFFSET;

#if defined(MATUNA_ACTIVATION_DERIVATIVE_SIGMOID)
	output[outputIndex] = inputs[inputIndex] - target[inputIndex];
#else
	const real_t input = inputs[inputIndex];
	const real_t temp2 = (input - target[inputIndex]) / (input * (ONE - input));
	output[outputIndex] =  ACTIVATION_DERIVATIVE(temp2, input);
#endif
}

#elif defined(MATUNA_CE_ANY)

__kernel void BackPropagation(
#ifdef CONSTANT_INPUT
		__constant real_t* input,
#else
		__global const real_t* input,
#endif
#ifdef CONSTANT_TARGET
		__constant real_t* target,
#else
		__global const real_t* target,
#endif
		__global real_t* output)
{
	const int xIndex = get_global_id(0);
	const int yIndex = get_global_id(1);
	const int zIndex = get_global_id(2);

	const int inputIndex = (zIndex + INPUT_UNIT_OFFSET)* INPUT_UNIT_MEMORY_ELEMENTS + (yIndex + INPUT_UNIT_MEMORY_HEIGHT_OFFSET) * INPUT_UNIT_MEMORY_WIDTH + xIndex + INPUT_UNIT_MEMORY_WIDTH_OFFSET;
	const int outputIndex = (zIndex + OUTPUT_UNIT_OFFSET) * OUTPUT_UNIT_MEMORY_ELEMENTS + (yIndex + OUTPUT_UNIT_MEMORY_HEIGHT_OFFSET) * OUTPUT_UNIT_MEMORY_WIDTH + xIndex + OUTPUT_UNIT_MEMORY_WIDTH_OFFSET;
	
#if defined(MATUNA_ACTIVATION_DERIVATIVE_SIGMOID)
	output[outputIndex] = -target[inputIndex] * (ONE - input[inputIndex]);
#else
	const real_t tempInput = input[inputIndex];
	const real_t temp2 = -target[inputIndex] / tempInput;
	output[outputIndex] = ACTIVATION_DERIVATIVE(temp2, tempInput);
#endif
}

#else
#error "There's no error function defined"
#endif
