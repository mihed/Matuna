// Important: This kernel only supports outputs that are single units and not image units (TODO!!).
// Furthermore it's important to note that the targets and input have the same memory description by definition of the network.
// In this case, we only have UNIT_OFFSET that can change the layout of the memory.

/**
 * Macros to define:
 * - DIFFERENCE:         The back prop will use a simple difference between the target and the input
 * - MSE_ANY:            The back prop will use the MSE together with any back-prop activation function.
 * - CE_ANY:             The back prop will use the CE together with any back-prop activation function.
 * - CE_BINARY_ANY:      The back prop will use the CE binary together with any back-prop activation function.
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

#ifndef INPUT_OFFSET_WIDTH
#define INPUT_OFFSET_WIDTH -1
#endif

#ifndef INPUT_OFFSET_HEIGHT
#define INPUT_OFFSET_HEIGHT -1
#endif

#ifndef OUTPUT_OFFSET_WIDTH
#define OUTPUT_OFFSET_WIDTH -1
#endif

#ifndef OUTPUT_OFFSET_HEIGHT
#define OUTPUT_OFFSET_HEIGHT -1
#endif

#ifndef INPUT_UNIT_OFFSET
#define INPUT_UNIT_OFFSET -1
#endif

#ifndef INPUT_STRIDE
#define INPUT_STRIDE -1
#endif

#ifndef OUTPUT_STRIDE
#define OUTPUT_STRIDE -1
#endif

//Width * Height
#ifndef OUTPUT_UNIT_ELEMENT_COUNT_INC_PADDING 
#define OUTPUT_UNIT_ELEMENT_COUNT_INC_PADDING -1
#endif

//Width * Height
#ifndef INPUT_UNIT_ELEMENT_COUNT_INC_PADDING 
#define INPUT_UNIT_ELEMENT_COUNT_INC_PADDING -1
#endif

#ifndef OUTPUT_UNIT_OFFSET
#define OUTPUT_UNIT_OFFSET -1
#endif

#if defined(DIFFERENCE)

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

	const int inputIndex = (zIndex + INPUT_UNIT_OFFSET)* INPUT_UNIT_ELEMENT_COUNT_INC_PADDING + (yIndex + INPUT_OFFSET_HEIGHT) * INPUT_STRIDE + xIndex + INPUT_OFFSET_WIDTH;
	const int outputIndex = (zIndex + OUTPUT_UNIT_OFFSET) * OUTPUT_UNIT_ELEMENT_COUNT_INC_PADDING + (yIndex + OUTPUT_OFFSET_HEIGHT) * OUTPUT_STRIDE + xIndex + OUTPUT_OFFSET_WIDTH;

	output[outputIndex] = input[inputIndex] - target[inputIndex];
}

#elif defined(MSE_ANY)

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

	const int inputIndex = (zIndex + INPUT_UNIT_OFFSET)* INPUT_UNIT_ELEMENT_COUNT_INC_PADDING + (yIndex + INPUT_OFFSET_HEIGHT) * INPUT_STRIDE + xIndex + INPUT_OFFSET_WIDTH;
	const int outputIndex = (zIndex + OUTPUT_UNIT_OFFSET) * OUTPUT_UNIT_ELEMENT_COUNT_INC_PADDING + (yIndex + OUTPUT_OFFSET_HEIGHT) * OUTPUT_STRIDE + xIndex + OUTPUT_OFFSET_WIDTH;

	const real_t tempInput = input[inputIndex];
	const real_t temp2 = tempInput - target[inputIndex];
	output[outputIndex] = ACTIVATION_DERIVATIVE(temp2, tempInput);

}

#elif defined(CE_BINARY_ANY)

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

	const int inputIndex = (zIndex + INPUT_UNIT_OFFSET)* INPUT_UNIT_ELEMENT_COUNT_INC_PADDING + (yIndex + INPUT_OFFSET_HEIGHT) * INPUT_STRIDE + xIndex + INPUT_OFFSET_WIDTH;
	const int outputIndex = (zIndex + OUTPUT_UNIT_OFFSET) * OUTPUT_UNIT_ELEMENT_COUNT_INC_PADDING + (yIndex + OUTPUT_OFFSET_HEIGHT) * OUTPUT_STRIDE + xIndex + OUTPUT_OFFSET_WIDTH;

#if defined(SIGMOID)
	output[outputIndex] = inputs[inputIndex] - target[inputIndex];
#else
	const real_t input = inputs[inputIndex];
	const real_t temp2 = (input - target[inputIndex]) / (input * (ONE - input));
	output[outputIndex] =  ACTIVATION_DERIVATIVE(temp2, input);
#endif
}

#elif defined(CE_ANY)

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

	const int inputIndex = (zIndex + INPUT_UNIT_OFFSET)* INPUT_UNIT_ELEMENT_COUNT_INC_PADDING + (yIndex + INPUT_OFFSET_HEIGHT) * INPUT_STRIDE + xIndex + INPUT_OFFSET_WIDTH;
	const int outputIndex = (zIndex + OUTPUT_UNIT_OFFSET) * OUTPUT_UNIT_ELEMENT_COUNT_INC_PADDING + (yIndex + OUTPUT_OFFSET_HEIGHT) * OUTPUT_STRIDE + xIndex + OUTPUT_OFFSET_WIDTH;

#if defined(SIGMOID)
	output[outputIndex] = -target[inputIndex] * (ONE - input[index]);
#elif defined(TANH)
	const real_t tempInput = input[inputIndex];
	const real_t temp2 = -target[inputIndex] / tempInput;
	output[outputIndex] = ACTIVATION_DERIVATIVE(temp2, tempInput);
#endif
}

#else
#error "There's no error function defined"
#endif
