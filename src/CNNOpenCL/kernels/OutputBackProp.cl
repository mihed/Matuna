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

#ifndef INPUT_UNIT_OFFSET
#define INPUT_UNIT_OFFSET 0
#endif

#ifndef OUTPUT_UNIT_OFFSET
#define OUTPUT_UNIT_OFFSET 0
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

#if defined(DIFFERENCE)

__kernel void BackPropagation(
#ifdef CONSTANT_INPUT
	__constant TYPE* input,
#else
    __global const TYPE* input,
#endif
#ifdef CONSTANT_TARGET
	__constant TYPE* target,
#else
    __global const TYPE* target,
#endif
	__global TYPE* output)
{
    const int globalID = get_global_id(0);
	const int inputIndex = globalID + INPUT_UNIT_OFFSET;
	output[globalID + OUTPUT_UNIT_OFFSET] = input[inputIndex] - target[inputIndex];
}

#elif defined(MSE_ANY)

__kernel void BackPropagation(
#ifdef CONSTANT_INPUT
	__constant TYPE* input,
#else
    __global const TYPE* input,
#endif
#ifdef CONSTANT_TARGET
	__constant TYPE* target,
#else
    __global const TYPE* target,
#endif
	__global TYPE* output)
{
    const int globalID = get_global_id(0);
	const int inputIndex = globalID + INPUT_UNIT_OFFSET;
#if defined(SIGMOID)
	const TYPE tempInput = input[inputIndex];
	output[globalID + OUTPUT_UNIT_OFFSET] = (tempInput - target[inputIndex]) * tempInput * (ONE - tempInput);
#elif defined(TANH)
	const TYPE tempInput = input[inputIndex];
	output[globalID + OUTPUT_UNIT_OFFSET] = (tempInput - target[inputIndex])  * TANH_INNER * (TANH_OUTER - (tempInput * tempInput) / TANH_OUTER);
#else
	output[globalID + OUTPUT_UNIT_OFFSET] = input[inputIndex] - target[inputIndex];
#endif
}

#elif defined(CE_BINARY_ANY)

__kernel void BackPropagation(
#ifdef CONSTANT_INPUT
	__constant TYPE* inputs,
#else
    __global const TYPE* inputs,
#endif
#ifdef CONSTANT_TARGET
	__constant TYPE* target,
#else
    __global const TYPE* target,
#endif
	__global TYPE* output)
{
	const int globalID = get_global_id(0);
	const int inputIndex = globalID + INPUT_UNIT_OFFSET;
#if defined(SIGMOID)
	const TYPE input = inputs[inputIndex];
	output[globalID + OUTPUT_UNIT_OFFSET] = input - target[inputIndex];
#elif defined(TANH)
	const TYPE input = inputs[inputIndex];
	output[globalID + OUTPUT_UNIT_OFFSET] = (input - target[inputIndex]) / (input * (ONE - input)) * TANH_INNER * (TANH_OUTER - (input * input) / TANH_OUTER);
#else
	const TYPE input = inputs[inputIndex];
	output[globalID + OUTPUT_UNIT_OFFSET] = (input - target[inputIndex]) / (input * (ONE - input));
#endif
}

#elif defined(CE_ANY)

__kernel void BackPropagation(
#ifdef CONSTANT_INPUT
	__constant TYPE* input,
#else
    __global const TYPE* input,
#endif
#ifdef CONSTANT_TARGET
	__constant TYPE* target,
#else
    __global const TYPE* target,
#endif
	__global TYPE* output)
{
	const int globalID = get_global_id(0);
	const int inputIndex = globalID + INPUT_UNIT_OFFSET;
#if defined(SIGMOID)
	output[globalID + OUTPUT_UNIT_OFFSET] = -target[inputIndex] * (ONE - input[index]);
#elif defined(TANH)
	const TYPE tempInput = input[inputIndex];
	output[globalID + OUTPUT_UNIT_OFFSET] = -target[inputIndex] / tempInput * TANH_INNER * (TANH_OUTER - (tempInput * tempInput) / TANH_OUTER);
#else
	output[globalID + OUTPUT_UNIT_OFFSET] = -target[inputIndex] / input[inputIndex];
#endif
}

#else
#error "There's no error function defined"
#endif
