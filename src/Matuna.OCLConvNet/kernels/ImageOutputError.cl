//Important: This is supposed to be executed as a task since there's only one work unit

/**
 * Macros to define:
 * - MSE:                 Mean Squared Error function
 * - CE:                  Cross Entropy error function
 * - CE_BINARY:           Cross Entropy Binary function
 * - CONSTANT_INPUT:      If we may put the inputs into the constant memory space.
 * - CONSTANT_TARGET:     If we may put the targets into the constant memory space.
 * - INPUT_UNIT_OFFSET:   The offset of the input / target memory .
 * - INPUT_COUNT:         The number of input units.
 * - DOUBLE_PRECISION:    If the kernel is to be executed with double precision.
 * - HALF_MATH:           If we use half precision math
 * - NATIVE_MATH:         If we use native precision math
 */

#include "RealType.h"

//<!@
#define INPUT_OFFSET_WIDTH -1
#define INPUT_WIDTH_LIMIT -1
#define INPUT_HEIGHT_LIMIT -1
#define INPUT_OFFSET_HEIGHT -1
#define INPUT_UNIT_OFFSET -1
#define INPUT_STRIDE -1
#define INPUT_UNIT_LIMIT -1
#define INPUT_UNIT_ELEMENT_COUNT_INC_PADDING -1
//#define MSE
//#define CE_BINARY
//#define CE
//#define CONSTANT_INPUT
//#define CONSTANT_TARGET
//!@>

#ifdef DOUBLE_PRECISION
#define ONE 1.0
#define HALF 0.5
#else
#define ONE 1.0f
#define HALF 0.5f
#endif

#if defined(CE_BINARY)

__kernel void Error(
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
		__global real_t* error)
{

	const int inputIndex = INPUT_UNIT_OFFSET * INPUT_UNIT_ELEMENT_COUNT_INC_PADDING + INPUT_OFFSET_HEIGHT * INPUT_STRIDE + INPUT_OFFSET_WIDTH;
	const real_t inputValue = input[inputIndex];
	const real_t targetValue = target[inputIndex];

#ifndef DOUBLE_PRECISION
#if defined(HALF_MATH)
	*error = -(targetValue * half_log(inputValue) + (ONE - targetValue) * half_log(ONE - inputValue));
#elif defined(NATIVE_MATH)
	*error = -(targetValue * native_log(inputValue) + (ONE - targetValue) * native_log(ONE - inputValue));
#else
	*error = -(targetValue * log(inputValue) + (ONE - targetValue) * log(ONE - inputValue));
#endif
#else
	*error = -(targetValue * log(inputValue) + (ONE - targetValue) * log(ONE - inputValue));
#endif
}

#elif defined(CE)

__kernel void Error(
#ifdef CONSTANT_INPUT
		__constant real_t* inputs,
#else
		__global const real_t* inputs,
#endif
#ifdef CONSTANT_TARGET
		__constant real_t* targets,
#else
		__global const real_t* targets,
#endif
		__global real_t* error)
{

	real_t sum = 0;
	int temp1;
	int temp2;
	int temp3;
	for (int i = INPUT_UNIT_OFFSET; i < INPUT_UNIT_LIMIT; i++)
	{
		temp1 = INPUT_UNIT_ELEMENT_COUNT_INC_PADDING * i;
		for (int j = INPUT_OFFSET_HEIGHT; j < INPUT_HEIGHT_LIMIT; j++)
		{
			temp2 = temp1 + INPUT_STRIDE * j;
			for (int k = INPUT_OFFSET_WIDTH; k < INPUT_WIDTH_LIMIT; k++)
			{
				temp3 = temp2 + k;
#ifndef DOUBLE_PRECISION
#if defined(HALF_MATH)
				sum += targets[temp3] * half_log(inputs[temp3]);
#elif defined(NATIVE_MATH)
				sum += targets[temp3] * native_log(inputs[temp3]);
#else
				sum += targets[temp3] * log(inputs[temp3]);
#endif
#else
				sum += targets[temp3] * log(inputs[temp3]);
#endif
			}
		}
	}

	*error = -sum;
}

#elif defined(MSE)

__kernel void Error(
#ifdef CONSTANT_INPUT
		__constant real_t* inputs,
#else
		__global const real_t* inputs,
#endif
#ifdef CONSTANT_TARGET
		__constant real_t* targets,
#else
		__global const real_t* targets,
#endif
		__global real_t* error)
{
	real_t sum = 0;
	real_t temp;
	int temp1;
	int temp2;
	int temp3;

	for (int i = INPUT_UNIT_OFFSET; i < INPUT_UNIT_LIMIT; i++)
	{
		temp1 = INPUT_UNIT_ELEMENT_COUNT_INC_PADDING * i;
		for (int j = INPUT_OFFSET_HEIGHT; j < INPUT_HEIGHT_LIMIT; j++)
		{
			temp2 = temp1 + INPUT_STRIDE * j;
			for (int k = INPUT_OFFSET_WIDTH; k < INPUT_WIDTH_LIMIT; k++)
			{
				temp3 = temp2 + k;
				temp = targets[temp3] - inputs[temp3];
				sum += temp * temp;
			}
		}
	}

	*error = HALF * sum;
}

#else
#error "No appropriate error function was chosen"
#endif
