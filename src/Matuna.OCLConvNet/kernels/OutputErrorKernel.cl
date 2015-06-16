//Important: This is supposed to be executed as a task since there's only one work unit

#include "RealType.h"

//<!@
#define IN_TARGET_UNIT_MEMORY_WIDTH_OFFSET -1
#define IN_TARGET_WIDTH_LIMIT -1
#define IN_TARGET_HEIGHT_LIMIT -1
#define IN_TARGET_UNIT_LIMIT -1
#define IN_TARGET_UNIT_MEMORY_HEIGHT_OFFSET -1
#define IN_TARGET_UNIT_OFFSET -1
#define IN_TARGET_UNIT_MEMORY_WIDTH -1
#define IN_TARGET_UNIT_MEMORY_ELEMENTS -1
//#define MATUNA_MSE
//#define MATUNA_CE_BINARY
//#define MATUNA_CE
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

#if defined(MATUNA_CE_BINARY)

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

	const int inputIndex = IN_TARGET_UNIT_OFFSET * IN_TARGET_UNIT_MEMORY_ELEMENTS + IN_TARGET_UNIT_MEMORY_HEIGHT_OFFSET * IN_TARGET_UNIT_MEMORY_WIDTH + IN_TARGET_UNIT_MEMORY_WIDTH_OFFSET;
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

#elif defined(MATUNA_CE)

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
	for (int i = IN_TARGET_UNIT_OFFSET; i < IN_TARGET_UNIT_LIMIT; i++)
	{
		temp1 = IN_TARGET_UNIT_MEMORY_ELEMENTS * i;
		for (int j = IN_TARGET_UNIT_MEMORY_HEIGHT_OFFSET; j < IN_TARGET_HEIGHT_LIMIT; j++)
		{
			temp2 = temp1 + IN_TARGET_UNIT_MEMORY_WIDTH * j;
			for (int k = IN_TARGET_UNIT_MEMORY_WIDTH_OFFSET; k < IN_TARGET_WIDTH_LIMIT; k++)
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

#elif defined(MATUNA_MSE)

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

	for (int i = IN_TARGET_UNIT_OFFSET; i < IN_TARGET_UNIT_LIMIT; i++)
	{
		temp1 = IN_TARGET_UNIT_MEMORY_ELEMENTS * i;
		for (int j = IN_TARGET_UNIT_MEMORY_HEIGHT_OFFSET; j < IN_TARGET_HEIGHT_LIMIT; j++)
		{
			temp2 = temp1 + IN_TARGET_UNIT_MEMORY_WIDTH * j;
			for (int k = IN_TARGET_UNIT_MEMORY_WIDTH_OFFSET; k < IN_TARGET_WIDTH_LIMIT; k++)
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
