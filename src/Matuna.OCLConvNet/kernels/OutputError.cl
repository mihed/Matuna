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
#define INPUT_UNIT_OFFSET 0
#define INPUT_COUNT 0
//#define MSE
//#define CE_BINARY
//#define CE
//#define CONSTANT_INPUT
//#define CONSTANT_TARGET
//!@>


//Do not use the double values for these since the specification does not state any particular macro for AMD double extension
#define ZERO_LIMIT 1.0E-12f
#define ONE_LIMIT (1.0f - 1.0E-12f)

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

const real_t tempInput = *input;
const real_t tempTarget = *target;

//This is simply done to increase the stability of the calculation
if (tempInput <= ZERO_LIMIT)
{
   *error = (tempTarget <= ZERO_LIMIT) ? 0 : FLT_MAX;
}
else if (tempInput >= ONE_LIMIT)
{
	*error = (tempTarget >= ONE_LIMIT) ? 0 : FLT_MAX;
}
else
{
#ifndef DOUBLE_PRECISION
#if defined(HALF_MATH)
	*error = -(tempTarget * half_log(tempInput) + (ONE - tempTarget) * half_log(ONE - tempInput));
#elif defined(NATIVE_MATH)
	*error = -(tempTarget * native_log(tempInput) + (ONE - tempTarget) * native_log(ONE - tempInput));
#else
	*error = -(tempTarget * log(tempInput) + (ONE - tempTarget) * log(ONE - tempInput));
#endif
#else
	*error = -(tempTarget * log(tempInput) + (ONE - tempTarget) * log(ONE - tempInput));
#endif
}

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
	for (int i = INPUT_UNIT_OFFSET; i < INPUT_COUNT; i++)
	{
		const real_t tempInput = inputs[i];
		const real_t tempTarget = targets[i];

		//This is done in order to increase the stability of the calculation
		if (tempInput <= ZERO_LIMIT)
		{
			if (tempTarget > ZERO_LIMIT)
			{
				*error = FLT_MAX;
				break;
			}
		}
		else
		{
#ifndef DOUBLE_PRECISION
#if defined(HALF_MATH)
			sum += targets[i] * half_log(inputs[i]);
#elif defined(NATIVE_MATH)
			sum += targets[i] * native_log(inputs[i]);
#else
			sum += targets[i] * log(inputs[i]);
#endif
#else
			sum += targets[i] * log(inputs[i]);
#endif
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
	for (int i = INPUT_UNIT_OFFSET; i < INPUT_COUNT; i++)
	{
		temp = targets[i] - inputs[i];
		sum += temp * temp;
	}
	*error = HALF * sum;
}

#else
#error "No appropriate error function was chosen"
#endif
