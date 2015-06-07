#include "RealType.h"

__kernel void AccumulateVectorKernel(
#ifdef CONSTANT_INPUT
		__constant real_t* input,
#else
		__global const real_t* input,
#endif
		__global real_t* accumulator
)
{
	const int index = get_global_id(0);
	accumulator[index] += input[index];
}

__kernel void AccumulateVectorWithScalarKernel(
#ifdef CONSTANT_INPUT
		__constant real_t* input,
#else
		__global const real_t* input,
#endif
		__global real_t* accumulator,
		const real_t scalar
)
{
	const int index = get_global_id(0);
	accumulator[index] += scalar * input[index];
}
