#include "RealType.h"

__kernel void AccumulateVectorKernel(
		__global const real_t* input,
		__global real_t* accumulator
)
{
	const int index = get_global_id(0);
	accumulator[index] += input[index];
}

__kernel void AccumulateVectorWithScalarKernel(
		__global const real_t* input,
		__global real_t* accumulator,
		const real_t scalar
)
{
	const int index = get_global_id(0);
	accumulator[index] += scalar * input[index];
}
