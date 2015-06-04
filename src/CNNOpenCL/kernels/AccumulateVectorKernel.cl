#ifdef DOUBLE_PRECISION

#if defined(cl_khr_fp64)  // Khronos extension available?
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
#elif defined(cl_amd_fp64)  // AMD extension available?
#pragma OPENCL EXTENSION cl_amd_fp64 : enable
#else
#error "Double precision floating point not supported by OpenCL implementation."
#endif

typedef double TYPE;
#else

typedef float TYPE;
#endif


__kernel void AccumulateVectorKernel(
#ifdef CONSTANT_INPUT
	__constant TYPE* input,
#else
	__global const TYPE* input,
#endif
    __global TYPE* accumulator
)
{
    const int index = get_global_id(0);
    accumulator[index] += input[index];
}

__kernel void AccumulateVectorWithScalarKernel(
#ifdef CONSTANT_INPUT
	__constant TYPE* input,
#else
	__global const TYPE* input,
#endif
    __global TYPE* accumulator,
    const TYPE scalar
)
{
    const int index = get_global_id(0);
    accumulator[index] += scalar * input[index];
}