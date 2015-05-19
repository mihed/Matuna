/**
*Macros to define:
* - DOUBLE_PRECISION: if we are using double precision
*/

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

__kernel void DivideByScalarKernel(__global TYPE* inputOutput, __constant TYPE* scalar)
{
    const TYPE privateScalar = *scalar;
    inputOutput[get_global_id(0)] /= privateScalar; 
}