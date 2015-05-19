//TODO: this kernel should use some more sophisticated reductiont technique in the future

/**
* Macros to define:
* - DOUBLE_PRECISION: if we are using double precision
* - INPUT_COUNT: The count of the input
*/



#ifndef INPUT_COUNT
#define INPUT_COUNT -1
#endif

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

__kernel void SimpleSumKernel(__global TYPE* input, __global TYPE* result) //We don't even bother to use constant here since there's nothing to cache
{
    TYPE privateResult = 0;
    
    for (int i = 0; i < INPUT_COUNT; i++)
    {
        privateResult += input[i];
    }
    
    *result = privateResult;
}