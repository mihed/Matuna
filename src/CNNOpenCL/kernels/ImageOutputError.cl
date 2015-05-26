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

#ifndef INPUT_OFFSET_WIDTH
#define INPUT_OFFSET_WIDTH -1
#endif

#ifndef INPUT_WIDTH_LIMIT
#define INPUT_WIDTH_LIMIT -1
#endif

#ifndef INPUT_HEIGHT_LIMIT
#define INPUT_HEIGHT_LIMIT -1
#endif

#ifndef INPUT_OFFSET_HEIGHT
#define INPUT_OFFSET_HEIGHT -1
#endif

#ifndef INPUT_UNIT_OFFSET
#define INPUT_UNIT_OFFSET -1
#endif

#ifndef INPUT_STRIDE
#define INPUT_STRIDE -1
#endif

#ifndef INPUT_UNIT_LIMIT
#define INPUT_UNIT_LIMIT -1
#endif

//Width * Height
#ifndef INPUT_UNIT_ELEMENT_COUNT_INC_PADDING 
#define INPUT_UNIT_ELEMENT_COUNT_INC_PADDING -1
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
#define HALF 0.5
typedef double TYPE;

#else
#define ONE 1.0f
#define HALF 0.5f
typedef float TYPE;
#endif

#if defined(CE_BINARY)

__kernel void Error(
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
	__global TYPE* error)
{
    
    const int inputIndex = INPUT_UNIT_OFFSET * INPUT_UNIT_ELEMENT_COUNT_INC_PADDING + INPUT_OFFSET_HEIGHT * INPUT_STRIDE + INPUT_OFFSET_WIDTH;
    const TYPE inputValue = input[inputIndex];
    const TYPE targetValue = target[inputIndex];
    
    
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
	__constant TYPE* inputs,
#else
    __global const TYPE* inputs,
#endif
#ifdef CONSTANT_TARGET
	__constant TYPE* targets,
#else
    __global const TYPE* targets,
#endif
	__global TYPE* error)
{

	TYPE sum = 0;
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
	__constant TYPE* inputs,
#else
    __global const TYPE* inputs,
#endif
#ifdef CONSTANT_TARGET
	__constant TYPE* targets,
#else
    __global const TYPE* targets,
#endif
	__global TYPE* error)
{
	TYPE sum = 0;
    TYPE temp;
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