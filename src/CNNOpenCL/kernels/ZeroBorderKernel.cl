
// Inclusive
#ifndef BORDER_START_LEFT 
#define BORDER_START_LEFT -1
#endif

// Inclusive
#ifndef BORDER_START_RIGHT 
#define BORDER_START_RIGHT -1
#endif

// Inclusive
#ifndef BORDER_START_UP 
#define BORDER_START_UP -1
#endif

// Inclusive
#ifndef BORDER_START_DOWN 
#define BORDER_START_DOWN -1
#endif

// BORDER_START_LEFT + BORDER_SIZE. Inclusive
#ifndef BORDER_LIMIT_LEFT
#define BORDER_LIMIT_LEFT -1
#endif

// BORDER_START_RIGHT + BORDER_SIZE. Inclusive
#ifndef BORDER_LIMIT_RIGHT
#define BORDER_LIMIT_RIGHT -1
#endif

// BORDER_START_UP + BORDER_SIZE. Inclusive
#ifndef BORDER_LIMIT_UP
#define BORDER_LIMIT_UP -1
#endif

// BORDER_START_DOWN + BORDER_SIZE. Inclusive
#ifndef BORDER_LIMIT_DOWN
#define BORDER_LIMIT_DOWN -1
#endif

#ifndef BORDER_SIZE
#define BORDER_SIZE -1
#endif

#ifndef INPUT_UNIT_ELEMENT_COUNT_INC_PADDING 
#define INPUT_UNIT_ELEMENT_COUNT_INC_PADDING -1
#endif

#ifndef INPUT_DATA_WIDTH
#define INPUT_DATA_WIDTH -1
#endif

#ifndef INPUT_DATA_HEIGHT
#define INPUT_DATA_HEIGHT -1
#endif

#ifndef INPUT_STRIDE
#define INPUT_STRIDE -1
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



__kernel void ZeroBorderKernel(__global TYPE* input)
{
    const int unitIndex = get_global_id(0) * INPUT_UNIT_ELEMENT_COUNT_INC_PADDING;
        
    //Adding a border in the height direction
    int tempIndex;
    const int toNextBorder = INPUT_DATA_WIDTH + BORDER_SIZE;
    for (int j = BORDER_LIMIT_UP + 1; j < BORDER_START_DOWN; j++)
    {
        tempIndex = INPUT_STRIDE * j + unitIndex; 
        for (int i = BORDER_START_LEFT; i <= BORDER_LIMIT_LEFT; i++)
        {    
            input[tempIndex + i] = 0;
            input[tempIndex + i + toNextBorder] = 0;
        }
    }
    
    int tempIndex2;
    const int toNextBorder2 = INPUT_DATA_HEIGHT + BORDER_SIZE;
    for (int j = BORDER_START_UP; j <= BORDER_LIMIT_UP; j++)
    {
        tempIndex = INPUT_STRIDE * j + unitIndex;
        tempIndex2 = tempIndex + INPUT_STRIDE * toNextBorder2;
        for (int i = BORDER_START_LEFT; i <= BORDER_LIMIT_RIGHT; i++)
        {
            input[tempIndex + i] = 0;
            input[tempIndex2 + i] = 0;
        }
    }
}