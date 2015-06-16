/**
 *Macros to define:
 * - DOUBLE_PRECISION: If the kernel is to be executed with double precision
 * - UNIT_LIMIT: This is the upper limit when summarizing the input. This must equal the number of data units + padding
 * - INPUT_UNIT_OFFSET: The memory offset of the input units
 * - INPUT_UNIT_MEMORY_WIDTH_OFFSET: The memory width offset of the input units
 * - INPUT_UNIT_MEMORY_HEIGHT_OFFSET: The memory width offset of the input units
 * - OUTPUT_UNIT_MEMORY_WIDTH_OFFSET: The memory width offset of the output units
 * - OUTPUT_UNIT_MEMORY_HEIGHT_OFFSET: The memory height offset of the output units
 * - INPUT_UNIT_MEMORY_WIDTH: The memory width of the input units
 * - INPUT_UNIT_MEMORY_ELEMENTS: The memory count of one unit in the input
 * - OUTPUT_UNIT_MEMORY_WIDTH: The memory width of the output units.
 * - CONSTANT_INPUT: If we put the input into __constant space
 */

#include "RealType.h"

//TEST---------------------
//#pragma OPENCL EXTENSION cl_intel_printf : enable
//END TEST-----------------


//<!@
#define UNIT_LIMIT -1
#define INPUT_UNIT_OFFSET -1
#define INPUT_UNIT_MEMORY_WIDTH_OFFSET -1
#define INPUT_UNIT_MEMORY_HEIGHT_OFFSET -1
#define OUTPUT_UNIT_MEMORY_WIDTH_OFFSET -1
#define OUTPUT_UNIT_MEMORY_HEIGHT_OFFSET -1
#define INPUT_UNIT_MEMORY_WIDTH -1
#define INPUT_UNIT_MEMORY_ELEMENTS -1
#define OUTPUT_UNIT_MEMORY_WIDTH -1
//#define CONSTANT_INPUT
//!@>

__kernel void SumAllUnitsKernel(
#ifdef CONSTANT_INPUT
		__constant real_t* input,
#else
		__global const real_t* input,
#endif
		__global real_t* output)

{

	//TEST---------------------
	/*#ifdef DOUBLE_PRECISION
	 printf("Double \n");
	 #else
	 printf("Float \n");
	 #endif
	 printf("UNIT_LIMIT: %i \n", UNIT_LIMIT);
	 printf("INPUT_UNIT_OFFSET: %i \n", INPUT_UNIT_OFFSET);
	 printf("INPUT_UNIT_MEMORY_WIDTH_OFFSET: %i \n", INPUT_UNIT_MEMORY_WIDTH_OFFSET);
	 printf("INPUT_UNIT_MEMORY_HEIGHT_OFFSET: %i \n", INPUT_UNIT_MEMORY_HEIGHT_OFFSET);
	 printf("OUTPUT_UNIT_MEMORY_WIDTH_OFFSET: %i \n", OUTPUT_UNIT_MEMORY_WIDTH_OFFSET);
	 printf("OUTPUT_UNIT_MEMORY_HEIGHT_OFFSET: %i \n", OUTPUT_UNIT_MEMORY_HEIGHT_OFFSET);
	 printf("INPUT_UNIT_MEMORY_WIDTH: %i \n", INPUT_UNIT_MEMORY_WIDTH);
	 printf("INPUT_UNIT_MEMORY_ELEMENTS : %i \n", INPUT_UNIT_MEMORY_ELEMENTS );
	 printf("OUTPUT_UNIT_MEMORY_WIDTH: %i \n", OUTPUT_UNIT_MEMORY_WIDTH);
	 */
	//END TEST-----------------
	const int xIndex = get_global_id(0);
	const int yIndex = get_global_id(1);

	const int temp = xIndex + INPUT_UNIT_MEMORY_WIDTH_OFFSET + INPUT_UNIT_MEMORY_WIDTH * (INPUT_UNIT_MEMORY_HEIGHT_OFFSET + yIndex);
	real_t sum = 0;
	for (int z = INPUT_UNIT_OFFSET; z < UNIT_LIMIT; z++)
	{
		sum += input[temp + INPUT_UNIT_MEMORY_ELEMENTS * z];
	}

	//We have a single unit as output
	output[OUTPUT_UNIT_MEMORY_WIDTH_OFFSET + xIndex + OUTPUT_UNIT_MEMORY_WIDTH * (OUTPUT_UNIT_MEMORY_HEIGHT_OFFSET + yIndex)] = sum;
}
