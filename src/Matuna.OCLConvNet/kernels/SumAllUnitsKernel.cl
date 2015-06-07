/**
 *Macros to define:
 * - DOUBLE_PRECISION: If the kernel is to be executed with double precision
 * - UNIT_COUNT_INC_PADDING: This is the upper limit when summarizing the input. This must equal the number of data units + padding
 * - UNIT_INPUT_OFFSET: The memory offset of the input units
 * - WIDTH_INPUT_OFFSET: The memory width offset of the input units
 * - HEIGHT_INPUT_OFFSET: The memory width offset of the input units
 * - WIDTH_OUTPUT_OFFSET: The memory width offset of the output units
 * - HEIGHT_OUTPUT_OFFSET: The memory height offset of the output units
 * - WIDTH_INPUT: The memory width of the input units
 * - INPUT_UNIT_ELEMENT_COUNT_INC_PADDING: The memory count of one unit in the input
 * - WIDTH_OUTPUT: The memory width of the output units.
 * - CONSTANT_INPUT: If we put the input into __constant space
 */

#include "RealType.h"

//TEST---------------------
//#pragma OPENCL EXTENSION cl_intel_printf : enable
//END TEST-----------------
//Offset + the data unit count.
#ifndef UNIT_COUNT_INC_PADDING
#define UNIT_COUNT_INC_PADDING -1
#endif

#ifndef UNIT_INPUT_OFFSET
#define UNIT_INPUT_OFFSET -1
#endif

#ifndef WIDTH_INPUT_OFFSET
#define WIDTH_INPUT_OFFSET -1
#endif

#ifndef HEIGHT_INPUT_OFFSET
#define HEIGHT_INPUT_OFFSET -1
#endif

#ifndef WIDTH_OUTPUT_OFFSET
#define WIDTH_OUTPUT_OFFSET -1
#endif

#ifndef HEIGHT_OUTPUT_OFFSET
#define HEIGHT_OUTPUT_OFFSET -1
#endif

#ifndef WIDTH_INPUT
#define WIDTH_INPUT -1
#endif

//Width * Height
#ifndef INPUT_UNIT_ELEMENT_COUNT_INC_PADDING 
#define INPUT_UNIT_ELEMENT_COUNT_INC_PADDING -1
#endif

#ifndef WIDTH_OUTPUT
#define WIDTH_OUTPUT -1
#endif

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
	 printf("UNIT_COUNT_INC_PADDING: %i \n", UNIT_COUNT_INC_PADDING);
	 printf("UNIT_INPUT_OFFSET: %i \n", UNIT_INPUT_OFFSET);
	 printf("WIDTH_INPUT_OFFSET: %i \n", WIDTH_INPUT_OFFSET);
	 printf("HEIGHT_INPUT_OFFSET: %i \n", HEIGHT_INPUT_OFFSET);
	 printf("WIDTH_OUTPUT_OFFSET: %i \n", WIDTH_OUTPUT_OFFSET);
	 printf("HEIGHT_OUTPUT_OFFSET: %i \n", HEIGHT_OUTPUT_OFFSET);
	 printf("WIDTH_INPUT: %i \n", WIDTH_INPUT);
	 printf("INPUT_UNIT_ELEMENT_COUNT_INC_PADDING : %i \n", INPUT_UNIT_ELEMENT_COUNT_INC_PADDING );
	 printf("WIDTH_OUTPUT: %i \n", WIDTH_OUTPUT);
	 */
	//END TEST-----------------
	const int xIndex = get_global_id(0);
	const int yIndex = get_global_id(1);

	const int temp = xIndex + WIDTH_INPUT_OFFSET + WIDTH_INPUT * (HEIGHT_INPUT_OFFSET + yIndex);
	real_t sum = 0;
	for (int z = UNIT_INPUT_OFFSET; z < UNIT_COUNT_INC_PADDING; z++)
	{
		sum += input[temp + INPUT_UNIT_ELEMENT_COUNT_INC_PADDING * z];
	}

	//We have a single unit as output
	output[WIDTH_OUTPUT_OFFSET + xIndex + WIDTH_OUTPUT * (HEIGHT_OUTPUT_OFFSET + yIndex)] = sum;
}
