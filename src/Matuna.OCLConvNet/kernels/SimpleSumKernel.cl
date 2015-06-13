//TODO: this kernel should use some more sophisticated reductiont technique in the future

/**
 * Macros to define:
 * - DOUBLE_PRECISION: if we are using double precision
 * - INPUT_COUNT: The count of the input
 */

#include "RealType.h"

//<!@
#define INPUT_COUNT -1
//!@>

__kernel void SimpleSumKernel(__global real_t* input, __global real_t* result) //We don't even bother to use constant here since there's nothing to cache
{
	real_t privateResult = 0;

	for (int i = 0; i < INPUT_COUNT; i++)
	{
		privateResult += input[i];
	}

	*result = privateResult;
}
