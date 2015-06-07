/**
 *Macros to define:
 * - DOUBLE_PRECISION: if we are using double precision
 */

#include "RealType.h"

__kernel void DivideByScalarKernel(__global real_t* inputOutput, __constant real_t* scalar)
{
	const real_t privateScalar = *scalar;
	inputOutput[get_global_id(0)] /= privateScalar;
}
