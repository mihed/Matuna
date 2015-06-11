/**
 *Macros to define:
 * - DOUBLE_PRECISION: if we are using double precision
 */

#include "RealType.h"

//<!@
//#define USE_OFFSET
#define OFFSET_SCALAR -1
//!@>

__kernel void DivideByScalarKernel(__global real_t* inputOutput, const real_t scalar)
{
#ifdef USE_OFFSET
	const real_t privateScalar = scalar * OFFSET_SCALAR;
#else
	const real_t privateScalar = scalar;
#endif

	inputOutput[get_global_id(0)] /= privateScalar;
}
