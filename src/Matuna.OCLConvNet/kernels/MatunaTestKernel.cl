/**
 *Macros to define:
 * - DOUBLE_PRECISION: if we are using double precision
 */

#include "RealType.h"

//<!@
//#define MATUNA_TEST_DEFINE
#define MATUNA_TEST_DEFINE2 23324
//!@>

__kernel void DivideByScalarKernel(__global real_t* inputOutput, __constant real_t* scalar)
{
	const real_t privateScalar = *scalar;
	inputOutput[get_global_id(0)] /= privateScalar;
}
