/*
 * RealType.h
 *
 *  Created on: Jun 7, 2015
 *      Author: Mikael
 */

#ifndef MATUNA_OCLCONVNET_KERNELS_REALTYPE_H_
#define MATUNA_OCLCONVNET_KERNELS_REALTYPE_H_

#ifdef DOUBLE_PRECISION

#if defined(cl_khr_fp64)  // Khronos extension available?
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
#elif defined(cl_amd_fp64)  // AMD extension available?
#pragma OPENCL EXTENSION cl_amd_fp64 : enable
#else
#error "Double precision floating point not supported by OpenCL implementation."
#endif

typedef double real_t;
#else

typedef float real_t;
#endif


#endif /* MATUNA_OCLCONVNET_KERNELS_REALTYPE_H_ */
