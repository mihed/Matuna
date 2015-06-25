/*
 * ActivationFunction.h
 *
 *  Created on: Jun 7, 2015
 *      Author: Mikael
 */

#ifndef MATUNA_MATUNA_OCLCONVNET_KERNELS_ACTIVATIONFUNCTION_H_
#define MATUNA_MATUNA_OCLCONVNET_KERNELS_ACTIVATIONFUNCTION_H_

#include "RealType.h"

#ifdef DOUBLE_PRECISION
#define ONE 1.0
#define TANH_OUTER 1.7159
#define TANH_INNER 0.666666666666666
#else
#define ONE 1.0f
#define TANH_OUTER 1.7159f
#define TANH_INNER 0.6666666f
#endif

#if defined(MATUNA_ACTIVATION_SIGMOID)
#ifndef DOUBLE_PRECISION
#if defined(HALF_MATH)
#define ACTIVATION(x)	(ONE / (ONE + half_exp(-(x))))
#elif defined(NATIVE_MATH)
#define ACTIVATION(x)	(ONE / (ONE + native_exp(-(x))))
#else
#define ACTIVATION(x)	(ONE / (ONE + exp(-(x))))
#endif
#else
#define ACTIVATION(x) 	(ONE / (ONE + exp(-(x))))
#endif

#elif defined(MATUNA_ACTIVATION_TANH)
#define ACTIVATION(x)	(TANH_OUTER * tanh(TANH_INNER * (x)))
#elif defined(MATUNA_ACTIVATION_SOFTMAX)
#ifndef DOUBLE_PRECISION
#if defined(HALF_MATH)
#define ACTIVATION(x)	(half_exp(x))
#elif defined(NATIVE_MATH)
#define ACTIVATION(x) 	(native_exp(x))
#else
#define ACTIVATION(x) 	(exp(x))
#endif
#else
#define ACTIVATION(x)	(exp(x))
#endif

#else
#define ACTIVATION(x)	(x)
#endif

#if defined(MATUNA_ACTIVATION_DERIVATIVE_SIGMOID)
#define ACTIVATION_DERIVATIVE(x,y)	(x) * (y) * (ONE - (y))
#elif defined(MATUNA_ACTIVATION_DERIVATIVE_TANH)
#define ACTIVATION_DERIVATIVE(x,y)	(x) * TANH_INNER * (TANH_OUTER - ((y) * (y)) / TANH_OUTER)
#else
//NO ACTIVATION DERIVATIVE FOR SOFTMAX AT THE MOMENT
#define ACTIVATION_DERIVATIVE(x,y)	(x)
#endif

#endif /* MATUNA_MATUNA_OCLCONVNET_KERNELS_ACTIVATIONFUNCTION_H_ */
