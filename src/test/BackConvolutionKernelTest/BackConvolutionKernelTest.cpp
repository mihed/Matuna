/*
 * BackConvolutionKernelTest.cpp
 *
 *  Created on: May 24, 2015
 *      Author: Mikael
 */


#define CATCH_CONFIG_MAIN
#include "catch/catch.hpp"
#include "OpenCLHelper/OpenCLHelper.h"
#include "CNNOpenCL/BackConvolutionKernel.h"
#include "Math/Matrix.h"
#include <memory>
#include <random>
#include <type_traits>

using namespace ATML::Helper;
using namespace ATML::Math;
using namespace ATML::MachineLearning;
