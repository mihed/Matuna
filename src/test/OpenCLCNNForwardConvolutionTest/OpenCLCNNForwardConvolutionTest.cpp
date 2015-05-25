/*
 * OpenCLCNNForwardConvolutionTest.cpp
 *
 *  Created on: May 25, 2015
 *      Author: Mikael
 */

#define CATCH_CONFIG_MAIN
#include "catch/catch.hpp"
#include "OpenCLHelper/OpenCLHelper.h"
#include "CNNOpenCL/CNNOpenCL.h"
#include "CNNOpenCL/ConvolutionLayer.h"
#include "CNN/ConvolutionLayerConfig.h"
#include "CNN/StandardOutputLayerConfig.h"
#include "Math/Matrix.h"
#include <memory>
#include <random>
#include <type_traits>

using namespace std;
using namespace ATML::MachineLearning;
using namespace ATML::Math;
using namespace ATML::Helper;

SCENARIO("Forward propagating a convolution layer in a OpenCLCNN")
{

}




