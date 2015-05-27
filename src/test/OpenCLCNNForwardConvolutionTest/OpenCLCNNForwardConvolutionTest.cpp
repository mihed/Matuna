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
	auto platformInfos = OpenCLHelper::GetPlatformInfos();
	random_device device;
	mt19937 mt(device());
	uniform_int_distribution<int> dimensionGenerator(20, 200);
	uniform_int_distribution<int> filterGenerator(1, 20);
	uniform_int_distribution<int> activationGenerator(1, 3);

	for (int dummy = 0; dummy < 10; dummy++)
	{

		vector<vector<OpenCLDeviceInfo>> deviceInfos;
		for (auto platformInfo : platformInfos)
			deviceInfos.push_back(OpenCLHelper::GetDeviceInfos(platformInfo));

		for (auto& deviceInfo : deviceInfos)
		{

			ATMLActivationFunction activationFunction;
			auto activation = activationGenerator(mt);
			switch (activation)
			{
			case 1:
				activationFunction = ATMLSigmoidActivation;
				break;
			case 2:
				activationFunction = ATMLLinearActivation;
				break;
			case 3:
				activationFunction = ATMLTanhActivation;
				break;
			}

			int inputUnits = filterGenerator(mt);
			int inputWidth = dimensionGenerator(mt);
			int inputHeight = dimensionGenerator(mt);

			vector<LayerDataDescription> inputDescriptions;
			LayerDataDescription inputDescription;
			inputDescription.Height = inputHeight;
			inputDescription.Width = inputWidth;
			inputDescription.Units = inputUnits;
			inputDescriptions.push_back(inputDescription);

			unique_ptr<CNNConfig> config(new CNNConfig(inputDescriptions));
			unique_ptr<ForwardBackPropLayerConfig> convConfig(new ConvolutionLayerConfig(
				filterGenerator(mt), filterGenerator(mt), filterGenerator(mt),
				activationFunction));

			unique_ptr<OutputLayerConfig> outputConfig(new StandardOutputLayerConfig());

			config->AddToBack(move(convConfig));
			config->SetOutputConfig(move(outputConfig));

			CNNOpenCL<float> network(deviceInfo, move(config));

			//TODO: Perform forward prop
		}
	}
}




