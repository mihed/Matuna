/*
 * StandardOutputLayerTest.cpp
 *
 *  Created on: May 15, 2015
 *      Author: Mikael
 */

#define CATCH_CONFIG_MAIN
#include "catch/catch.hpp"
#include "OpenCLHelper/OpenCLHelper.h"
#include "CNNOpenCL/CNNOpenCL.h"
#include "CNN/CNNConfig.h"
#include "CNN/StandardOutputLayerConfig.h"
#include "CNN/StandardOutputLayerConfig.h"
#include "CNN/LayerDescriptions.h"
#include "Math/Matrix.h"
#include <memory>
#include <random>
#include <type_traits>

using namespace std;
using namespace ATML::MachineLearning;
using namespace ATML::Helper;
using namespace ATML::Math;

SCENARIO("Creating a CNN with an standard output layer")
{
	auto platformInfos = OpenCLHelper::GetPlatformInfos();
	random_device device;
	mt19937 mt(device());
	uniform_int_distribution<int> dimensionGenerator(1, 1000);

	vector<vector<OpenCLDeviceInfo>> deviceInfos;
	for (auto platformInfo : platformInfos)
		deviceInfos.push_back(OpenCLHelper::GetDeviceInfos(platformInfo));

	int iterations = 10;

	WHEN("Creating a float CNN with MSE and Linear activation")
	{
		THEN("The back propagation must be equal to difference between target and input")
		{
			for (int dummy = 0; dummy < iterations; dummy++)
			{
				for (auto& deviceInfo : deviceInfos)
				{
					unique_ptr<OutputLayerConfig> outputLayerConfig(new StandardOutputLayerConfig());
					LayerDataDescription inputDescription;
					inputDescription.Height = 1;
					inputDescription.Width = 1;
					inputDescription.Units = dimensionGenerator(mt);
					vector<LayerDataDescription> temp;
					temp.push_back(inputDescription);
					unique_ptr<CNNConfig> config(new CNNConfig(temp));
					config->SetOutputConfig(move(outputLayerConfig));
					CNNOpenCL<float> network(deviceInfo, move(config));
					auto randomInputs = Matrix<float>::RandomNormal(inputDescription.Units, 1);
					auto randomTargets = Matrix<float>::RandomNormal(inputDescription.Units, 1);

					auto result = network.BackPropUnaligned(randomInputs.Data, 0, randomTargets.Data);
					auto manualDifference = randomInputs - randomTargets;
					for (int i = 0; i < inputDescription.Units; i++)
					{
						auto absDifference = (manualDifference.Data[i] - result[i]);
						if (manualDifference.Data[i] > 1E-8)
							absDifference /=  manualDifference.Data[i];
						CHECK(absDifference < 1E-7);
					}
				}
			}
		}
	}
}


