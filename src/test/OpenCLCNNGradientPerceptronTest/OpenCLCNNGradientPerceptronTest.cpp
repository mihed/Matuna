/*
 * OpenCLCNNGradientPerceptronTest.cpp
 *
 *  Created on: May 17, 2015
 *      Author: Mikael
 */
#define CATCH_CONFIG_MAIN
#include "catch/catch.hpp"
#include "OpenCLHelper/OpenCLHelper.h"
#include "CNNOpenCL/CNNOpenCL.h"
#include "CNNOpenCL/PerceptronLayer.h"
#include "CNN/PerceptronLayerConfig.h"
#include "CNN/StandardOutputLayerConfig.h"
#include "CNNOpenCL/PerceptronLayer.h"
#include "Math/Matrix.h"
#include <memory>
#include <random>
#include <type_traits>

using namespace std;
using namespace ATML::MachineLearning;
using namespace ATML::Math;
using namespace ATML::Helper;

unique_ptr<CNNConfig> CreateRandomCNNPerceptronConfig(mt19937& mt,
	uniform_int_distribution<int>& layerGenerator,
	uniform_int_distribution<int>& dimensionGenerator)
{
	vector<LayerDataDescription> dataDescriptions;
	LayerDataDescription dataDescription;
	dataDescription.Height = 1;
	dataDescription.Width = 1;
	dataDescription.Units = dimensionGenerator(mt);
	dataDescriptions.push_back(dataDescription);

	int layerCount = layerGenerator(mt);
	uniform_int_distribution<int> activationGenerator(1, 3);

	INFO("Initializing the CNN config");
	unique_ptr<CNNConfig> config(new CNNConfig(dataDescriptions));

	INFO("Creating the layers config");
	for (int i = 0; i < layerCount; i++)
	{
		auto activation = activationGenerator(mt);
		ATMLActivationFunction activationFunction;
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

		unique_ptr<PerceptronLayerConfig> perceptronConfig(new PerceptronLayerConfig(dimensionGenerator(mt), activationFunction));
		config->AddToBack(move(perceptronConfig));
	}

	unique_ptr<StandardOutputLayerConfig> outputConfig(new StandardOutputLayerConfig());
	config->SetOutputConfig(move(outputConfig));

	return move(config);
}

SCENARIO("Calculating the gradient of a CNN using perceptron layers")
{
	auto platformInfos = OpenCLHelper::GetPlatformInfos();
	random_device device;
	mt19937 mt(device());
	uniform_int_distribution<int> dimensionGenerator(1, 200);
	uniform_int_distribution<int> layerGenerator(1, 5);

	for (int dummy = 0; dummy < 5; dummy++)
	{

		vector<vector<OpenCLDeviceInfo>> deviceInfos;
		for (auto platformInfo : platformInfos)
		{
			auto temp = OpenCLHelper::GetDeviceInfos(platformInfo);
			vector<OpenCLDeviceInfo> capabaleDevices;
			for (auto& deviceInfo : temp)
				if (deviceInfo.PreferredDoubleVectorWidth() != 0)
					capabaleDevices.push_back(deviceInfo);

			deviceInfos.push_back(capabaleDevices);
		}

		for (auto& deviceInfo : deviceInfos)
		{
			auto config = CreateRandomCNNPerceptronConfig(mt, layerGenerator, dimensionGenerator);
			CNNOpenCL<double> network(deviceInfo, move(config));

			LayerDataDescription inputDataDesc = network.InputForwardDataDescriptions()[0];
			LayerDataDescription outputDataDesc = network.OutputForwardDataDescriptions()[0];

			auto input = Matrix<double>::RandomNormal(inputDataDesc.Units, 1, 0, 4);
			auto target = Matrix<double>::RandomNormal(outputDataDesc.Units, 1, 0, 4);

			auto parameterCount = network.GetParameterCount();

			unique_ptr<double[]> gradient = network.CalculateGradientUnaligned(input.Data, 0, target.Data);
			Matrix<double> gradientMatrix(parameterCount, 1, gradient.get());
			gradient.reset();

			//Let us now compare the calculated gradient to the finite difference gradient
			auto parameters = network.GetParameters();
			Matrix<double> parameterMatrix(parameterCount, 1, parameters.get());
			parameters.reset();
			double h = 1E-5;

			Matrix<double> finiteDifferenceGradient(parameterCount, 1);

			for (int i = 0; i < parameterCount; i++)
			{
				Matrix<double> param1 = parameterMatrix;
				param1.At(i, 0) = param1.At(i, 0) - h;
				network.SetParameters(param1.Data);
				auto forward1 = network.FeedForwardUnaligned(input.Data, 0);
				Matrix<double> forward1Matrix(outputDataDesc.Units, 1, forward1.get());
				forward1.reset();

				//TODO: Refactor this method so that it requires forward prop
				auto minusValue = network.CalculateErrorAligned(forward1Matrix.Data, 0, target.Data);

				Matrix<double> param2 = parameterMatrix;
				param2.At(i, 0) = param2.At(i, 0) + h;
				network.SetParameters(param2.Data);
				auto forward2 = network.FeedForwardUnaligned(input.Data, 0);
				Matrix<double> forward2Matrix(outputDataDesc.Units, 1, forward2.get());
				forward2.reset();

				auto plusValue = network.CalculateErrorAligned(forward2Matrix.Data, 0, target.Data);

				finiteDifferenceGradient.At(i, 0) = (plusValue - minusValue) / (2 * h);
			}
			//for (int i = 0; i < parameterCount; i++)
			//{
			//	cout << "Gradient: " << gradientMatrix.At(i, 0) << endl;
			//	cout << "Finite difference: " << finiteDifferenceGradient.At(i, 0) << endl;
			//}

			auto difference = (gradientMatrix - finiteDifferenceGradient).Norm2Square() / parameterCount;
			cout << "Difference: " << difference << endl;
			CHECK(difference < 1E-13);
		}
	}
}

