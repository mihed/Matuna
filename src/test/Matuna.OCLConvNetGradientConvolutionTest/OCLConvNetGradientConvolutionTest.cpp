/*
* OCLConvNetGradientConvolutionTest.cpp
*
*  Created on: Jun 1, 2015
*      Author: Mikael
*/
#define CATCH_CONFIG_MAIN
#include "catch/catch.hpp"
#include "Matuna.OCLHelper/OCLHelper.h"
#include "Matuna.OCLConvNet/OCLConvNet.h"
#include "Matuna.OCLConvNet/ConvolutionLayer.h"
#include "Matuna.ConvNet/ConvolutionLayerConfig.h"
#include "Matuna.ConvNet/StandardOutputLayerConfig.h"
#include "Matuna.OCLConvNet/PerceptronLayer.h"
#include "Matuna.Math/Matrix.h"
#include <memory>
#include <random>
#include <type_traits>

using namespace std;
using namespace Matuna::MachineLearning;
using namespace Matuna::Math;
using namespace Matuna::Helper;


unique_ptr<ConvNetConfig> CreateRandomConvNetConvolutionConfig(mt19937& mt,
															   uniform_int_distribution<int>& layerGenerator,
															   uniform_int_distribution<int>& dimensionGenerator,
															   uniform_int_distribution<int>& unitGenerator,
															   uniform_int_distribution<int>& filterGenerator)
{
	vector<LayerDataDescription> dataDescriptions;
	LayerDataDescription dataDescription;
	dataDescription.Height = dimensionGenerator(mt);
	dataDescription.Width = dimensionGenerator(mt);
	dataDescription.Units = unitGenerator(mt);
	dataDescriptions.push_back(dataDescription);

	int layerCount = layerGenerator(mt);
	uniform_int_distribution<int> activationGenerator(1, 3);

	cout << "\n---------------------Network---------------------" << endl;

	cout << "Input height: " << dataDescription.Height << endl;
	cout << "Input width: " << dataDescription.Width << endl;
	cout << "Input units: " << dataDescription.Units << endl;
	cout << "Layers: " << layerCount << endl << endl;

	INFO("Initializing the ConvNet config");
	unique_ptr<ConvNetConfig> config(new ConvNetConfig(dataDescriptions));

	MatunaActivationFunction activationFunction;
	INFO("Creating the layers config");
	for (int i = 0; i < layerCount; i++)
	{
		cout << "--------Layer: " << i << " --------" << endl;
		auto activation = activationGenerator(mt);
		switch (activation)
		{
		case 1:
			activationFunction = MatunaSigmoidActivation;
			cout << "Sigmoid activation" << endl;
			break;
		case 2:
			activationFunction = MatunaLinearActivation;
			cout << "Linear activation" << endl;
			break;
		case 3:
			activationFunction = MatunaTanhActivation;
			cout << "Tanh activation" << endl;
			break;
		default:
			throw runtime_error("The activation is not implemented yet");
		}

		int filterCount = unitGenerator(mt);
		int filterWidth = filterGenerator(mt);
		int filterHeight = filterGenerator(mt);

		cout << "Filter count: " << filterCount << endl;
		cout << "Filter width: " << filterWidth << endl;
		cout << "Filter height: " << filterHeight << endl << endl << endl;

		unique_ptr<ForwardBackPropLayerConfig> convConfig(new ConvolutionLayerConfig(
			filterCount, filterWidth, filterHeight, activationFunction));
		config->AddToBack(move(convConfig));
	}

	unique_ptr<StandardOutputLayerConfig> outputConfig(new StandardOutputLayerConfig());
	config->SetOutputConfig(move(outputConfig));

	return move(config);
}

SCENARIO("Calculating the gradient of a ConvNet using convolution layers")
{
	auto platformInfos = OCLHelper::GetPlatformInfos();
	platformInfos.erase(platformInfos.begin());
	random_device device;
	mt19937 mt(device());
	uniform_int_distribution<int> dimensionGenerator(60, 100);
	uniform_int_distribution<int> filterGenerator(1, 15);
	uniform_int_distribution<int> unitGenerator(1, 10);
	uniform_int_distribution<int> layerGenerator(1, 4);

	for (int dummy = 0; dummy < 10; dummy++)
	{

		vector<vector<OCLDeviceInfo>> deviceInfos;
		for (auto platformInfo : platformInfos)
		{
			auto temp = OCLHelper::GetDeviceInfos(platformInfo);
			vector<OCLDeviceInfo> capabaleDevices;
			for (auto& deviceInfo : temp)
				if (deviceInfo.PreferredDoubleVectorWidth() != 0)
					capabaleDevices.push_back(deviceInfo);

			deviceInfos.push_back(capabaleDevices);
		}

		for (auto& deviceInfo : deviceInfos)
		{
			unique_ptr<ConvNetConfig> config = CreateRandomConvNetConvolutionConfig(mt, layerGenerator, dimensionGenerator, unitGenerator, filterGenerator);
			OCLConvNet<double> network(deviceInfo, move(config));

			LayerDataDescription inputDescription = network.InputForwardDataDescriptions()[0];
			LayerDataDescription outputDescription = network.OutputForwardDataDescriptions()[0];
			LayerDataDescription outBackDescription = network.OutputBackDataDescriptions()[0];
			int inputUnits = inputDescription.Units;

			vector<Matrixd> inputs;
			unique_ptr<double[]> inputMemory(new double[inputDescription.TotalUnits()]);
			for (int i = 0; i < inputUnits; i++)
			{
				auto input = Matrixd::RandomNormal(inputDescription.Height, inputDescription.Width, 0, 4);
				memcpy(inputMemory.get() + i * inputDescription.Height * inputDescription.Width, input.Data, inputDescription.Height * inputDescription.Width * sizeof(double));
				inputs.push_back(input);
			}

			vector<Matrixd> targets;
			unique_ptr<double[]> targetMemory (new double[outputDescription.TotalUnits()]);
			for (int i = 0; i < outputDescription.Units; i++)
			{
				auto target = Matrixd::RandomNormal(outputDescription.Height, outputDescription.Width);
				target = (1.0f / target.Sum()) * target;
				memcpy(targetMemory.get() + i * outputDescription.Height * outputDescription.Width, target.Data, outputDescription.Height * outputDescription.Width * sizeof(double));
				targets.push_back(target);
			}

			auto parameterCount = network.GetParameterCount();

			unique_ptr<double[]> gradient =  network.CalculateGradientUnaligned(inputMemory.get(), 0, targetMemory.get());
			Matrixd gradientMatrix(parameterCount, 1, gradient.get());
			gradient.reset();

			//cout << gradientMatrix.GetString() << endl;

			//Let us now compare the calculated gradient to the finite difference gradient
			auto parameters = network.GetParameters();
			Matrixd parameterMatrix(parameterCount, 1, parameters.get());
			parameters.reset();
			double h = 1E-5;

			Matrixd finiteDifferenceGradient(parameterCount, 1);

			for (size_t i = 0; i < parameterCount; i++)
			{
				Matrixd param1 = parameterMatrix;
				param1.At(i, 0) = param1.At(i, 0) - h;
				network.SetParameters(param1.Data);

				auto minusValue = network.CalculateErrorUnaligned(inputMemory.get(), 0, targetMemory.get());

				Matrixd param2 = parameterMatrix;
				param2.At(i, 0) = param2.At(i, 0) + h;
				network.SetParameters(param2.Data);

				auto plusValue = network.CalculateErrorUnaligned(inputMemory.get(), 0, targetMemory.get());

				finiteDifferenceGradient.At(i, 0) = (plusValue - minusValue) / (2 * h);
			}

			auto difference = (gradientMatrix - finiteDifferenceGradient).Norm2Square() / parameterCount;
			cout << "Difference: " << difference << endl;
			//if (difference > 1E-8)
			//{
			//	cout << "Check this one manually:" << endl;
			//	for (int i = 0; i < parameterCount; i++)
			//	{
			//		cout << "Calculus gradient: " << gradientMatrix.At(i, 0) << endl;
			//		cout << "Finite difference: " << finiteDifferenceGradient.At(i, 0) << endl;
			//	}

			//}
			CHECK(difference < 1E-7);
		}
	}

}
