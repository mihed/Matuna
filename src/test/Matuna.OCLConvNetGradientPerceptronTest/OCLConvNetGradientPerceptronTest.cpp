/*
* OCLConvNetGradientPerceptronTest.cpp
*
*  Created on: May 17, 2015
*      Author: Mikael
*/
#define CATCH_CONFIG_MAIN
#include "catch/catch.hpp"
#include "Matuna.OCLHelper/OCLHelper.h"
#include "Matuna.OCLConvNet/OCLConvNet.h"
#include "Matuna.OCLConvNet/PerceptronLayer.h"
#include "Matuna.ConvNet/PerceptronLayerConfig.h"
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

unique_ptr<ConvNetConfig> CreateRandomConvNetPerceptronConfig(mt19937& mt,
													  uniform_int_distribution<int>& layerGenerator,
													  uniform_int_distribution<int>& dimensionGenerator,
													  bool useSoftMax)
{
	vector<LayerDataDescription> dataDescriptions;
	LayerDataDescription dataDescription;
	dataDescription.Height = 1;
	dataDescription.Width = 1;
	dataDescription.Units = dimensionGenerator(mt);
	dataDescriptions.push_back(dataDescription);

	int layerCount = layerGenerator(mt);
	uniform_int_distribution<int> activationGenerator(1, 3);

	INFO("Initializing the ConvNet config");
	unique_ptr<ConvNetConfig> config(new ConvNetConfig(dataDescriptions));

	MatunaActivationFunction activationFunction;
	INFO("Creating the layers config");
	for (int i = 0; i < layerCount; i++)
	{
		auto activation = activationGenerator(mt);
		switch (activation)
		{
		case 1:
			activationFunction = MatunaSigmoidActivation;
			break;
		case 2:
			activationFunction = MatunaLinearActivation;
			break;
		case 3:
			activationFunction = MatunaTanhActivation;
			break;
		default:
			throw runtime_error("The activation is not implemented yet");
		}

		//Simply to avoid overflow when using softmax
		if (useSoftMax)
			if (i == (layerCount - 2) && activationFunction == MatunaLinearActivation)
				activationFunction = MatunaTanhActivation;

		auto temp = dimensionGenerator(mt);
		if (useSoftMax)
		{
			if (i == (layerCount - 1))
			{
				activationFunction = MatunaSoftMaxActivation;
				temp = temp > 1 ? temp : 2;
			}
		}

		unique_ptr<PerceptronLayerConfig> perceptronConfig(new PerceptronLayerConfig(temp, activationFunction));
		config->AddToBack(move(perceptronConfig));
	}

	if (useSoftMax)
	{
		unique_ptr<StandardOutputLayerConfig> outputConfig(new StandardOutputLayerConfig(MatunaCrossEntropy));
		config->SetOutputConfig(move(outputConfig));
	}
	else
	{
		unique_ptr<StandardOutputLayerConfig> outputConfig(new StandardOutputLayerConfig());
		config->SetOutputConfig(move(outputConfig));
	}

	return move(config);
}

unique_ptr<ConvNetConfig> CreateRandomConvNetPerceptronConfigWithImage(mt19937& mt,
															   uniform_int_distribution<int>& layerGenerator,
															   uniform_int_distribution<int>& dimensionGenerator,
															   uniform_int_distribution<int>& imageGenerator,
															   bool useSoftMax)
{
	vector<LayerDataDescription> dataDescriptions;
	LayerDataDescription dataDescription;
	dataDescription.Height = imageGenerator(mt);
	dataDescription.Width = imageGenerator(mt);
	dataDescription.Units = dimensionGenerator(mt);
	dataDescriptions.push_back(dataDescription);

	int layerCount = layerGenerator(mt);
	uniform_int_distribution<int> activationGenerator(1, 3);

	INFO("Initializing the ConvNet config");
	unique_ptr<ConvNetConfig> config(new ConvNetConfig(dataDescriptions));

	MatunaActivationFunction activationFunction;
	INFO("Creating the layers config");
	for (int i = 0; i < layerCount; i++)
	{
		auto activation = activationGenerator(mt);
		switch (activation)
		{
		case 1:
			activationFunction = MatunaSigmoidActivation;
			break;
		case 2:
			activationFunction = MatunaLinearActivation;
			break;
		case 3:
			activationFunction = MatunaTanhActivation;
			break;
		default:
			throw runtime_error("This activation is not implemented yet");
		}

		//Simply to avoid overflow when using softmax
		if (useSoftMax)
			if (i == (layerCount - 2) && activationFunction == MatunaLinearActivation)
				activationFunction = MatunaTanhActivation;

		auto temp = dimensionGenerator(mt);
		if (useSoftMax)
		{
			if (i == (layerCount - 1))
			{
				activationFunction = MatunaSoftMaxActivation;
				temp = temp > 1 ? temp : 2;
			}
		}

		unique_ptr<PerceptronLayerConfig> perceptronConfig(new PerceptronLayerConfig(temp, activationFunction));
		config->AddToBack(move(perceptronConfig));
	}

	if (useSoftMax)
	{
		unique_ptr<StandardOutputLayerConfig> outputConfig(new StandardOutputLayerConfig(MatunaCrossEntropy));
		config->SetOutputConfig(move(outputConfig));
	}
	else
	{
		unique_ptr<StandardOutputLayerConfig> outputConfig(new StandardOutputLayerConfig());
		config->SetOutputConfig(move(outputConfig));
	}

	return move(config);
}

double CalculateCEError(const Matrix<double>& input, const Matrix<double>& target)
{
	double result = 0;
	for (int i = 0; i < input.RowCount(); i++)
		result -= target.At(i, 0) * log(input.At(i, 0));
	return result;
}

SCENARIO("Calculating the gradient of a ConvNet using perceptron layers with image input")
{
	auto platformInfos = OCLHelper::GetPlatformInfos();
	random_device device;
	mt19937 mt(device());
	uniform_int_distribution<int> dimensionGenerator(1, 20);
	uniform_int_distribution<int> imageDimensionGenerator(1,10);
	uniform_int_distribution<int> layerGenerator(1, 4);

	for (int dummy = 0; dummy < 5; dummy++)
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
			auto config = CreateRandomConvNetPerceptronConfigWithImage(mt, layerGenerator, dimensionGenerator, imageDimensionGenerator,false);
			OCLConvNet<double> network(deviceInfo, move(config));

			LayerDataDescription inputDataDesc = network.InputForwardDataDescriptions()[0];
			LayerDataDescription outputDataDesc = network.OutputForwardDataDescriptions()[0];

			int inputUnits = inputDataDesc.Units;
			int inputHeight = inputDataDesc.Height;
			int inputWidth = inputDataDesc.Width;

			vector<Matrixd> inputs;
			unique_ptr<double[]> rawInputs(new double[inputDataDesc.TotalUnits()]);
			for (int i = 0; i < inputUnits; i++)
			{
				inputs.push_back(Matrixd::RandomNormal(inputHeight, inputWidth));
				memcpy(rawInputs.get() + i * inputHeight * inputWidth, inputs[i].Data, sizeof(double) * inputHeight * inputWidth);
			}

			auto target = Matrix<double>::RandomNormal(outputDataDesc.Units, 1, 0, 4);

			int parameterCount = static_cast<int>(network.GetParameterCount());

			unique_ptr<double[]> gradient = network.CalculateGradientUnaligned(rawInputs.get(), 0, target.Data);
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

				auto minusValue = network.CalculateErrorUnaligned(rawInputs.get(), 0, target.Data);

				Matrix<double> param2 = parameterMatrix;
				param2.At(i, 0) = param2.At(i, 0) + h;
				network.SetParameters(param2.Data);

				auto plusValue = network.CalculateErrorUnaligned(rawInputs.get(), 0, target.Data);

				finiteDifferenceGradient.At(i, 0) = (plusValue - minusValue) / (2 * h);
			}
			//for (int i = 0; i < parameterCount; i++)
			//{
			//	cout << "Gradient: " << gradientMatrix.At(i, 0) << endl;
			//	cout << "Finite difference: " << finiteDifferenceGradient.At(i, 0) << endl;
			//}

			auto difference = (gradientMatrix - finiteDifferenceGradient).Norm2Square() / parameterCount;
			cout << "Difference: " << difference << endl;
			CHECK(difference < 1E-11);
		}
	}
}

SCENARIO("Calculating the gradient of a ConvNet using softmax")
{
	auto platformInfos = OCLHelper::GetPlatformInfos();
	random_device device;
	mt19937 mt(device());
	uniform_int_distribution<int> dimensionGenerator(2, 200);
	uniform_int_distribution<int> layerGenerator(1, 5);

	for (int dummy = 0; dummy < 5; dummy++)
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
			auto config = CreateRandomConvNetPerceptronConfig(mt, layerGenerator, dimensionGenerator, true);
			OCLConvNet<double> network(deviceInfo, move(config));

			LayerDataDescription inputDataDesc = network.InputForwardDataDescriptions()[0];
			LayerDataDescription outputDataDesc = network.OutputForwardDataDescriptions()[0];

			auto input = Matrix<double>::RandomNormal(inputDataDesc.Units, 1, 0, 4);
			auto target = Matrix<double>::RandomNormal(outputDataDesc.Units, 1);
			target = (1.0 / target.Sum()) * target;

			int parameterCount = static_cast<int>(network.GetParameterCount());

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

				auto minusValue = network.CalculateErrorUnaligned(input.Data, 0, target.Data);

				Matrix<double> param2 = parameterMatrix;
				param2.At(i, 0) = param2.At(i, 0) + h;
				network.SetParameters(param2.Data);

				auto plusValue = network.CalculateErrorUnaligned(input.Data, 0, target.Data);

				finiteDifferenceGradient.At(i, 0) = (plusValue - minusValue) / (2 * h);
			}
			//for (int i = 0; i < parameterCount; i++)
			//{
			//	cout << "Gradient: " << gradientMatrix.At(i, 0) << endl;
			//	cout << "Finite difference: " << finiteDifferenceGradient.At(i, 0) << endl;
			//}

			auto difference = (gradientMatrix - finiteDifferenceGradient).Norm2Square() / parameterCount;
			cout << "Difference: " << difference << endl;
			CHECK(difference < 1E-11);
		}
	}
}

SCENARIO("Calculating the gradient of a ConvNet using perceptron layers")
{
	auto platformInfos = OCLHelper::GetPlatformInfos();
	random_device device;
	mt19937 mt(device());
	uniform_int_distribution<int> dimensionGenerator(1, 200);
	uniform_int_distribution<int> layerGenerator(1, 5);

	for (int dummy = 0; dummy < 5; dummy++)
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
			auto config = CreateRandomConvNetPerceptronConfig(mt, layerGenerator, dimensionGenerator, false);
			OCLConvNet<double> network(deviceInfo, move(config));

			LayerDataDescription inputDataDesc = network.InputForwardDataDescriptions()[0];
			LayerDataDescription outputDataDesc = network.OutputForwardDataDescriptions()[0];

			auto input = Matrix<double>::RandomNormal(inputDataDesc.Units, 1, 0, 4);
			auto target = Matrix<double>::RandomNormal(outputDataDesc.Units, 1, 0, 4);

			int parameterCount = static_cast<int>(network.GetParameterCount());

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

				auto minusValue = network.CalculateErrorUnaligned(input.Data, 0, target.Data);

				Matrix<double> param2 = parameterMatrix;
				param2.At(i, 0) = param2.At(i, 0) + h;
				network.SetParameters(param2.Data);

				auto plusValue = network.CalculateErrorUnaligned(input.Data, 0, target.Data);

				finiteDifferenceGradient.At(i, 0) = (plusValue - minusValue) / (2 * h);
			}
			//for (int i = 0; i < parameterCount; i++)
			//{
			//	cout << "Gradient: " << gradientMatrix.At(i, 0) << endl;
			//	cout << "Finite difference: " << finiteDifferenceGradient.At(i, 0) << endl;
			//}

			auto difference = (gradientMatrix - finiteDifferenceGradient).Norm2Square() / parameterCount;
			cout << "Difference: " << difference << endl;
			CHECK(difference < 1E-11);
		}
	}
}

