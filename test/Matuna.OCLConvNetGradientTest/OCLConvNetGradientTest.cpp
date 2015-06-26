/*
* OCLConvNetGradientTest.cpp
*
*  Created on: Jun 3, 2015
*      Author: Mikael
*/

#define CATCH_CONFIG_MAIN
#include "catch/catch.hpp"
#include "Matuna.OCLHelper/OCLHelper.h"
#include "Matuna.OCLConvNet/OCLConvNet.h"
#include "Matuna.OCLConvNet/PerceptronLayer.h"
#include "Matuna.ConvNet/PerceptronLayerConfig.h"
#include "Matuna.ConvNet/ConvolutionLayerConfig.h"
#include "Matuna.ConvNet/StandardOutputLayerConfig.h"
#include "Matuna.ConvNet/VanillaSamplingLayerConfig.h"
#include "Matuna.ConvNet/MaxPoolingLayerConfig.h"
#include "Matuna.Math/Matrix.h"
#include <memory>
#include <random>
#include <type_traits>

using namespace std;
using namespace Matuna::MachineLearning;
using namespace Matuna::Math;
using namespace Matuna::Helper;

unique_ptr<ConvNetConfig> CreateRandomConvNetVanillaConfig(mt19937& mt,
														   uniform_int_distribution<int>& perceptronLayerGenerator,
														   uniform_int_distribution<int>& convolutionLayerGenerator,
														   uniform_int_distribution<int>& imageDimensionGenerator,
														   uniform_int_distribution<int>& filterDimensionGenerator,
														   uniform_int_distribution<int>& dimensionGenerator,
														   uniform_int_distribution<int>& vanillaSamplingSizeGenerator,
														   bool useSoftMax,
														   bool useVanillaSampling)
{
	vector<LayerDataDescription> dataDescriptions;
	LayerDataDescription dataDescription;
	dataDescription.Height = imageDimensionGenerator(mt);
	dataDescription.Width = imageDimensionGenerator(mt);
	dataDescription.Units = dimensionGenerator(mt);
	dataDescriptions.push_back(dataDescription);

	int perceptronLayerCount = perceptronLayerGenerator(mt);
	int convolutionLayerCount = convolutionLayerGenerator(mt);
	uniform_int_distribution<int> activationGenerator(1, 3);

	INFO("Initializing the ConvNet config");
	unique_ptr<ConvNetConfig> config(new ConvNetConfig(dataDescriptions));

	MatunaActivationFunction activationFunction;

	for (int i = 0; i < convolutionLayerCount; i++)
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

		unique_ptr<ConvolutionLayerConfig> convConfig(
			new ConvolutionLayerConfig(dimensionGenerator(mt),
			filterDimensionGenerator(mt),
			filterDimensionGenerator(mt), activationFunction));

		config->AddToBack(move(convConfig));
		if (useVanillaSampling)
		{
			unique_ptr<VanillaSamplingLayerConfig> samplingConfig(
				new VanillaSamplingLayerConfig(vanillaSamplingSizeGenerator(mt), vanillaSamplingSizeGenerator(mt)));
			config->AddToBack(move(samplingConfig));
		}
	}

	INFO("Creating the layers config");
	for (int i = 0; i < perceptronLayerCount; i++)
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
			if (i == (perceptronLayerCount - 2)
				&& activationFunction == MatunaLinearActivation)
				activationFunction = MatunaTanhActivation;

		auto temp = dimensionGenerator(mt);
		if (useSoftMax)
		{
			if (i == (perceptronLayerCount - 1))
			{
				activationFunction = MatunaSoftMaxActivation;
				temp = temp > 1 ? temp : 2;
			}
		}

		unique_ptr<PerceptronLayerConfig> perceptronConfig(
			new PerceptronLayerConfig(temp, activationFunction));
		config->AddToBack(move(perceptronConfig));
	}

	if (useSoftMax)
	{
		unique_ptr<StandardOutputLayerConfig> outputConfig(
			new StandardOutputLayerConfig(MatunaCrossEntropy));
		config->SetOutputConfig(move(outputConfig));
	}
	else
	{
		unique_ptr<StandardOutputLayerConfig> outputConfig(
			new StandardOutputLayerConfig());
		config->SetOutputConfig(move(outputConfig));
	}

	return move(config);
}

unique_ptr<ConvNetConfig> CreateRandomConvNetMaxConfig(mt19937& mt,
													   uniform_int_distribution<int>& perceptronLayerGenerator,
													   uniform_int_distribution<int>& convolutionLayerGenerator,
													   uniform_int_distribution<int>& imageDimensionGenerator,
													   uniform_int_distribution<int>& filterDimensionGenerator,
													   uniform_int_distribution<int>& dimensionGenerator,
													   uniform_int_distribution<int>& vanillaSamplingSizeGenerator,
													   bool useSoftMax,
													   bool useMaxSampling)
{
	vector<LayerDataDescription> dataDescriptions;
	LayerDataDescription dataDescription;
	dataDescription.Height = imageDimensionGenerator(mt);
	dataDescription.Width = imageDimensionGenerator(mt);
	dataDescription.Units = dimensionGenerator(mt);
	dataDescriptions.push_back(dataDescription);


	cout << "\n\n------------Network-------------------" << endl;

	cout << "Width: " << dataDescription.Width << " Height: " << dataDescription.Height << " Units: " << dataDescription.Units << endl;

	int perceptronLayerCount = perceptronLayerGenerator(mt);
	int convolutionLayerCount = convolutionLayerGenerator(mt);
	uniform_int_distribution<int> activationGenerator(1, 3);

	INFO("Initializing the ConvNet config");
	unique_ptr<ConvNetConfig> config(new ConvNetConfig(dataDescriptions));

	MatunaActivationFunction activationFunction;

	for (int i = 0; i < convolutionLayerCount; i++)
	{
		cout << "------Convolution layer " << i << " -------" << endl;
		auto activation = activationGenerator(mt);
		switch (activation)
		{
		case 1:
			activationFunction = MatunaSigmoidActivation;
			cout << "Sigmoid" << endl;
			break;
		case 2:
			activationFunction = MatunaLinearActivation;
			cout << "Linear" << endl;
			break;
		case 3:
			activationFunction = MatunaTanhActivation;
			cout << "Tanh" << endl;
			break;
		default:
			throw runtime_error("The activation is not implemented yet");
		}

		int filterWidth = filterDimensionGenerator(mt);
		int filterHeight = filterDimensionGenerator(mt);
		int filterCount = dimensionGenerator(mt);

		cout << "Filter width: " << filterWidth << " Filter height: " << filterHeight << " Filter count: " << filterCount << endl;

		unique_ptr<ConvolutionLayerConfig> convConfig(
			new ConvolutionLayerConfig(filterCount,
			filterWidth,
			filterHeight, activationFunction));

		config->AddToBack(move(convConfig));
		if (useMaxSampling)
		{
			int samplingWidth = vanillaSamplingSizeGenerator(mt);
			int samplingHeight = vanillaSamplingSizeGenerator(mt);

			cout << "-------- Max pooling " << i << " -------" << endl;
			cout << "Sampling width: " << samplingWidth << " Sampling height: " << samplingHeight << endl;
			unique_ptr<MaxPoolingLayerConfig> samplingConfig(
				new MaxPoolingLayerConfig(samplingWidth, samplingHeight));
			config->AddToBack(move(samplingConfig));
		}
	}

	INFO("Creating the layers config");
	for (int i = 0; i < perceptronLayerCount; i++)
	{
		cout << "------Perceptrn layer " << i << " -------" << endl;
		auto activation = 1;//activationGenerator(mt);
		switch (activation)
		{
		case 1:
			activationFunction = MatunaSigmoidActivation;
			cout << "Sigmoid" << endl;
			break;
		case 2:
			activationFunction = MatunaLinearActivation;
			cout << "Linear" << endl;
			break;
		case 3:
			activationFunction = MatunaTanhActivation;
			cout << "Tanh" << endl;
			break;
		default:
			throw runtime_error("The activation is not implemented yet");
		}

		//Simply to avoid overflow when using softmax
		if (useSoftMax)
			if (i == (perceptronLayerCount - 2)
				&& activationFunction == MatunaLinearActivation)
				activationFunction = MatunaTanhActivation;

		auto temp = dimensionGenerator(mt);
		if (useSoftMax)
		{
			if (i == (perceptronLayerCount - 1))
			{
				activationFunction = MatunaSoftMaxActivation;
				temp = temp > 1 ? temp : 2;
			}
		}

		cout << "Units: " << temp << endl;

		unique_ptr<PerceptronLayerConfig> perceptronConfig(
			new PerceptronLayerConfig(temp, activationFunction));
		config->AddToBack(move(perceptronConfig));
	}

	cout << "-------- Output --------------" << endl;
	if (useSoftMax)
	{
		cout << "Softmax" << endl << endl;
		unique_ptr<StandardOutputLayerConfig> outputConfig(
			new StandardOutputLayerConfig(MatunaCrossEntropy));
		config->SetOutputConfig(move(outputConfig));
	}
	else
	{
		cout << "MSE" << endl << endl;
		unique_ptr<StandardOutputLayerConfig> outputConfig(
			new StandardOutputLayerConfig());
		config->SetOutputConfig(move(outputConfig));
	}

	return move(config);
}

SCENARIO("Calcultating the gradient of a ConvNet using random convolution, perceptron and max sampling layers")
{
	auto platformInfos = OCLHelper::GetPlatformInfos();
	random_device device;
	mt19937 mt(device());
	uniform_int_distribution<int> dimensionGenerator(1, 5);
	uniform_int_distribution<int> imageDimensionGenerator(100, 150);
	uniform_int_distribution<int> perceptronLayerGenerator(1, 2);
	uniform_int_distribution<int> convolutionLayerGenerator(1, 2);
	uniform_int_distribution<int> maxSamplingSizeGenerator(1, 3);
	uniform_int_distribution<int> filterGenerator(1, 10);

	for (int dummy = 0; dummy < 20; dummy++)
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

			if (deviceInfo[0].PlatformInfo().GetString().find("CUDA") != string::npos)
				continue;

			unique_ptr<ConvNetConfig> config;

			config = CreateRandomConvNetMaxConfig(mt, perceptronLayerGenerator,
				convolutionLayerGenerator, imageDimensionGenerator,
				filterGenerator, dimensionGenerator, maxSamplingSizeGenerator, false, true);

			OCLConvNet<double> network(deviceInfo, move(config));

			LayerDataDescription inputDataDesc =
				network.InputForwardDataDescriptions()[0];
			LayerDataDescription outputDataDesc =
				network.OutputForwardDataDescriptions()[0];

			int inputUnits = inputDataDesc.Units;
			int inputHeight = inputDataDesc.Height;
			int inputWidth = inputDataDesc.Width;

			unique_ptr<double[]> rawInputs(new double[inputDataDesc.TotalUnits()]);
			for (int i = 0; i < inputUnits; i++)
			{
				auto tempInput = Matrixd::RandomNormal(inputHeight, inputWidth);
				memcpy(rawInputs.get() + i * inputHeight * inputWidth, tempInput.Data, sizeof(double) * inputHeight * inputWidth);
			}

			auto target = Matrix<double>::RandomNormal(outputDataDesc.Units, 1, 0, 4);
			unique_ptr<OCLMemory> inputMemory;
			unique_ptr<OCLMemory> targetMemory;

			if(network.RequireForwardInputAlignment(0))
			{
				auto tempInput = network.AlignToForwardInput(rawInputs.get(), 0);
				inputMemory = network.CreateInputMemory(tempInput.get(), 0, 0);
			}
			else
				inputMemory = network.CreateInputMemory(rawInputs.get(), 0, 0);

			if (network.RequireForwardOutputAlignment(0))
			{
				auto tempTarget = network.AlignToForwardOutput(target.Data, 0);
				targetMemory = network.CreateTargetMemory(tempTarget.get(), 0, 0);
			}
			else
				targetMemory = network.CreateTargetMemory(target.Data, 0, 0);

			int parameterCount = static_cast<int>(network.GetParameterCount());

			unique_ptr<double[]> gradient = network.CalculateGradientAligned(inputMemory.get(), 0, targetMemory.get());
			Matrix<double> gradientMatrix(parameterCount, 1, gradient.get());
			gradient.reset();

			//Let us now compare the calculated gradient to the finite difference gradient
			auto parameters = network.GetParameters();
			Matrix<double> parameterMatrix(parameterCount, 1, parameters.get());
			parameters.reset();
			double h = 1E-8;

			Matrix<double> finiteDifferenceGradient(parameterCount, 1);

			for (int i = 0; i < parameterCount; i++)
			{
				Matrix<double> param1 = parameterMatrix;
				param1.At(i, 0) = param1.At(i, 0) - h;
				network.SetParameters(param1.Data);

				auto minusValue = network.CalculateErrorAligned(inputMemory.get(), 0, targetMemory.get());

				Matrix<double> param2 = parameterMatrix;
				param2.At(i, 0) = param2.At(i, 0) + h;
				network.SetParameters(param2.Data);

				auto plusValue = network.CalculateErrorAligned(inputMemory.get(), 0, targetMemory.get());

				finiteDifferenceGradient.At(i, 0) = (plusValue - minusValue)
					/ (2 * h);
			}

			auto difference =
				(gradientMatrix - finiteDifferenceGradient).Norm2Square()
				/ parameterCount;


			auto differenceTest = (gradientMatrix - finiteDifferenceGradient);

			//cout << differenceTest.GetString() << endl << endl;

			if (difference > 1E-8)
			{
				for (int i = 0; i < parameterCount; i++)
					if (abs(differenceTest.Data[i]) > 1E-4)
						printf("Problematic index %i out of %i\n", i, differenceTest.ElementCount());
			}

			cout << "Difference: " << difference << endl;
			CHECK(difference < 1E-8);
		}
	}
}

SCENARIO("Calcultating the gradient of a ConvNet using random convolution, perceptron and vanilla sampling layers")
{
	auto platformInfos = OCLHelper::GetPlatformInfos();
	random_device device;
	mt19937 mt(device());
	uniform_int_distribution<int> dimensionGenerator(1, 5);
	uniform_int_distribution<int> imageDimensionGenerator(100, 150);
	uniform_int_distribution<int> perceptronLayerGenerator(1, 2);
	uniform_int_distribution<int> convolutionLayerGenerator(1, 2);
	uniform_int_distribution<int> vanillaSamplingSizeGenerator(1, 3);
	uniform_int_distribution<int> filterGenerator(1, 10);

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

			unique_ptr<ConvNetConfig> config;

			if (dummy < 4)
				config = CreateRandomConvNetVanillaConfig(mt, perceptronLayerGenerator,
				convolutionLayerGenerator, imageDimensionGenerator,
				filterGenerator, dimensionGenerator, vanillaSamplingSizeGenerator, false, true);
			else
				config = CreateRandomConvNetVanillaConfig(mt, perceptronLayerGenerator,
				convolutionLayerGenerator, imageDimensionGenerator,
				filterGenerator, dimensionGenerator, vanillaSamplingSizeGenerator, false, false);

			OCLConvNet<double> network(deviceInfo, move(config));

			LayerDataDescription inputDataDesc =
				network.InputForwardDataDescriptions()[0];
			LayerDataDescription outputDataDesc =
				network.OutputForwardDataDescriptions()[0];

			int inputUnits = inputDataDesc.Units;
			int inputHeight = inputDataDesc.Height;
			int inputWidth = inputDataDesc.Width;

			vector<Matrixd> inputs;
			unique_ptr<double[]> rawInputs(
				new double[inputDataDesc.TotalUnits()]);
			for (int i = 0; i < inputUnits; i++)
			{
				inputs.push_back(
					Matrixd::RandomNormal(inputHeight, inputWidth));
				memcpy(rawInputs.get() + i * inputHeight * inputWidth, inputs[i].Data, sizeof(double) * inputHeight * inputWidth);
			}

			auto target = Matrix<double>::RandomNormal(outputDataDesc.Units, 1,
				0, 4);

			int parameterCount = static_cast<int>(network.GetParameterCount());

			unique_ptr<double[]> gradient = network.CalculateGradientUnaligned(
				rawInputs.get(), 0, target.Data);
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

				auto minusValue = network.CalculateErrorUnaligned(
					rawInputs.get(), 0, target.Data);

				Matrix<double> param2 = parameterMatrix;
				param2.At(i, 0) = param2.At(i, 0) + h;
				network.SetParameters(param2.Data);

				auto plusValue = network.CalculateErrorUnaligned(
					rawInputs.get(), 0, target.Data);

				finiteDifferenceGradient.At(i, 0) = (plusValue - minusValue)
					/ (2 * h);
			}
			//for (int i = 0; i < parameterCount; i++)
			//{
			//	cout << "Gradient: " << gradientMatrix.At(i, 0) << endl;
			//	cout << "Finite difference: " << finiteDifferenceGradient.At(i, 0) << endl;
			//}

			auto difference =
				(gradientMatrix - finiteDifferenceGradient).Norm2Square()
				/ parameterCount;
			cout << "Difference: " << difference << endl;
			CHECK(difference < 1E-11);
		}
	}
}



