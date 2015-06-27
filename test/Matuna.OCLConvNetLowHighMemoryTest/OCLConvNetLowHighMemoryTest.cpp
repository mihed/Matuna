/*
* OCLConvNetLowHighMemoryTest.cpp
*
*  Created on: Jun 27, 2015
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

void CreateConfig2(bool useLowMemory, mt19937& mt,
				   uniform_int_distribution<int>& perceptronLayerGenerator,
				   uniform_int_distribution<int>& convolutionLayerGenerator,
				   uniform_int_distribution<int>& imageDimensionGenerator,
				   uniform_int_distribution<int>& filterDimensionGenerator,
				   uniform_int_distribution<int>& dimensionGenerator,
				   uniform_int_distribution<int>& vanillaSamplingSizeGenerator,
				   bool useSoftMax,
				   bool useMaxSampling,
				   vector<unique_ptr<ConvNetConfig>>& result,
				   int numberOfConfigs = 2)
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

	for (int i = 0; i < numberOfConfigs; i++)
	{
		unique_ptr<ConvNetConfig> config(new ConvNetConfig(dataDescriptions, useLowMemory));
		result.push_back(move(config));
	}

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

		for (auto& config : result)
		{
			unique_ptr<ConvolutionLayerConfig> convConfig(
				new ConvolutionLayerConfig(filterCount,
				filterWidth,
				filterHeight, activationFunction));

			config->AddToBack(move(convConfig));
		}
		if (useMaxSampling)
		{
			int samplingWidth = vanillaSamplingSizeGenerator(mt);
			int samplingHeight = vanillaSamplingSizeGenerator(mt);

			cout << "-------- Max pooling " << i << " -------" << endl;
			cout << "Sampling width: " << samplingWidth << " Sampling height: " << samplingHeight << endl;
			for (auto& config : result)
			{
				unique_ptr<MaxPoolingLayerConfig> samplingConfig(
					new MaxPoolingLayerConfig(samplingWidth, samplingHeight));
				config->AddToBack(move(samplingConfig));
			}
		}
	}

	for (int i = 0; i < perceptronLayerCount; i++)
	{
		cout << "------Perceptrn layer " << i << " -------" << endl;
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
		for (auto& config : result)
		{
			unique_ptr<PerceptronLayerConfig> perceptronConfig(
				new PerceptronLayerConfig(temp, activationFunction));
			config->AddToBack(move(perceptronConfig));
		}
	}

	cout << "-------- Output --------------" << endl;
	if (useSoftMax)
	{
		cout << "Softmax" << endl << endl;
		for (auto& config : result)
		{
			unique_ptr<StandardOutputLayerConfig> outputConfig(
				new StandardOutputLayerConfig(MatunaCrossEntropy));
			config->SetOutputConfig(move(outputConfig));
		}
	}
	else
	{
		cout << "MSE" << endl << endl;
		for (auto& config : result)
		{
			unique_ptr<StandardOutputLayerConfig> outputConfig(
				new StandardOutputLayerConfig());
			config->SetOutputConfig(move(outputConfig));
		}
	}
}

unique_ptr<float[]> GetDataFromDescription(LayerDataDescription dataDescription)
{
	unique_ptr<float[]> outputs(new float[dataDescription.TotalUnits()]); 
	for (int i = 0; i < dataDescription.Units; i++)
	{
		auto targetMatrix = Matrixf::RandomNormal(dataDescription.Height, dataDescription.Width);
		memcpy(outputs.get() + i * dataDescription.Width * dataDescription.Height, targetMatrix.Data, dataDescription.Width * dataDescription.Height * sizeof(float));
	}

	return move(outputs);
}


TEST_CASE("Making sure that the low and high memory usages give the same result")
{
	auto platformInfos = OCLHelper::GetPlatformInfos();
	random_device randomDevice;
	mt19937 mt(randomDevice());
	uniform_int_distribution<int> dimensionGenerator(1, 5);
	uniform_int_distribution<int> imageDimensionGenerator(100, 150);
	uniform_int_distribution<int> perceptronLayerGenerator(1, 2);
	uniform_int_distribution<int> convolutionLayerGenerator(1, 2);
	uniform_int_distribution<int> maxSamplingSizeGenerator(1, 3);
	uniform_int_distribution<int> filterGenerator(1, 10);


	for (int dummy = 0; dummy < 20; dummy++)
	{
		for (auto platformInfo : platformInfos)
		{
			auto deviceInfos = OCLHelper::GetDeviceInfos(platformInfo);
			for (auto& deviceInfo : deviceInfos)
			{

				unique_ptr<ConvNetConfig> config;

				vector<unique_ptr<ConvNetConfig>> configs;
				CreateConfig2(true, mt, perceptronLayerGenerator, convolutionLayerGenerator,
					imageDimensionGenerator, filterGenerator, dimensionGenerator, maxSamplingSizeGenerator, false, true, configs);

				auto& lowMemoryConfig = configs[0];
				auto& highMemoryConfig = configs[1];
				highMemoryConfig->SetLowMemoryUsage(false);

				vector<OCLDeviceInfo> oneDeviceInfo;
				oneDeviceInfo.push_back(deviceInfo);

				cout << deviceInfo.PlatformInfo().PlatformName() << endl << deviceInfo.DeviceName() << endl << "Type: " << deviceInfo.Type() << endl << endl;

				//Create the first convnet and fetch the parameters
				unique_ptr<OCLConvNet<float>> lowMemoryConvNet(new OCLConvNet<float>(oneDeviceInfo, move(lowMemoryConfig)));
				auto parameters = lowMemoryConvNet->GetParameters();

				auto inputMemory = GetDataFromDescription(lowMemoryConvNet->InputForwardDataDescriptions()[0]);
				auto targetMemory = GetDataFromDescription(lowMemoryConvNet->OutputForwardDataDescriptions()[0]);

				int forwardSize = lowMemoryConvNet->OutputForwardDataDescriptions()[0].TotalUnits();
				int backSize = lowMemoryConvNet->OutputBackDataDescriptions()[0].TotalUnits();
				size_t parametersSize = lowMemoryConvNet->GetParameterCount();

				auto forwardTestLow = lowMemoryConvNet->FeedForwardUnaligned(inputMemory.get(), 0);
				auto backTestLow = lowMemoryConvNet->BackPropUnaligned(inputMemory.get(), 0, targetMemory.get());
				auto gradientTestLow = lowMemoryConvNet->CalculateGradientUnaligned(inputMemory.get(), 0, targetMemory.get());

				lowMemoryConvNet.reset();

				unique_ptr<OCLConvNet<float>> highMemoryConvNet(new OCLConvNet<float>(oneDeviceInfo, move(highMemoryConfig)));


				CHECK(forwardSize == highMemoryConvNet->OutputForwardDataDescriptions()[0].TotalUnits());
				CHECK(backSize == highMemoryConvNet->OutputBackDataDescriptions()[0].TotalUnits());
				CHECK(parametersSize == highMemoryConvNet->GetParameterCount());

				INFO("Setting the parameters from the low network");
				highMemoryConvNet->SetParameters(parameters.get());

				auto forwardTestHigh = highMemoryConvNet->FeedForwardUnaligned(inputMemory.get(), 0);
				auto backTestHigh = highMemoryConvNet->BackPropUnaligned(inputMemory.get(), 0, targetMemory.get());
				auto gradientTestHigh = highMemoryConvNet->CalculateGradientUnaligned(inputMemory.get(), 0, targetMemory.get());

				INFO("Comparing high and low memory usages");

				for (int i = 0; i < forwardSize; i++)
					CHECK(forwardTestLow[i] == forwardTestHigh[i]);

				cout << "Forward checks out" << endl;

				for (int i = 0; i < backSize; i++)
					CHECK(backTestLow[i] == backTestHigh[i]);

				cout << "Back checks out " << endl;

				for (size_t i = 0; i < parametersSize; i++)
					CHECK(gradientTestLow[i] == gradientTestHigh[i]);

				cout << "Gradient check out " << endl;
			}
		}
	}
}