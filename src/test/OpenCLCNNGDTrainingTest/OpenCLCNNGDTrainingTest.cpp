/*
* OpenCLCNNGDTrainingTest.cpp
*
*  Created on: Jun 3, 2015
*      Author: Mikael
*/

#define CATCH_CONFIG_MAIN
#include "catch/catch.hpp"
#include "OpenCLHelper/OpenCLHelper.h"
#include "CNNOpenCL/CNNOpenCL.h"
#include "CNNOpenCL/PerceptronLayer.h"
#include "CNN/GradientDescentConfig.h"
#include "CNN/PerceptronLayerConfig.h"
#include "CNN/StandardOutputLayerConfig.h"
#include "CNN/ConvolutionLayerConfig.h"
#include "CNNOpenCL/PerceptronLayer.h"
#include "TestCNNTrainer.h"
#include "Math/Matrix.h"
#include <memory>
#include <random>
#include <type_traits>

using namespace std;
using namespace ATML::MachineLearning;
using namespace ATML::Math;
using namespace ATML::Helper;

unique_ptr<CNNConfig> CreateRandomCNNConfig(mt19937& mt,
											uniform_int_distribution<int>& perceptronLayerGenerator,
											uniform_int_distribution<int>& convolutionLayerGenerator,
											uniform_int_distribution<int>& imageDimensionGenerator,
											uniform_int_distribution<int>& filterDimensionGenerator,
											uniform_int_distribution<int>& dimensionGenerator, bool useSoftMax)
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

	INFO("Initializing the CNN config");
	unique_ptr<CNNConfig> config(new CNNConfig(dataDescriptions));

	ATMLActivationFunction activationFunction;

	for (int i = 0; i < convolutionLayerCount; i++)
	{
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

		unique_ptr<ConvolutionLayerConfig> convConfig(
			new ConvolutionLayerConfig(dimensionGenerator(mt),
			filterDimensionGenerator(mt),
			filterDimensionGenerator(mt), activationFunction));

		config->AddToBack(move(convConfig));
	}

	INFO("Creating the layers config");
	for (int i = 0; i < perceptronLayerCount; i++)
	{
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

		//Simply to avoid overflow when using softmax
		if (useSoftMax)
			if (i == (perceptronLayerCount - 2)
				&& activationFunction == ATMLLinearActivation)
				activationFunction = ATMLTanhActivation;

		auto temp = dimensionGenerator(mt);
		if (useSoftMax)
		{
			if (i == (perceptronLayerCount - 1))
			{
				activationFunction = ATMLSoftMaxActivation;
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
			new StandardOutputLayerConfig(ATMLCrossEntropy));
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

SCENARIO("Testing the gradient descent training algorithm")
{
	auto platformInfos = OpenCLHelper::GetPlatformInfos();
	random_device device;
	mt19937 mt(device());
	uniform_int_distribution<int> dimensionGenerator(1, 10);
	uniform_int_distribution<int> imageDimensionGenerator(40, 100);
	uniform_int_distribution<int> perceptronLayerGenerator(1, 3);
	uniform_int_distribution<int> convolutionLayerGenerator(1, 3);
	uniform_int_distribution<int> filterGenerator(1, 10);

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
			auto config = CreateRandomCNNConfig(mt, perceptronLayerGenerator,
				convolutionLayerGenerator, imageDimensionGenerator,
				filterGenerator, dimensionGenerator, false);

			CNNOpenCL<double> network(deviceInfo, move(config));

			auto trainer = new TestCNNTrainer<double>(network.InputForwardDataDescriptions(), network.OutputForwardDataDescriptions(),
				network.InputForwardMemoryDescriptions(), network.OutputForwardMemoryDescriptions(), &network);

			auto algorithmConfig = new GradientDescentConfig<double>();
			algorithmConfig->SetBatchSize(5);
			algorithmConfig->SetEpochs(100);
			algorithmConfig->SetSamplesPerEpoch(5);
			algorithmConfig->SetStepSizeCallback([] (int x){ return -0.00001;});

			for (int i = 0; i < network.OutputForwardMemoryDescriptions().size(); i++)
			{
				auto tempInData = network.InputForwardDataDescriptions();
				auto tempInMem = network.InputForwardMemoryDescriptions();

				CHECK(tempInMem[i].HeightOffset == 0);
				CHECK(tempInMem[i].WidthOffset == 0);
				CHECK(tempInMem[i].UnitOffset == 0);
				CHECK(tempInMem[i].Height == tempInData[i].Height);
				CHECK(tempInMem[i].Width == tempInData[i].Width);
				CHECK(tempInMem[i].Units == tempInData[i].Units);

				auto tempOutData = network.OutputForwardDataDescriptions();
				auto tempOutMem = network.OutputForwardMemoryDescriptions();

				CHECK(tempOutMem[i].HeightOffset == 0);
				CHECK(tempOutMem[i].WidthOffset == 0);
				CHECK(tempOutMem[i].UnitOffset == 0);
				CHECK(tempOutMem[i].Height == tempOutData[i].Height);
				CHECK(tempOutMem[i].Width == tempOutData[i].Width);
				CHECK(tempOutMem[i].Units == tempOutData[i].Units);
			}


			LayerDataDescription inputDataDesc =
				network.InputForwardDataDescriptions()[0];
			LayerDataDescription outputDataDesc =
				network.OutputForwardDataDescriptions()[0];

			int inputUnits = inputDataDesc.Units;
			int inputHeight = inputDataDesc.Height;
			int inputWidth = inputDataDesc.Width;

			auto rawInputs = new double[inputDataDesc.TotalUnits()];
			for (int i = 0; i < inputUnits; i++)
			{
				auto tempInput =  Matrixd::RandomNormal(inputHeight, inputWidth);
				memcpy(rawInputs + i * inputHeight * inputWidth,
					tempInput.Data,
					sizeof(double) * inputHeight * inputWidth);
			}
			unique_ptr<double[]> inputs(rawInputs);

			auto tempTarget = Matrixd::RandomNormal(outputDataDesc.Units, 1,
				0, 1);
			auto rawTarget = new double[outputDataDesc.Units];
			memcpy(rawTarget, tempTarget.Data, sizeof(double) * outputDataDesc.Units);
			unique_ptr<double[]> target(rawTarget);

			double errorBefore = network.CalculateErrorUnaligned(inputs.get(), 0, target.get());
			cout << "Error before iteration: " << errorBefore << endl;
			trainer->SetInput(inputs.get());
			trainer->SetTarget(target.get());
			network.TrainNetwork(unique_ptr<TestCNNTrainer<double>>(trainer), unique_ptr<GradientDescentConfig<double>>(algorithmConfig));
			double errorAfter = network.CalculateErrorUnaligned(inputs.get(), 0, target.get());
			cout << "Error after iteration: " << errorAfter << endl;

			CHECK(errorAfter <= errorBefore);

		}
	}
}


