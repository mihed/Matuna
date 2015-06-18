/*
 * OCLConvNetForwardConvolutionTest.cpp
 *
 *  Created on: May 25, 2015
 *      Author: Mikael
 */

#define CATCH_CONFIG_MAIN
#include "catch/catch.hpp"
#include "Matuna.OCLHelper/OCLHelper.h"
#include "Matuna.OCLConvNet/OCLConvNet.h"
#include "Matuna.OCLConvNet/ConvolutionLayer.h"
#include "Matuna.ConvNet/ConvolutionLayerConfig.h"
#include "Matuna.ConvNet/StandardOutputLayerConfig.h"
#include "Matuna.Math/Matrix.h"
#include <memory>
#include <random>
#include <type_traits>

using namespace std;
using namespace Matuna::MachineLearning;
using namespace Matuna::Math;
using namespace Matuna::Helper;

float SigmoidActivationFloat(float x)
{
	return 1 / (1 + exp(-x));
}

double SigmoidActivationDouble(double x)
{
	return 1 / (1 + exp(-x));
}

float TanhActivationFloat(float x)
{
	return 1.7159f *  tanh(0.6666666f * x);
}

double TanhActivationDouble(double x)
{
	return 1.7159 *  tanh(0.666666666666666 * x);
}

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
		unique_ptr<ForwardBackPropLayerConfig> convConfig(new ConvolutionLayerConfig(
			unitGenerator(mt), filterGenerator(mt), filterGenerator(mt),
			activationFunction));
		config->AddToBack(move(convConfig));
	}

	unique_ptr<StandardOutputLayerConfig> outputConfig(new StandardOutputLayerConfig());
	config->SetOutputConfig(move(outputConfig));

	return move(config);
}


SCENARIO("Forward propagating a convolution layer in an OCLConvNet")
{
	auto platformInfos = OCLHelper::GetPlatformInfos();
	random_device device;
	mt19937 mt(device());
	uniform_int_distribution<int> dimensionGenerator(80, 400);
	uniform_int_distribution<int> filterGenerator(1, 20);
	uniform_int_distribution<int> unitGenerator(1, 16);
	uniform_int_distribution<int> layerGenerator(1, 4);

	for (int dummy = 0; dummy < 10; dummy++)
	{

		vector<vector<OCLDeviceInfo>> deviceInfos;
		for (auto platformInfo : platformInfos)
			deviceInfos.push_back(OCLHelper::GetDeviceInfos(platformInfo));

		for (auto& deviceInfo : deviceInfos)
		{

			unique_ptr<ConvNetConfig> config = CreateRandomConvNetConvolutionConfig(mt, layerGenerator, dimensionGenerator, unitGenerator, filterGenerator);
			OCLConvNet<float> network(deviceInfo, move(config));

			LayerDataDescription inputDescription = network.InputForwardDataDescriptions()[0];
			LayerDataDescription outputDescription = network.OutputForwardDataDescriptions()[0];
			int inputUnits = inputDescription.Units;
			int inputHeight = inputDescription.Height;
			int inputWidth = inputDescription.Width;

			vector<Matrixf> inputs;
			unique_ptr<float[]> rawInputs(new float[inputDescription.TotalUnits()]);
			for (int i = 0; i < inputUnits; i++)
			{
				inputs.push_back(Matrixf::RandomNormal(inputHeight, inputWidth));
				memcpy(rawInputs.get() + i * inputHeight * inputWidth, inputs[i].Data, sizeof(float) * inputHeight * inputWidth);
			}

			auto tempResult = network.FeedForwardUnaligned(rawInputs.get(), 0);

			vector<Matrixf> oclResult;
			for (int i = 0; i < outputDescription.Units; i++)
			{
				Matrixf tempMatrix(outputDescription.Height, outputDescription.Width);
				memcpy(tempMatrix.Data, tempResult.get() + i * outputDescription.Height * outputDescription.Width, outputDescription.Height * outputDescription.Width * sizeof(float));
				oclResult.push_back(tempMatrix);
			}

			INFO("Fetching the convolution layers");
			auto layers = network.GetLayers();
			vector<ConvolutionLayer<float>*> convLayers;
			for (auto layer : layers)
				convLayers.push_back(dynamic_cast<ConvolutionLayer<float>*>(layer));

			vector<vector<Matrixf>> filters;
			vector<vector<float>> biases;
			vector<MatunaActivationFunction> activationFunctions;
			for (auto convLayer : convLayers)
			{
				filters.push_back(convLayer->GetFilters());
				biases.push_back(convLayer->GetBiases());
				activationFunctions.push_back(convLayer->GetConfig().ActivationFunction());
			}

			auto count = filters.size();
			auto tempInputs = inputs;
			INFO("Manually calculating the network");
			for (size_t i = 0; i < count; i++)
			{
				auto& tempFilters = filters[i];
				auto& tempBiases = biases[i];
				vector<Matrixf> nextInputs;
				LayerDataDescription outputDescription = convLayers[i]->OutForwardPropDataDescriptions()[0];
				CHECK(tempFilters.size() == tempBiases.size());
				for (size_t j = 0; j < tempFilters.size(); j++)
				{
					auto& filter = tempFilters[j];
					Matrixf tempResult = Matrixf::Zeros(outputDescription.Height, outputDescription.Width);
					for (size_t k = 0; k < tempInputs.size(); k++)
						tempResult += tempInputs[k].Convolve(filter);

					tempResult += tempBiases[j];
					if (activationFunctions[i] == MatunaSigmoidActivation)
						tempResult.Transform(&SigmoidActivationFloat);
					else if (activationFunctions[i] == MatunaTanhActivation)
						tempResult.Transform(&TanhActivationFloat);
					else if (activationFunctions[i] == MatunaSoftMaxActivation)
						throw runtime_error("Invalid activation in the test");

					nextInputs.push_back(tempResult);
				}

				tempInputs = nextInputs;
			}

			CHECK(tempInputs.size() == oclResult.size());
			for (size_t i = 0; i < tempInputs.size(); i++)
			{
				auto difference = (tempInputs[i] - oclResult[i]).Norm2Square() / tempInputs.size();
				cout << "Difference " << difference << endl;
				CHECK(difference < 1E-6f);
			}
		}
	}
}




