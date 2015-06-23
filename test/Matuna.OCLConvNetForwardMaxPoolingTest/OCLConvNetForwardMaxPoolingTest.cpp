/*
* OCLConvNetForwardMaxPoolingTest.cpp
*
*  Created on: Jun 23, 2015
*      Author: Mikael
*/

#define CATCH_CONFIG_MAIN
#include "catch/catch.hpp"
#include "Matuna.OCLHelper/OCLHelper.h"
#include "Matuna.OCLConvNet/OCLConvNet.h"
#include "Matuna.OCLConvNet/ConvolutionLayer.h"
#include "Matuna.ConvNet/ConvolutionLayerConfig.h"
#include "Matuna.ConvNet/StandardOutputLayerConfig.h"
#include "Matuna.ConvNet/MaxPoolingLayerConfig.h"
#include "Matuna.OCLConvNet/MaxPoolingLayer.h"
#include "Matuna.Math/Matrix.h"

#include <memory>
#include <cmath>
#include <random>
#include <tuple>
#include <type_traits>

using namespace Matuna::MachineLearning;
using namespace Matuna::Helper;
using namespace Matuna::Math;



unique_ptr<ConvNetConfig> CreateRandomConvNetMaxPoolConfig(mt19937& mt,
														   uniform_int_distribution<int>& layerGenerator,
														   uniform_int_distribution<int>& dimensionGenerator,
														   uniform_int_distribution<int>& unitGenerator,
														   uniform_int_distribution<int>& samplingSizeGenerator,
														   bool useDivisableLayers)
{

	cout << "\n\n-------------------Network-------------------" << endl;

	int layerCount = layerGenerator(mt);

	INFO("Creating the layers config");
	int requiredWidthMultiple = 1;
	int requiredHeightMultiple = 1;
	vector<MaxPoolingLayerConfig*> maxPoolingConfigs;
	for (int i = 0; i < layerCount; i++)
	{
		cout << "----------Layer " << i << "----------------" << endl;
		int widthSamplingSize = samplingSizeGenerator(mt);
		requiredWidthMultiple *= widthSamplingSize;
		int heightSamplingSize = samplingSizeGenerator(mt);
		requiredHeightMultiple *= heightSamplingSize;
		maxPoolingConfigs.push_back(new MaxPoolingLayerConfig(widthSamplingSize, heightSamplingSize));

		cout << "Width Sampling: " << widthSamplingSize << " Height Sampling: " << heightSamplingSize << endl;
	}

	//Let us now find the dimension that is closest to the multiples
	int width = dimensionGenerator(mt);
	int height = dimensionGenerator(mt);
	int units = unitGenerator(mt);

	if (useDivisableLayers)
	{
		width = static_cast<int>(ceil(double(width) / requiredWidthMultiple) * requiredWidthMultiple);
		height = static_cast<int>(ceil(double(height) / requiredHeightMultiple) * requiredHeightMultiple);
		CHECK((width % requiredWidthMultiple) == 0);
		CHECK((height % requiredHeightMultiple) == 0);
	}

	vector<LayerDataDescription> dataDescriptions;
	LayerDataDescription dataDescription;

	dataDescription.Width = width;
	dataDescription.Height = height;
	dataDescription.Units = units;

	cout << endl << "Width: " << width << " Height: " << height << " Units: " << units << endl << endl;

	dataDescriptions.push_back(dataDescription);
	unique_ptr<ConvNetConfig> config(new ConvNetConfig(dataDescriptions));

	for(auto vanillaConfig : maxPoolingConfigs)
		config->AddToBack(unique_ptr<ForwardBackPropLayerConfig>(vanillaConfig));

	unique_ptr<StandardOutputLayerConfig> outputConfig(new StandardOutputLayerConfig());
	config->SetOutputConfig(move(outputConfig));

	return move(config);
}


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

unique_ptr<ConvNetConfig> CreateRandomConvNetMaxPoolingConvolutionConfig(mt19937& mt,
																	  uniform_int_distribution<int>& layerGenerator,
																	  uniform_int_distribution<int>& dimensionGenerator,
																	  uniform_int_distribution<int>& unitGenerator,
																	  uniform_int_distribution<int>& filterGenerator,
																	  uniform_int_distribution<int>& samplingSizeGenerator)
{

	cout << "\n\n-------------------Network-------------------" << endl;

	vector<LayerDataDescription> dataDescriptions;
	LayerDataDescription dataDescription;
	dataDescription.Height = dimensionGenerator(mt);
	dataDescription.Width = dimensionGenerator(mt);
	dataDescription.Units = unitGenerator(mt);
	dataDescriptions.push_back(dataDescription);

	cout << "Height: " << dataDescription.Height << endl;
	cout << "Width: " << dataDescription.Width << endl;
	cout << "Units: " << dataDescription.Units << endl;

	int layerCount = layerGenerator(mt);
	uniform_int_distribution<int> activationGenerator(1, 3);

	INFO("Initializing the ConvNet config");
	unique_ptr<ConvNetConfig> config(new ConvNetConfig(dataDescriptions));

	MatunaActivationFunction activationFunction;
	INFO("Creating the layers config");
	for (int i = 0; i < layerCount; i++)
	{
		cout << "----------Conv Layer " << i << "----------------" << endl;
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

		int units = unitGenerator(mt);
		int filterWidth = filterGenerator(mt);
		int filterHeight = filterGenerator(mt);

		cout << "Units: " << units << " Filter Width: " << filterWidth << " Filter height: " << filterHeight << endl;

		unique_ptr<ForwardBackPropLayerConfig> convConfig(new ConvolutionLayerConfig(
			unitGenerator(mt), filterWidth, filterHeight,
			activationFunction));
		config->AddToBack(move(convConfig));


		cout << "----------Max Pooling Layer " << i << "----------------" << endl;

		int widthSampling = samplingSizeGenerator(mt);
		int heightSampling = samplingSizeGenerator(mt);

		cout << "Height sampling: " << heightSampling << " Width Sampling: " << widthSampling << endl;

		unique_ptr<ForwardBackPropLayerConfig> vanillaConfig(new MaxPoolingLayerConfig(widthSampling, heightSampling));
		config->AddToBack(move(vanillaConfig));
	}

	cout << endl;

	unique_ptr<StandardOutputLayerConfig> outputConfig(new StandardOutputLayerConfig());
	config->SetOutputConfig(move(outputConfig));

	return move(config);
}


SCENARIO("Creating a max pooling sampling network with divisable layers")
{
	auto platformInfos = OCLHelper::GetPlatformInfos();
	random_device device;
	mt19937 mt(device());
	uniform_int_distribution<int> dimensionGenerator(100, 400);
	uniform_int_distribution<int> samplingSizeGenerator(1, 3);
	uniform_int_distribution<int> unitGenerator(1, 15);
	uniform_int_distribution<int> layerGenerator(1, 4);

	for (int dummy = 0; dummy < 10; dummy++)
	{

		vector<vector<OCLDeviceInfo>> deviceInfos;
		for (auto platformInfo : platformInfos)
			deviceInfos.push_back(OCLHelper::GetDeviceInfos(platformInfo));

		for (auto& deviceInfo : deviceInfos)
		{
			unique_ptr<ConvNetConfig> config = CreateRandomConvNetMaxPoolConfig(mt, layerGenerator, dimensionGenerator, unitGenerator, samplingSizeGenerator, true);

			OCLConvNet<float> network(deviceInfo, move(config));

			auto layers = network.GetLayers();
			size_t layersCount= layers.size(); 
			vector<tuple<int, int>> samplingSizes;
			for (auto layer : layers)
			{
				auto config = dynamic_cast<MaxPoolingLayer<float>*>(layer)->GetConfig();
				samplingSizes.push_back(make_tuple(config.SamplingSizeWidth(), config.SamplingSizeHeight()));
			}

			LayerDataDescription inForwardDesc = network.InputForwardDataDescriptions()[0];
			LayerDataDescription outForwardDesc = network.OutputForwardDataDescriptions()[0];

			vector<Matrixf> inputMatrices;
			for (int i = 0; i < inForwardDesc.Units; i++)
				inputMatrices.push_back(Matrixf::RandomNormal(inForwardDesc.Height, inForwardDesc.Width));

			unique_ptr<float[]> contiguousMemory(new float[inForwardDesc.TotalUnits()]);
			for (int i = 0; i < inForwardDesc.Units; i++)
				memcpy(contiguousMemory.get() + i * inForwardDesc.Width * inForwardDesc.Height, inputMatrices[i].Data, inForwardDesc.Width * inForwardDesc.Height * sizeof(float));

			INFO("Feed forwarding the vanilla network");
			auto networkResult = network.FeedForwardUnaligned(contiguousMemory.get(), 0);

			INFO("Calculating the manual network");
			for (size_t i = 0; i < layersCount; i++)
			{
				auto& samplingSize = samplingSizes[i];
				vector<Matrixf> resultMatrices;
				for (size_t j = 0; j < inputMatrices.size(); j++)
					resultMatrices.push_back(inputMatrices[j].MaxDownSample(get<0>(samplingSize), get<1>(samplingSize)));

				inputMatrices = resultMatrices;
			}

			for(size_t i = 0; i < inputMatrices.size(); i++)
			{
				auto inputMatrix = inputMatrices[i];
				Matrixf networkMatrix(inputMatrix.RowCount(), inputMatrix.ColumnCount(), networkResult.get() + i * inputMatrix.RowCount() * inputMatrix.ColumnCount());
				auto difference = (inputMatrix - networkMatrix).Norm2Square();
				cout << "difference: " << difference << endl;
				CHECK(difference < 1E-13f);
			}
		}

	}
}


SCENARIO("Creating a vanilla sampling network with undivisable layers")
{
	auto platformInfos = OCLHelper::GetPlatformInfos();
	random_device device;
	mt19937 mt(device());
	uniform_int_distribution<int> dimensionGenerator(1, 1000);
	uniform_int_distribution<int> samplingSizeGenerator(1, 13);
	uniform_int_distribution<int> unitGenerator(1, 15);
	uniform_int_distribution<int> layerGenerator(1, 4);

	for (int dummy = 0; dummy < 10; dummy++)
	{

		vector<vector<OCLDeviceInfo>> deviceInfos;
		for (auto platformInfo : platformInfos)
			deviceInfos.push_back(OCLHelper::GetDeviceInfos(platformInfo));

		for (auto& deviceInfo : deviceInfos)
		{
			unique_ptr<ConvNetConfig> config = CreateRandomConvNetMaxPoolConfig(mt, layerGenerator, dimensionGenerator, unitGenerator, samplingSizeGenerator, false);

			OCLConvNet<float> network(deviceInfo, move(config));

			auto layers = network.GetLayers();
			size_t layersCount= layers.size(); 
			vector<tuple<int, int>> samplingSizes;
			for (auto layer : layers)
			{
				auto config = dynamic_cast<MaxPoolingLayer<float>*>(layer)->GetConfig();
				samplingSizes.push_back(make_tuple(config.SamplingSizeWidth(), config.SamplingSizeHeight()));
			}

			LayerDataDescription inForwardDesc = network.InputForwardDataDescriptions()[0];
			LayerDataDescription outForwardDesc = network.OutputForwardDataDescriptions()[0];

			vector<Matrixf> inputMatrices;
			for (int i = 0; i < inForwardDesc.Units; i++)
				inputMatrices.push_back(Matrixf::RandomNormal(inForwardDesc.Height, inForwardDesc.Width));

			unique_ptr<float[]> contiguousMemory(new float[inForwardDesc.TotalUnits()]);
			for (int i = 0; i < inForwardDesc.Units; i++)
				memcpy(contiguousMemory.get() + i * inForwardDesc.Width * inForwardDesc.Height, inputMatrices[i].Data, inForwardDesc.Width * inForwardDesc.Height * sizeof(float));

			INFO("Feed forwarding the vanilla network");
			auto networkResult = network.FeedForwardUnaligned(contiguousMemory.get(), 0);

			INFO("Calculating the manual network");
			for (size_t i = 0; i < layersCount; i++)
			{
				auto& samplingSize = samplingSizes[i];
				vector<Matrixf> resultMatrices;
				for (size_t j = 0; j < inputMatrices.size(); j++)
					resultMatrices.push_back(inputMatrices[j].MaxDownSample(get<0>(samplingSize), get<1>(samplingSize)));

				inputMatrices = resultMatrices;
			}

			for(size_t i = 0; i < inputMatrices.size(); i++)
			{
				auto inputMatrix = inputMatrices[i];
				//cout << "Input matrix: " << endl << inputMatrix.GetString() << endl;
				Matrixf networkMatrix(inputMatrix.RowCount(), inputMatrix.ColumnCount(), networkResult.get() + i * inputMatrix.RowCount() * inputMatrix.ColumnCount());
				//cout << "Network matrix: " << endl << networkMatrix.GetString() << endl;
				auto difference = (inputMatrix - networkMatrix).Norm2Square();
				cout << "difference: " << difference << endl;
				CHECK(difference < 1E-13f);
			}
		}

	}
}

SCENARIO("Creating a max sampling network combined with convolution")
{
	auto platformInfos = OCLHelper::GetPlatformInfos();
	random_device device;
	mt19937 mt(device());
	uniform_int_distribution<int> dimensionGenerator(1600, 2000);
	uniform_int_distribution<int> filterGenerator(1, 20);
	uniform_int_distribution<int> samplingSizeGenerator(1, 3);
	uniform_int_distribution<int> unitGenerator(1, 5);
	uniform_int_distribution<int> layerGenerator(1, 3);

	for (int dummy = 0; dummy < 10; dummy++)
	{

		vector<vector<OCLDeviceInfo>> deviceInfos;
		for (auto platformInfo : platformInfos)
			deviceInfos.push_back(OCLHelper::GetDeviceInfos(platformInfo));

		for (auto& deviceInfo : deviceInfos)
		{

			unique_ptr<ConvNetConfig> config = CreateRandomConvNetMaxPoolingConvolutionConfig(mt, layerGenerator, dimensionGenerator, unitGenerator, filterGenerator, samplingSizeGenerator);
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
			vector<tuple<int, int>> samplingSizes;
			for (auto layer : layers)
			{

				auto convLayer = dynamic_cast<ConvolutionLayer<float>*>(layer);
				if (convLayer)
					convLayers.push_back(dynamic_cast<ConvolutionLayer<float>*>(layer));
				else
				{
					CHECK(dynamic_cast<MaxPoolingLayer<float>*>(layer));
					auto config = dynamic_cast<MaxPoolingLayer<float>*>(layer)->GetConfig();
					samplingSizes.push_back(make_tuple(config.SamplingSizeWidth(), config.SamplingSizeHeight()));
				}
			}

			CHECK(samplingSizes.size() == convLayers.size());

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

				auto& samplingSize = samplingSizes[i];

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

					tempResult = tempResult.MaxDownSample(get<0>(samplingSize), get<1>(samplingSize));
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