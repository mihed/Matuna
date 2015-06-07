/*
 * OCLConvNetBackConvolutionTest.cpp
 *
 *  Created on: May 29, 2015
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

float SigmoidActivationDerivativeFloat(float x)
{
	return x * (1 - x);
}

double SigmoidActivationDerivativeDouble(double x)
{
	return  x * (1 - x);
}

float TanhActivationDerivativeFloat(float x)
{
	return 0.6666666f * (1.7159f - (x * x) / 1.7159f);
}

double TanhActivationDerivativeDouble(double x)
{
	return 0.666666666666666 * (1.7159 - (x * x) / 1.7159);
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

SCENARIO("Back propagating a convolution layer in an OCLConvNet")
{
	auto platformInfos = OCLHelper::GetPlatformInfos();
	random_device device;
	mt19937 mt(device());
	uniform_int_distribution<int> dimensionGenerator(40, 400);
	uniform_int_distribution<int> filterGenerator(1, 10);
	uniform_int_distribution<int> unitGenerator(1, 20);
	uniform_int_distribution<int> layerGenerator(1, 4);

	for (int dummy = 0; dummy < 10; dummy++)
	{

		vector<vector<OCLDeviceInfo>> deviceInfos;
		for (auto platformInfo : platformInfos)
			deviceInfos.push_back(OCLHelper::GetDeviceInfos(platformInfo));

		for (auto& deviceInfo : deviceInfos)
		{

			//if (deviceInfo[0].PlatformInfo().GetString().find("Experimental") != string::npos)
			//	continue;

			unique_ptr<ConvNetConfig> config = CreateRandomConvNetConvolutionConfig(mt, layerGenerator, dimensionGenerator, unitGenerator, filterGenerator);
			OCLConvNet<float> network(deviceInfo, move(config));

			LayerDataDescription inputDescription = network.InputForwardDataDescriptions()[0];
			LayerDataDescription outputDescription = network.OutputForwardDataDescriptions()[0];
			LayerDataDescription outBackDescription = network.OutputBackDataDescriptions()[0];
			int inputUnits = inputDescription.Units;
			int inputHeight = inputDescription.Height;
			int inputWidth = inputDescription.Width;

			vector<Matrixf> inputs;
			vector<Matrixf> targets;
			unique_ptr<float[]> rawInputs(new float[inputDescription.TotalUnits()]);
			unique_ptr<float[]> rawTargets(new float[outputDescription.TotalUnits()]);
			for (int i = 0; i < inputUnits; i++)
			{
				inputs.push_back(Matrixf::RandomNormal(inputHeight, inputWidth, 0, 3));
				memcpy(rawInputs.get() + i * inputHeight * inputWidth, inputs[i].Data, sizeof(float) * inputHeight * inputWidth);
			}

			auto tempResult = network.FeedForwardUnaligned(rawInputs.get(), 0);

			vector<Matrixf> oclForwardResult;
			for (int i = 0; i < outputDescription.Units; i++)
			{
				Matrixf tempMatrix(outputDescription.Height, outputDescription.Width);
				memcpy(tempMatrix.Data, tempResult.get() + i * outputDescription.Height * outputDescription.Width, outputDescription.Height * outputDescription.Width * sizeof(float));
				oclForwardResult.push_back(tempMatrix);
				targets.push_back(Matrixf::RandomNormal(outputDescription.Height, outputDescription.Width));
				memcpy(rawTargets.get() + i * outputDescription.Height * outputDescription.Width, targets[i].Data, sizeof(float) * outputDescription.Height * outputDescription.Width);
			}
			tempResult.reset();

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

			int count = filters.size();
			CHECK(count == biases.size());
			auto tempInputs = inputs;
			vector<vector<Matrixf>> intermediateInputs;
			intermediateInputs.push_back(tempInputs);
			INFO("Manually calculating the network");
			for (int i = 0; i < count; i++)
			{
				auto& tempFilters = filters[i];
				auto& tempBiases = biases[i];
				vector<Matrixf> nextInputs;
				LayerDataDescription outputDescription = convLayers[i]->OutForwardPropDataDescriptions()[0];
				CHECK(tempFilters.size() == tempBiases.size());
				for (int j = 0; j < tempFilters.size(); j++)
				{
					auto& filter = tempFilters[j];
					Matrixf tempResult = Matrixf::Zeros(outputDescription.Height, outputDescription.Width);
					for (int k = 0; k < tempInputs.size(); k++)
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
				intermediateInputs.push_back(tempInputs);
			}

			CHECK((count + 1) == intermediateInputs.size());

			INFO("Checking the forward result");
			CHECK(tempInputs.size() == oclForwardResult.size());

			vector<Matrixf> outputDeltas;
			for (int i = 0; i < tempInputs.size(); i++)
			{
				auto differenceMatrix = tempInputs[i] - oclForwardResult[i];
				auto difference = differenceMatrix.Norm2Square() / tempInputs.size();
				cout << "Forward difference " << difference << endl;
				CHECK(difference < 1E-6f);

				Matrixf outputDelta;
				Matrixf tmpOutput = tempInputs[i];
				Matrixf targetDifference = tempInputs[i] - targets[i];
				switch (activationFunctions[activationFunctions.size() - 1])
				{
				case MatunaSigmoidActivation:
					tmpOutput.Transform(&SigmoidActivationDerivativeFloat);
					outputDelta = targetDifference % tmpOutput;
					break;
				case MatunaTanhActivation:
					tmpOutput.Transform(&TanhActivationDerivativeFloat);
					outputDelta = targetDifference % tmpOutput;
					break;
				case MatunaLinearActivation:
					outputDelta = targetDifference;
					break;
				default:
					throw runtime_error("not implemented");
					break;
				}

				outputDeltas.push_back(outputDelta);
			}



			INFO("Back propagating the targets through the OCL network");
			auto rawOCLBackProp = network.BackPropUnaligned(rawInputs.get(), 0, rawTargets.get());

			vector<Matrixf> oclBackResult;
			for (int i = 0; i < outBackDescription.Units; i++)
			{
				Matrixf tempMatrix(outBackDescription.Height, outBackDescription.Width);
				memcpy(tempMatrix.Data, rawOCLBackProp.get() + i * outBackDescription.Height * outBackDescription.Width,
					outBackDescription.Height * outBackDescription.Width * sizeof(float));
				//cout << tempMatrix.GetString() << endl;
				oclBackResult.push_back(tempMatrix);
			}

			CHECK(convLayers.size() == filters.size());
			CHECK(convLayers.size() == activationFunctions.size());
			CHECK((convLayers.size() + 1) == intermediateInputs.size());
			for (int i = convLayers.size() - 1; i >= 1; i--)
			{
				vector<Matrixf> tempOutputs;
				auto& tempFilters = filters[i];
				auto convLayer = convLayers[i];
				LayerDataDescription dataDesc = convLayer->InForwardPropDataDescriptions()[0];
				auto& tempInputs = intermediateInputs[i];
				auto activationFunction = activationFunctions[i - 1];
				CHECK(tempInputs.size() == dataDesc.Units);
				CHECK(tempFilters.size() == outputDeltas.size());
				for (auto& input : tempInputs)
				{
					Matrixf tempOutput = Matrixf::Zeros(dataDesc.Height, dataDesc.Width);
					for (int j = 0; j < tempFilters.size(); j++)
					{
						auto& filter = tempFilters[j];
						tempOutput += outputDeltas[j].AddZeroBorder(filter.ColumnCount() - 1,
							filter.ColumnCount() - 1, filter.RowCount() - 1, filter.RowCount() - 1).Convolve(filter.Rotate180());
					}
					
					Matrixf tempInput = input;
					switch (activationFunction)
					{
					case MatunaSigmoidActivation:
						tempInput.Transform(&SigmoidActivationDerivativeFloat);
						tempOutputs.push_back(tempOutput % tempInput);
						break;
					case MatunaTanhActivation:
						tempInput.Transform(&TanhActivationDerivativeFloat);
						tempOutputs.push_back(tempOutput % tempInput);
						break;
					case MatunaLinearActivation:
						tempOutputs.push_back(tempOutput);
						break;
					default:
						throw runtime_error("not implemented");
						break;
					}
				}

				outputDeltas = tempOutputs;
			}

			CHECK(oclBackResult.size() == outputDeltas.size());
			for (int i = 0; i < outputDeltas.size(); i++)
			{
				//cout << oclBackResult[i].GetString() << endl;
				//cout << outputDeltas[i].GetString() << endl;
				auto difference = (oclBackResult[i] - outputDeltas[i]).Norm2Square() / outputDeltas[i].ElementCount();
				cout << "Back Difference " << difference << endl;
				CHECK(difference < 1E-5f);
			}
		}
	}
}

