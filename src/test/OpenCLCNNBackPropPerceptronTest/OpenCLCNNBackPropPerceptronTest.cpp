/*
 * OpenCLCNNBackPropPerceptronTest.cpp
 *
 *  Created on: May 16, 2015
 *      Author: Mikael
 */
#define CATCH_CONFIG_MAIN
#include "catch/catch.hpp"
#include "OpenCLHelper/OpenCLHelper.h"
#include "CNNOpenCL/CNNOpenCL.h"
#include "CNNOpenCL/PerceptronLayer.h"
#include "CNN/PerceptronLayerConfig.h"
#include "CNN/StandardOutputLayerConfig.h"
#include "Math/Matrix.h"
#include <memory>
#include <random>
#include <type_traits>

using namespace std;
using namespace ATML::MachineLearning;
using namespace ATML::Math;
using namespace ATML::Helper;

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

SCENARIO("Back propagating a perceptron using MSE")
{
	auto platformInfos = OpenCLHelper::GetPlatformInfos();
	random_device device;
	mt19937 mt(device());
	uniform_int_distribution<int> dimensionGenerator(1, 100);
	uniform_int_distribution<int> layerGenerator(1, 8);

	int iterations = 0;

	WHEN("Back propagating a perceptron layer")
	{
		THEN("The result must equal the manually calculated layer")
		{
			for (int dummy = 0; dummy < 50; dummy++)
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
					auto layers = network.GetLayers();
					vector<PerceptronLayer<double>*> perceptronlayers;
					for (auto layer : layers)
						perceptronlayers.push_back(dynamic_cast<PerceptronLayer<double>*>(layer));

					vector<Matrix<double>> weights;
					vector<Matrix<double>> biases;
					vector<ATMLActivationFunction> activationFunctions;
					LayerDataDescription inputDataDesc = network.InputForwardDataDescriptions()[0];
					LayerDataDescription outputDataDesc = network.OutputForwardDataDescriptions()[0];

					auto input = Matrix<double>::RandomNormal(inputDataDesc.Units, 1, 0, 4);
					auto target = Matrix<double>::RandomNormal(outputDataDesc.Units, 1, 0, 4);

					INFO("Fetching all the matrices and biases from the layers");
					for (auto perceptron : perceptronlayers)
					{
						weights.push_back(perceptron->GetWeights());
						biases.push_back(perceptron->GetBias());
						activationFunctions.push_back(perceptron->GetConfig().ActivationFunction());
					}

					CHECK(weights.size() == biases.size());
					CHECK(weights.size() == activationFunctions.size());

					INFO("Forward propagating the network by the use of matrices");
					Matrix<double> result = input;
					vector<Matrix<double>> inputs;
					inputs.push_back(result);
					for (int i = 0; i < weights.size(); i++)
					{

						result = weights[i] * result + biases[i];

						if (activationFunctions[i] == ATMLSigmoidActivation)
							result.Transform(&SigmoidActivationDouble);
						else if (activationFunctions[i] == ATMLTanhActivation)
							result.Transform(&TanhActivationDouble);
						else if (activationFunctions[i] == ATMLSoftMaxActivation)
							throw runtime_error("Invalid activation in the test");

						inputs.push_back(result);
					}


					INFO("Making sure the manually calculated perceptron corresponds to the output dimension of the network");
					CHECK(outputDataDesc.Units == result.RowCount());
					CHECK(outputDataDesc.Width == 1);
					CHECK(outputDataDesc.Height == 1);

					INFO("Forward propagating the network");
					unique_ptr<double[]> outputPointer = move(network.FeedForwardUnaligned(input.Data, 0));


					INFO("Comparing the manually calculated perceptron with the OCL version");
					for (int i = 0; i < outputDataDesc.Units; i++)
					{
						float absDifference;
						if (abs(result.At(i, 0))  > 1E-8)
							absDifference = abs((outputPointer[i] - result.At(i, 0)) / result.At(i, 0));
						else
							absDifference = abs(outputPointer[i] - result.At(i, 0));
						CHECK(absDifference < 1E-3);
					}

					INFO("Back propagating the network");
					unique_ptr<double[]> backOutputPointer = move(network.BackPropUnaligned(input.Data, 0, target.Data));

					INFO("Calculating the manually back-propagated network");
					auto delta = result - target;

					if (activationFunctions[activationFunctions.size() - 1] == ATMLSigmoidActivation)
					{
						result.Transform(&SigmoidActivationDerivativeFloat);
						delta = delta % result;
					}
					else if (activationFunctions[activationFunctions.size() - 1] == ATMLTanhActivation)
					{
						result.Transform(&TanhActivationDerivativeFloat);
						delta = delta % result;
					}

					for (int i = activationFunctions.size() - 1; i >= 1; i--)
					{
						auto& input = inputs[i];
						auto weight = weights[i].Transpose();
						delta = (weight * delta);
						if (activationFunctions[i - 1] == ATMLSigmoidActivation)
						{
							input.Transform(&SigmoidActivationDerivativeDouble);
							delta = delta % input;
						}
						else if (activationFunctions[i - 1] == ATMLTanhActivation)
						{
							input.Transform(&TanhActivationDerivativeDouble);
							delta = delta % input;
						}
					}

					CHECK(delta.ColumnCount() == 1);

					Matrix<double> networkOutputMatrix(delta.RowCount(), delta.ColumnCount(), backOutputPointer.get());

					INFO("Comparing the manual back propagation with the OCL propagation");
					auto norm = (delta - networkOutputMatrix).Norm2() / delta.RowCount();
					cout << "Norm: " << norm << endl;
					CHECK(norm < 1E-5);
				}
			}
		}
	}
}





