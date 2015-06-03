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
#include "CNN/ConvolutionLayerConfig.h"
#include "CNNOpenCL/ConvolutionLayer.h"
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

	INFO("Initializing the CNN config");
	unique_ptr<CNNConfig> config(new CNNConfig(dataDescriptions));

	ATMLActivationFunction activationFunction;
	INFO("Creating the layers config");
	for (int i = 0; i < layerCount; i++)
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
			if (i == (layerCount - 2) && activationFunction == ATMLLinearActivation)
				activationFunction = ATMLTanhActivation;

		auto temp = dimensionGenerator(mt);
		if (useSoftMax)
		{
			if (i == (layerCount - 1))
			{
				activationFunction = ATMLSoftMaxActivation;
				temp = temp > 1 ? temp : 2;
			}
		}

		unique_ptr<PerceptronLayerConfig> perceptronConfig(new PerceptronLayerConfig(temp, activationFunction));
		config->AddToBack(move(perceptronConfig));
	}

	if (useSoftMax)
	{
		unique_ptr<StandardOutputLayerConfig> outputConfig(new StandardOutputLayerConfig(ATMLCrossEntropy));
		config->SetOutputConfig(move(outputConfig));
	}
	else
	{
		unique_ptr<StandardOutputLayerConfig> outputConfig(new StandardOutputLayerConfig());
		config->SetOutputConfig(move(outputConfig));
	}

	return move(config);
}

unique_ptr<CNNConfig> CreateRandomCNNPerceptronConfigWithImage(mt19937& mt,
															   uniform_int_distribution<int>& layerGenerator,
															   uniform_int_distribution<int>& imageDimensionGenerator,
															   uniform_int_distribution<int>& filterDimensionGenerator,
															   uniform_int_distribution<int>& dimensionGenerator,
															   bool useSoftMax)
{
	vector<LayerDataDescription> dataDescriptions;
	LayerDataDescription dataDescription;
	dataDescription.Height = imageDimensionGenerator(mt);
	dataDescription.Width = imageDimensionGenerator(mt);
	dataDescription.Units = dimensionGenerator(mt);
	dataDescriptions.push_back(dataDescription);

	int layerCount = layerGenerator(mt);
	uniform_int_distribution<int> activationGenerator(1, 3);

	INFO("Initializing the CNN config");
	unique_ptr<CNNConfig> config(new CNNConfig(dataDescriptions));

	unique_ptr<ConvolutionLayerConfig> convConfig(new ConvolutionLayerConfig(dimensionGenerator(mt), 
		filterDimensionGenerator(mt), filterDimensionGenerator(mt)));
	config->AddToBack(move(convConfig));

	ATMLActivationFunction activationFunction;
	INFO("Creating the layers config");
	for (int i = 0; i < layerCount; i++)
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
			if (i == (layerCount - 2) && activationFunction == ATMLLinearActivation)
				activationFunction = ATMLTanhActivation;

		auto temp = dimensionGenerator(mt);
		if (useSoftMax)
		{
			if (i == (layerCount - 1))
			{
				activationFunction = ATMLSoftMaxActivation;
				temp = temp > 1 ? temp : 2;
			}
		}

		unique_ptr<PerceptronLayerConfig> perceptronConfig(new PerceptronLayerConfig(temp, activationFunction));
		config->AddToBack(move(perceptronConfig));
	}

	if (useSoftMax)
	{
		unique_ptr<StandardOutputLayerConfig> outputConfig(new StandardOutputLayerConfig(ATMLCrossEntropy));
		config->SetOutputConfig(move(outputConfig));
	}
	else
	{
		unique_ptr<StandardOutputLayerConfig> outputConfig(new StandardOutputLayerConfig());
		config->SetOutputConfig(move(outputConfig));
	}

	return move(config);
}

SCENARIO("Back propagating a perceptron where the input is an image")
{
	auto platformInfos = OpenCLHelper::GetPlatformInfos();
	random_device device;
	mt19937 mt(device());
	uniform_int_distribution<int> dimensionGenerator(2, 15);
	uniform_int_distribution<int> layerGenerator(1, 4);
	uniform_int_distribution<int> filterGenerator(1, 30);
	uniform_int_distribution<int> imageDimensionGenerator(30, 100);

	WHEN("Back propagating a perceptron layer")
	{
		THEN("The result must equal the manually calculated layer")
		{
			for (int dummy = 0; dummy < 10; dummy++)
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
					auto config = CreateRandomCNNPerceptronConfigWithImage(mt, layerGenerator, imageDimensionGenerator, filterGenerator, dimensionGenerator, true);
					CNNOpenCL<double> network(deviceInfo, move(config));
					auto layers = network.GetLayers();
					auto layerCount = layers.size();

					ConvolutionLayer<double>* convLayer = dynamic_cast<ConvolutionLayer<double>*>(layers[0]);
					StandardOutputLayer<double>* outputLayer = dynamic_cast<StandardOutputLayer<double>*>(network.GetOutputLayer());
					vector<PerceptronLayer<double>*> perceptronlayers;
					for (int i = 1; i < layerCount; i++)
						perceptronlayers.push_back(dynamic_cast<PerceptronLayer<double>*>(layers[i]));

					auto filters = convLayer->GetFilters();
					auto convBiases = convLayer->GetBiases();

					vector<Matrix<double>> weights;
					vector<Matrix<double>> biases;
					vector<ATMLActivationFunction> activationFunctions;
					activationFunctions.push_back(convLayer->GetConfig().ActivationFunction());

					LayerDataDescription inputDataDesc = network.InputForwardDataDescriptions()[0];
					LayerDataDescription outputDataDesc = network.OutputForwardDataDescriptions()[0];
					LayerDataDescription outputBackDataDesc = network.OutputBackDataDescriptions()[0];

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

					auto target = Matrix<double>::RandomUniform(outputDataDesc.Units, 1);
					target = (1.0f / target.Sum()) * target;

					INFO("Fetching all the matrices and biases from the layers");
					for (auto perceptron : perceptronlayers)
					{
						weights.push_back(perceptron->GetWeights());
						biases.push_back(perceptron->GetBias());
						activationFunctions.push_back(perceptron->GetConfig().ActivationFunction());
					}

					CHECK(weights.size() == biases.size());
					CHECK((weights.size() + 1) == activationFunctions.size());

					INFO("Forward propagating the network by the use of matrices");

					vector<Matrixd> convOutput;
					LayerDataDescription outputDescription = convLayer->OutForwardPropDataDescriptions()[0];
					for (int j = 0; j < filters.size(); j++)
					{
						auto& filter = filters[j];
						Matrixd tempResult = Matrixd::Zeros(outputDescription.Height, outputDescription.Width);
						for (int k = 0; k < inputs.size(); k++)
							tempResult += inputs[k].Convolve(filter);

						tempResult += convBiases[j];
						if (convLayer->GetConfig().ActivationFunction() == ATMLSigmoidActivation)
							tempResult.Transform(&SigmoidActivationDouble);
						else if (convLayer->GetConfig().ActivationFunction() == ATMLTanhActivation)
							tempResult.Transform(&TanhActivationDouble);
						else if (convLayer->GetConfig().ActivationFunction() == ATMLSoftMaxActivation)
							throw runtime_error("Invalid activation in the test");

						convOutput.push_back(tempResult);
					}

					Matrixd perceptronInput = convOutput[0].Reshape(convOutput[0].ElementCount(), 1);
					for (int i = 1; i < convOutput.size(); i++)
						perceptronInput = perceptronInput.AppendDown(convOutput[i].Reshape(convOutput[i].ElementCount(), 1));

					Matrixd result = perceptronInput;

					vector<Matrix<double>> intermediatePerceptronInputs;
					intermediatePerceptronInputs.push_back(perceptronInput);
					for (int i = 0; i < weights.size(); i++)
					{

						result = weights[i] * result + biases[i];

						if (activationFunctions[i + 1] == ATMLSigmoidActivation)
							result.Transform(&SigmoidActivationDouble);
						else if (activationFunctions[i + 1] == ATMLTanhActivation)
							result.Transform(&TanhActivationDouble);
						else if (activationFunctions[i + 1] == ATMLSoftMaxActivation)
						{
							if (i != (weights.size() - 1))
								throw runtime_error("Not supported by the test");

							result.Transform([](double x) { return exp(x); });
							auto resultSum = result.Sum();
							result = (1.0f / resultSum) * result;
						}

						intermediatePerceptronInputs.push_back(result);
					}


					INFO("Making sure the manually calculated perceptron corresponds to the output dimension of the network");
					CHECK(outputDataDesc.Units == result.RowCount());
					CHECK(outputDataDesc.Width == 1);
					CHECK(outputDataDesc.Height == 1);

					INFO("Forward propagating the network");
					unique_ptr<double[]> outputPointer = move(network.FeedForwardUnaligned(rawInputs.get(), 0));

					INFO("Comparing the manually calculated perceptron with the OCL version");
					for (int i = 0; i < outputDataDesc.Units; i++)
					{
						float absDifference;
						if (abs(result.At(i, 0))  > 1E-8)
							absDifference = abs((outputPointer[i] - result.At(i, 0)) / result.At(i, 0));
						else
							absDifference = abs(outputPointer[i] - result.At(i, 0));
						cout << "Forward Difference: " << absDifference << endl;
						CHECK(absDifference < 1E-3);
					}


					INFO("Back propagating the network");
					unique_ptr<double[]> backOutputPointer = network.BackPropUnaligned(rawInputs.get(), 0, target.Data);

					vector<Matrixd> oclBackResult;
					for (int i = 0; i < outputBackDataDesc.Units; i++)
					{
						Matrixd tempMatrix(outputBackDataDesc.Height, outputBackDataDesc.Width);
						memcpy(tempMatrix.Data, backOutputPointer.get() + i * outputBackDataDesc.Height * outputBackDataDesc.Width,
							outputBackDataDesc.Height * outputBackDataDesc.Width * sizeof(double));
						//cout << "OCl result: \n" << tempMatrix.GetString() << endl;
						oclBackResult.push_back(tempMatrix);
					}

					INFO("Calculating the manually back-propagated network"); //We know that we use softmax here in the last layer.
					Matrixd perceptronDelta;
					if (activationFunctions[activationFunctions.size() - 1] == ATMLSoftMaxActivation)
						perceptronDelta = result - target;
					else if (activationFunctions[activationFunctions.size() - 1] == ATMLSigmoidActivation)
					{
						Matrixd temp = result;
						temp.Transform(&SigmoidActivationDerivativeDouble);
						perceptronDelta = (result - target) % temp;

					}
					else if (activationFunctions[activationFunctions.size() - 1] == ATMLTanhActivation)
					{
						Matrixd temp = result;
						temp.Transform(&TanhActivationDerivativeDouble);
						perceptronDelta = (result - target) % temp;
					}
					else
						perceptronDelta = result - target;

					//cout << "Perceptron delta: \n" << perceptronDelta.GetString() << endl;

					for (int i = weights.size() - 1; i >= 0; i--)
					{
						auto& input = intermediatePerceptronInputs[i];
						auto weight = weights[i].Transpose();

						perceptronDelta = (weight * perceptronDelta);
						if (activationFunctions[i] == ATMLSigmoidActivation)
						{
							input.Transform(&SigmoidActivationDerivativeDouble);
							perceptronDelta = perceptronDelta % input;
						}
						else if (activationFunctions[i] == ATMLTanhActivation)
						{
							input.Transform(&TanhActivationDerivativeDouble);
							perceptronDelta = perceptronDelta % input;
						}

						//cout << "Perceptron delta: \n" << perceptronDelta.GetString() << endl;
					}

					CHECK(perceptronDelta.ColumnCount() == 1);

					INFO("Reshaping the perceptron delta into images");
					vector<Matrixd> manualOutputDeltas;
					for (int i = 0; i < outputBackDataDesc.Units; i++)
						manualOutputDeltas.push_back(perceptronDelta.GetSubMatrix(i * outputBackDataDesc.Width * outputBackDataDesc.Height,
						0, outputBackDataDesc.Width * outputBackDataDesc.Height, 1).Reshape(outputBackDataDesc.Height, outputBackDataDesc.Width));

					CHECK(manualOutputDeltas.size() == oclBackResult.size());

					for (int i = 0; i < manualOutputDeltas.size(); i++)
					{
						//cout << "Manual result: \n" << manualOutputDeltas[i].GetString() << endl;
						auto difference = (manualOutputDeltas[i] - oclBackResult[i]).Norm2Square() / oclBackResult[i].ElementCount();
						cout << "Back Difference: " << difference << endl;
						CHECK(difference < 1E-11);
					}
				}
			}
		}
	}
}

SCENARIO("Back propagating a perceptron using Softmax")
{
	auto platformInfos = OpenCLHelper::GetPlatformInfos();
	random_device device;
	mt19937 mt(device());
	uniform_int_distribution<int> dimensionGenerator(2, 100);
	uniform_int_distribution<int> layerGenerator(1, 8);

	WHEN("Back propagating a perceptron layer")
	{
		THEN("The result must equal the manually calculated layer")
		{
			for (int dummy = 0; dummy < 10; dummy++)
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
					auto config = CreateRandomCNNPerceptronConfig(mt, layerGenerator, dimensionGenerator, true);
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
					auto target = Matrix<double>::RandomUniform(outputDataDesc.Units, 1);
					target = (1.0f / target.Sum()) * target;

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
						{
							if (i != (weights.size() - 1))
								throw runtime_error("Not supported by the test");

							result.Transform([](double x) { return exp(x); });
							auto resultSum = result.Sum();
							result = (1.0f / resultSum) * result;
						}

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

					INFO("Calculating the manually back-propagated network"); //We know that we use softmax here in the last layer.
					auto delta = result - target;

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
			for (int dummy = 0; dummy < 10; dummy++)
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
					auto config = CreateRandomCNNPerceptronConfig(mt, layerGenerator, dimensionGenerator, false);
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





