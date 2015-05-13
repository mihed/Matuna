/*
 * OpenCLCNNPerceptronTest.cpp
 *
 *  Created on: May 11, 2015
 *      Author: Mikael
 */

#define CATCH_CONFIG_MAIN
#include "catch/catch.hpp"
#include "OpenCLHelper/OpenCLDeviceHandler.h"
#include "CNNOpenCL/CNNOpenCL.h"
#include "CNNOpenCL/PerceptronLayer.h"
#include "CNN/PerceptronLayerConfig.h"
#include "CNN/StandardOutputLayerConfig.h"
#include "Math/Matrix.h"
#include <memory>
#include <random>

using namespace std;
using namespace ATML::MachineLearning;
using namespace ATML::Math;
using namespace ATML::Helper;

/*
template<class T>
vector<vector<T>> GetSimplePerceptronInputs()
{
vector<vector<T>> inputs;
vector<T> input1;
input1.push_back(0);
input1.push_back(0);
vector<T> input2;
input2.push_back(1);
input2.push_back(0);
vector<T> input3;
input3.push_back(0);
input3.push_back(1);
vector<T> input4;
input4.push_back(1);
input4.push_back(1);
inputs.push_back(input1);
inputs.push_back(input2);
inputs.push_back(input3);
inputs.push_back(input4);

return inputs;
}

template<class T>
void CalculateORPerceptron(unique_ptr<CNNConfig> config, unique_ptr<OpenCLContext> contextPointer,
unique_ptr<PerceptronLayerConfig> perceptronConfig,
vector<vector<T>> inputs, vector<T> targets)
{
unique_ptr<StandardOutputLayerConfig> outputLayerConfig(new StandardOutputLayerConfig());

OpenCLPlatformInfo platformInfo = contextPointer->GetPlatformInfo();

INFO("Pushng the perceptron config to the cnn config");
config->AddToBack(move(perceptronConfig));
config->SetOutputConfig(move(outputLayerConfig));

INFO("Initializing the network");
CNNOpenCL<T> network(move(contextPointer), move(config));
CHECK(network.Interlocked());

INFO("Creating a pointer with previusly calculated parameters");
//We know that the weight matrix is located in memory like an image buffer, with the bias value directly after.
unique_ptr<T[]> parameters(new T[3]);
parameters[0] = 1.3860e03f;
parameters[1] = 776.3274f;
parameters[2] = -397.6140f;

INFO("Setting the network parameters to the previosly calculated OR parameters");
network.SetParameters(parameters.get());

CHECK(network.GetParameterCount() == 3);

INFO("Making sure the read parameters correspond to the set parameters");
unique_ptr<T[]> readParameters(new T[network.GetParameterCount()]);
network.GetParameters(readParameters.get());

for (int i = 0; i < 3; i++)
{
CHECK(readParameters[i] == parameters[i]);
}

CHECK(targets.size() == inputs.size());
for (int i = 0; i < targets.size(); i++)
{
T output;
network.FeedForward(inputs[i].data(), 0, &output);
auto difference = abs(targets[i] - output);
if (difference >= 0.01)
{
cout << "The failed platform: " << endl << platformInfo.GetString().c_str() << endl;
}
REQUIRE(difference < 0.01);
}
}


template<class T>
void CalculateANDPerceptron(unique_ptr<CNNConfig> config, unique_ptr<OpenCLContext> contextPointer,
unique_ptr<PerceptronLayerConfig> perceptronConfig,
vector<vector<T>> inputs, vector<T> targets)
{
unique_ptr<StandardOutputLayerConfig> outputLayerConfig(new StandardOutputLayerConfig());

INFO("Pushng the perceptron config to the cnn config");
config->AddToBack(move(perceptronConfig));
config->SetOutputConfig(move(outputLayerConfig));

OpenCLPlatformInfo platformInfo = contextPointer->GetPlatformInfo();

INFO("Initializing the network");
CNNOpenCL<T> network(move(contextPointer), move(config));
CHECK(network.Interlocked());

INFO("Creating a pointer with previusly calculated parameters");
//We know that the weight matrix is located in memory like an image buffer, with the bias value directly after.
unique_ptr<T[]> parameters(new T[3]);
parameters[0] = 1.5483e03f;
parameters[1] = 1.5965e03f;
parameters[2] = -2.0258e03f;

INFO("Setting the network parameters to the previosly calculated AND parameters");
network.SetParameters(parameters.get());

CHECK(network.GetParameterCount() == 3);

INFO("Making sure the read parameters correspond to the set parameters");
unique_ptr<T[]> readParameters(new T[network.GetParameterCount()]);
network.GetParameters(readParameters.get());

for (int i = 0; i < 3; i++)
{
CHECK(readParameters[i] == parameters[i]);
}

CHECK(targets.size() == inputs.size());
for (int i = 0; i < targets.size(); i++)
{
T output;
network.FeedForward(inputs[i].data(), 0, &output);
auto difference = abs(targets[i] - output);
if (difference >= 0.01)
{
cout << "The failed platform: " << endl << platformInfo.GetString().c_str() << endl;
}
REQUIRE(difference < 0.01);
}
}

unique_ptr<OpenCLContext> GetDoubleCapableContext(const OpenCLPlatformInfo& platfomInfo)
{
auto deviceInfos = OpenCLDeviceHandler::GetDeviceInfos(platfomInfo);
vector<OpenCLDeviceInfo> capabaleDevices;
for (auto& deviceInfo : deviceInfos)
if (deviceInfo.PreferredDoubleVectorWidth() != 0)
capabaleDevices.push_back(deviceInfo);

vector<tuple<OpenCLDeviceConfig, OpenCLDeviceInfo>> configAndInfos;
for (auto& deviceInfo : capabaleDevices)
{
OpenCLDeviceConfig config;
config.AddCommandQueue();
configAndInfos.push_back(make_tuple(config, deviceInfo));
}

return OpenCLDeviceHandler::GetContext(platfomInfo, configAndInfos);
}



SCENARIO("Forward propagating an OR CNN network")
{

INFO("Initializing the data descriptions for the OR perceptron");
vector<LayerDataDescription> dataDescriptions;
LayerDataDescription dataDescription;
dataDescription.Height = 1;
dataDescription.Width = 1;
dataDescription.Units = 2;
dataDescriptions.push_back(dataDescription);

auto inputsFloat = GetSimplePerceptronInputs<cl_float>();
vector<cl_float> targetsFloat;
targetsFloat.push_back(0);
targetsFloat.push_back(1);
targetsFloat.push_back(1);
targetsFloat.push_back(1);

auto inputsDouble = GetSimplePerceptronInputs<cl_double>();
vector<cl_double> targetsDouble;
targetsDouble.push_back(0);
targetsDouble.push_back(1);
targetsDouble.push_back(1);
targetsDouble.push_back(1);


WHEN("Creating an OR network with float, standard precision and no relaxed math")
{
INFO("Creating the contexts");
auto platformInfos = OpenCLDeviceHandler::GetPlatformInfos();

vector<unique_ptr<OpenCLContext>> contexts;
for (auto platformInfo : platformInfos)
contexts.push_back(move(OpenCLDeviceHandler::GetContext(platformInfo)));

THEN("We must have correct OR output with the known parameters")
{
INFO("For every context we create a network");
for (auto& contextPointer : contexts)
{
INFO("Initializing the CNN config");
unique_ptr<CNNConfig> config(new CNNConfig(dataDescriptions));

INFO("Creating a perceptron layer config");
unique_ptr<PerceptronLayerConfig> perceptronConfig(new PerceptronLayerConfig(1));
CalculateORPerceptron<cl_float>(move(config), move(contextPointer), move(perceptronConfig), inputsFloat, targetsFloat);
}
}
}


WHEN("Creating an OR network with float, native precision and no relaxed math")
{
INFO("Creating the contexts");
auto platformInfos = OpenCLDeviceHandler::GetPlatformInfos();

vector<unique_ptr<OpenCLContext>> contexts;
for (auto platformInfo : platformInfos)
contexts.push_back(move(OpenCLDeviceHandler::GetContext(platformInfo)));

THEN("We must have correct OR output with the known parameters")
{
INFO("For every context we create a network");
for (auto& contextPointer : contexts)
{
INFO("Initializing the CNN config");
unique_ptr<CNNConfig> config(new CNNConfig(dataDescriptions));

INFO("Creating a perceptron layer config");
unique_ptr<PerceptronLayerConfig> perceptronConfig(new PerceptronLayerConfig(1, ATMLSigmoidActivation, ATMLFullConnection, false, ATMLNativePrecision));
CalculateORPerceptron<cl_float>(move(config), move(contextPointer), move(perceptronConfig), inputsFloat, targetsFloat);
}
}
}

WHEN("Creating an OR network with float, half precision and no relaxed math")
{
INFO("Creating the contexts");
auto platformInfos = OpenCLDeviceHandler::GetPlatformInfos();

vector<unique_ptr<OpenCLContext>> contexts;
for (auto platformInfo : platformInfos)
contexts.push_back(move(OpenCLDeviceHandler::GetContext(platformInfo)));

THEN("We must have correct OR output with the known parameters")
{
INFO("For every context we create a network");
for (auto& contextPointer : contexts)
{
INFO("Initializing the CNN config");
unique_ptr<CNNConfig> config(new CNNConfig(dataDescriptions));

INFO("Creating a perceptron layer config");
unique_ptr<PerceptronLayerConfig> perceptronConfig(new PerceptronLayerConfig(1, ATMLSigmoidActivation, ATMLFullConnection, false, ATMLHalfPrecision));
CalculateORPerceptron<cl_float>(move(config), move(contextPointer), move(perceptronConfig), inputsFloat, targetsFloat);
}
}
}

WHEN("Creating an OR network with float, half precision and relaxed math")
{
INFO("Creating the contexts");
auto platformInfos = OpenCLDeviceHandler::GetPlatformInfos();

vector<unique_ptr<OpenCLContext>> contexts;
for (auto platformInfo : platformInfos)
contexts.push_back(move(OpenCLDeviceHandler::GetContext(platformInfo)));

THEN("We must have correct OR output with the known parameters")
{
INFO("For every context we create a network");
for (auto& contextPointer : contexts)
{
INFO("Initializing the CNN config");
unique_ptr<CNNConfig> config(new CNNConfig(dataDescriptions));

cout << contextPointer->GetPlatformInfo().GetString().c_str() << endl;

INFO("Creating a perceptron layer config");
unique_ptr<PerceptronLayerConfig> perceptronConfig(new PerceptronLayerConfig(1, ATMLSigmoidActivation, ATMLFullConnection, true, ATMLHalfPrecision));
CalculateORPerceptron<cl_float>(move(config), move(contextPointer), move(perceptronConfig), inputsFloat, targetsFloat);
}
}
}

WHEN("Creating an OR network with double, standard precision and no relaxed math")
{
INFO("Creating the contexts");
auto platformInfos = OpenCLDeviceHandler::GetPlatformInfos();

vector<unique_ptr<OpenCLContext>> contexts;
for (auto platformInfo : platformInfos)
contexts.push_back(move(GetDoubleCapableContext(platformInfo)));

THEN("We must have correct OR output with the known parameters")
{
INFO("For every context we create a network");
for (auto& contextPointer : contexts)
{
INFO("Initializing the CNN config");
unique_ptr<CNNConfig> config(new CNNConfig(dataDescriptions));

INFO("Creating a perceptron layer config");
unique_ptr<PerceptronLayerConfig> perceptronConfig(new PerceptronLayerConfig(1));
CalculateORPerceptron<cl_double>(move(config), move(contextPointer), move(perceptronConfig), inputsDouble, targetsDouble);
}
}
}

WHEN("Creating an OR network with double, native precision and no relaxed math")
{
INFO("Creating the contexts");
auto platformInfos = OpenCLDeviceHandler::GetPlatformInfos();

vector<unique_ptr<OpenCLContext>> contexts;
for (auto platformInfo : platformInfos)
contexts.push_back(move(GetDoubleCapableContext(platformInfo)));

THEN("We must have correct OR output with the known parameters")
{
INFO("For every context we create a network");
for (auto& contextPointer : contexts)
{
INFO("Initializing the CNN config");
unique_ptr<CNNConfig> config(new CNNConfig(dataDescriptions));

INFO("Creating a perceptron layer config");
unique_ptr<PerceptronLayerConfig> perceptronConfig(new PerceptronLayerConfig(1, ATMLSigmoidActivation, ATMLFullConnection, false, ATMLNativePrecision));
CalculateORPerceptron<cl_double>(move(config), move(contextPointer), move(perceptronConfig), inputsDouble, targetsDouble);
}
}
}

WHEN("Creating an OR network with double, half precision and no relaxed math")
{
INFO("Creating the contexts");
auto platformInfos = OpenCLDeviceHandler::GetPlatformInfos();

vector<unique_ptr<OpenCLContext>> contexts;
for (auto platformInfo : platformInfos)
contexts.push_back(move(GetDoubleCapableContext(platformInfo)));

THEN("We must have correct OR output with the known parameters")
{
INFO("For every context we create a network");
for (auto& contextPointer : contexts)
{
INFO("Initializing the CNN config");
unique_ptr<CNNConfig> config(new CNNConfig(dataDescriptions));

INFO("Creating a perceptron layer config");
unique_ptr<PerceptronLayerConfig> perceptronConfig(new PerceptronLayerConfig(1, ATMLSigmoidActivation, ATMLFullConnection, false, ATMLHalfPrecision));
CalculateORPerceptron<cl_double>(move(config), move(contextPointer), move(perceptronConfig), inputsDouble, targetsDouble);
}
}
}

WHEN("Creating an OR network with double, half precision and relaxed math")
{
INFO("Creating the contexts");
auto platformInfos = OpenCLDeviceHandler::GetPlatformInfos();

vector<unique_ptr<OpenCLContext>> contexts;
for (auto platformInfo : platformInfos)
contexts.push_back(move(GetDoubleCapableContext(platformInfo)));

THEN("We must have correct OR output with the known parameters")
{
INFO("For every context we create a network");
for (auto& contextPointer : contexts)
{
INFO("Initializing the CNN config");
unique_ptr<CNNConfig> config(new CNNConfig(dataDescriptions));

INFO("Creating a perceptron layer config");
unique_ptr<PerceptronLayerConfig> perceptronConfig(new PerceptronLayerConfig(1, ATMLSigmoidActivation, ATMLFullConnection, true, ATMLHalfPrecision));
CalculateORPerceptron<cl_double>(move(config), move(contextPointer), move(perceptronConfig), inputsDouble, targetsDouble);
}
}
}
}



SCENARIO("Forward propagating an AND CNN network")
{

INFO("Initializing the data descriptions for the AND perceptron");
vector<LayerDataDescription> dataDescriptions;
LayerDataDescription dataDescription;
dataDescription.Height = 1;
dataDescription.Width = 1;
dataDescription.Units = 2;
dataDescriptions.push_back(dataDescription);

auto inputsFloat = GetSimplePerceptronInputs<cl_float>();
vector<cl_float> targetsFloat;
targetsFloat.push_back(0);
targetsFloat.push_back(0);
targetsFloat.push_back(0);
targetsFloat.push_back(1);

auto inputsDouble = GetSimplePerceptronInputs<cl_double>();
vector<cl_double> targetsDouble;
targetsDouble.push_back(0);
targetsDouble.push_back(0);
targetsDouble.push_back(0);
targetsDouble.push_back(1);


WHEN("Creating an AND network with float, standard precision and no relaxed math")
{
INFO("Creating the contexts");
auto platformInfos = OpenCLDeviceHandler::GetPlatformInfos();

vector<unique_ptr<OpenCLContext>> contexts;
for (auto platformInfo : platformInfos)
contexts.push_back(move(OpenCLDeviceHandler::GetContext(platformInfo)));

THEN("We must have correct AND output with the known parameters")
{
INFO("For every context we create a network");
for (auto& contextPointer : contexts)
{
INFO("Initializing the CNN config");
unique_ptr<CNNConfig> config(new CNNConfig(dataDescriptions));

INFO("Creating a perceptron layer config");
unique_ptr<PerceptronLayerConfig> perceptronConfig(new PerceptronLayerConfig(1));
CalculateANDPerceptron<cl_float>(move(config), move(contextPointer), move(perceptronConfig), inputsFloat, targetsFloat);
}
}
}


WHEN("Creating an AND network with float, native precision and no relaxed math")
{
INFO("Creating the contexts");
auto platformInfos = OpenCLDeviceHandler::GetPlatformInfos();

vector<unique_ptr<OpenCLContext>> contexts;
for (auto platformInfo : platformInfos)
contexts.push_back(move(OpenCLDeviceHandler::GetContext(platformInfo)));

THEN("We must have correct AND output with the known parameters")
{
INFO("For every context we create a network");
for (auto& contextPointer : contexts)
{
INFO("Initializing the CNN config");
unique_ptr<CNNConfig> config(new CNNConfig(dataDescriptions));

INFO("Creating a perceptron layer config");
unique_ptr<PerceptronLayerConfig> perceptronConfig(new PerceptronLayerConfig(1, ATMLSigmoidActivation, ATMLFullConnection, false, ATMLNativePrecision));
CalculateANDPerceptron<cl_float>(move(config), move(contextPointer), move(perceptronConfig), inputsFloat, targetsFloat);
}
}
}

WHEN("Creating an AND network with float, half precision and no relaxed math")
{
INFO("Creating the contexts");
auto platformInfos = OpenCLDeviceHandler::GetPlatformInfos();

vector<unique_ptr<OpenCLContext>> contexts;
for (auto platformInfo : platformInfos)
contexts.push_back(move(OpenCLDeviceHandler::GetContext(platformInfo)));

THEN("We must have correct AND output with the known parameters")
{
INFO("For every context we create a network");
for (auto& contextPointer : contexts)
{
INFO("Initializing the CNN config");
unique_ptr<CNNConfig> config(new CNNConfig(dataDescriptions));

INFO("Creating a perceptron layer config");
unique_ptr<PerceptronLayerConfig> perceptronConfig(new PerceptronLayerConfig(1, ATMLSigmoidActivation, ATMLFullConnection, false, ATMLHalfPrecision));
CalculateANDPerceptron<cl_float>(move(config), move(contextPointer), move(perceptronConfig), inputsFloat, targetsFloat);
}
}
}

WHEN("Creating an AND network with float, half precision and relaxed math")
{
INFO("Creating the contexts");
auto platformInfos = OpenCLDeviceHandler::GetPlatformInfos();

vector<unique_ptr<OpenCLContext>> contexts;
for (auto platformInfo : platformInfos)
contexts.push_back(move(OpenCLDeviceHandler::GetContext(platformInfo)));

THEN("We must have correct AND output with the known parameters")
{
INFO("For every context we create a network");
for (auto& contextPointer : contexts)
{
INFO("Initializing the CNN config");
unique_ptr<CNNConfig> config(new CNNConfig(dataDescriptions));

//FIXME: the AMD platform fails with the unsafe math
//if (contextPointer->GetPlatformInfo().PlatformName().find("AMD") == string::npos)
//	continue;

cout << contextPointer->GetPlatformInfo().GetString().c_str() << endl;

INFO("Creating a perceptron layer config");
unique_ptr<PerceptronLayerConfig> perceptronConfig(new PerceptronLayerConfig(1, ATMLSigmoidActivation, ATMLFullConnection, true, ATMLHalfPrecision));
CalculateANDPerceptron<cl_float>(move(config), move(contextPointer), move(perceptronConfig), inputsFloat, targetsFloat);
}
}
}

WHEN("Creating an AND network with double, standard precision and no relaxed math")
{
INFO("Creating the contexts");
auto platformInfos = OpenCLDeviceHandler::GetPlatformInfos();

vector<unique_ptr<OpenCLContext>> contexts;
for (auto platformInfo : platformInfos)
contexts.push_back(move(GetDoubleCapableContext(platformInfo)));

THEN("We must have correct AND output with the known parameters")
{
INFO("For every context we create a network");
for (auto& contextPointer : contexts)
{
INFO("Initializing the CNN config");
unique_ptr<CNNConfig> config(new CNNConfig(dataDescriptions));

INFO("Creating a perceptron layer config");
unique_ptr<PerceptronLayerConfig> perceptronConfig(new PerceptronLayerConfig(1));
CalculateANDPerceptron<cl_double>(move(config), move(contextPointer), move(perceptronConfig), inputsDouble, targetsDouble);
}
}
}

WHEN("Creating an AND network with double, native precision and no relaxed math")
{
INFO("Creating the contexts");
auto platformInfos = OpenCLDeviceHandler::GetPlatformInfos();

vector<unique_ptr<OpenCLContext>> contexts;
for (auto platformInfo : platformInfos)
contexts.push_back(move(GetDoubleCapableContext(platformInfo)));

THEN("We must have correct AND output with the known parameters")
{
INFO("For every context we create a network");
for (auto& contextPointer : contexts)
{
INFO("Initializing the CNN config");
unique_ptr<CNNConfig> config(new CNNConfig(dataDescriptions));

INFO("Creating a perceptron layer config");
unique_ptr<PerceptronLayerConfig> perceptronConfig(new PerceptronLayerConfig(1, ATMLSigmoidActivation, ATMLFullConnection, false, ATMLNativePrecision));
CalculateANDPerceptron<cl_double>(move(config), move(contextPointer), move(perceptronConfig), inputsDouble, targetsDouble);
}
}
}

WHEN("Creating an AND network with double, half precision and no relaxed math")
{
INFO("Creating the contexts");
auto platformInfos = OpenCLDeviceHandler::GetPlatformInfos();

vector<unique_ptr<OpenCLContext>> contexts;
for (auto platformInfo : platformInfos)
contexts.push_back(move(GetDoubleCapableContext(platformInfo)));

THEN("We must have correct AND output with the known parameters")
{
INFO("For every context we create a network");
for (auto& contextPointer : contexts)
{
INFO("Initializing the CNN config");
unique_ptr<CNNConfig> config(new CNNConfig(dataDescriptions));

INFO("Creating a perceptron layer config");
unique_ptr<PerceptronLayerConfig> perceptronConfig(new PerceptronLayerConfig(1, ATMLSigmoidActivation, ATMLFullConnection, false, ATMLHalfPrecision));
CalculateANDPerceptron<cl_double>(move(config), move(contextPointer), move(perceptronConfig), inputsDouble, targetsDouble);
}
}
}

WHEN("Creating an AND network with double, half precision and relaxed math")
{
INFO("Creating the contexts");
auto platformInfos = OpenCLDeviceHandler::GetPlatformInfos();

vector<unique_ptr<OpenCLContext>> contexts;
for (auto platformInfo : platformInfos)
contexts.push_back(move(GetDoubleCapableContext(platformInfo)));

THEN("We must have correct AND output with the known parameters")
{
INFO("For every context we create a network");
for (auto& contextPointer : contexts)
{
INFO("Initializing the CNN config");
unique_ptr<CNNConfig> config(new CNNConfig(dataDescriptions));

INFO("Creating a perceptron layer config");
unique_ptr<PerceptronLayerConfig> perceptronConfig(new PerceptronLayerConfig(1, ATMLSigmoidActivation, ATMLFullConnection, true, ATMLHalfPrecision));
CalculateANDPerceptron<cl_double>(move(config), move(contextPointer), move(perceptronConfig), inputsDouble, targetsDouble);
}
}
}
}
*/

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
	return 1.7159 *  tanh(0.6666666f * x);
}

double TanhActivationDouble(double x)
{
	return 1.7159 *  tanh(0.666666666666666 * x);
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

	INFO("Initializing the CNN config");
	unique_ptr<CNNConfig> config(new CNNConfig(dataDescriptions));

	INFO("Creating the layers config");
	for (int i = 0; i < layerCount; i++)
	{
		unique_ptr<PerceptronLayerConfig> perceptronConfig(new PerceptronLayerConfig(dimensionGenerator(mt), ATMLLinearActivation));
		config->AddToBack(move(perceptronConfig));
	}

	unique_ptr<StandardOutputLayerConfig> outputConfig(new StandardOutputLayerConfig());
	config->SetOutputConfig(move(outputConfig));

	return move(config);
}

SCENARIO("Forward propagating multi-layer perceptron network")
{
	auto platformInfos = OpenCLDeviceHandler::GetPlatformInfos();
	random_device device;
	mt19937 mt(device());
	uniform_int_distribution<int> dimensionGenerator(1, 100);
	uniform_int_distribution<int> layerGenerator(1, 5);

	WHEN("Forward propagating using single precision, sigmoid and standard math")
	{
		vector<unique_ptr<OpenCLContext>> contexts;
		for (auto platformInfo : platformInfos)
			contexts.push_back(move(OpenCLDeviceHandler::GetContext(platformInfo)));

		THEN("The values must match the hand-calculated values")
		{
			for (auto& contextPointer : contexts)
			{
				auto config = CreateRandomCNNPerceptronConfig(mt, layerGenerator, dimensionGenerator);
				auto device = contextPointer->GetDevices()[0];
				CNNOpenCL<float> network(move(contextPointer), move(config));
				auto layers = network.GetLayers();
				vector<PerceptronLayer<float>*> perceptronlayers;
				for (auto layer : layers)
					perceptronlayers.push_back(dynamic_cast<PerceptronLayer<float>*>(layer));

				vector<Matrix<float>> weights;
				vector<Matrix<float>> biases;
				vector<ATMLActivationFunction> activationFunctions;
				LayerDataDescription inputDataDesc = network.InputDataDescriptions()[0];

				auto input = Matrix<float>::RandomNormal(inputDataDesc.Units, 1);

				INFO("Fetching all the matrices and biases from the layers");
				for (auto perceptron : perceptronlayers)
				{
					unique_ptr<float[]> rawParameters(new float[perceptron->GetParameterCount()]);
					perceptron->GetParameters(rawParameters.get(), device, 0, true);

					auto outDataDesc = perceptron->OutForwardPropDataDescription()[0];
					Matrix<float> weightMatrix(outDataDesc.Units, inputDataDesc.Units, rawParameters.get());
					Matrix<float> biasVector(outDataDesc.Units, 1, rawParameters.get() + weightMatrix.ElementCount());
					weights.push_back(weightMatrix);
					biases.push_back(biasVector);
					activationFunctions.push_back(perceptron->GetConfig().ActivationFunction());
					inputDataDesc = outDataDesc;
				}

				CHECK(weights.size() == biases.size());
				CHECK(weights.size() == activationFunctions.size());

				INFO("Forward propagating the network by the use of matrices");
				Matrix<float> result = input;
				for (int i = 0; i < weights.size(); i++)
				{

					result = weights[i] * result + biases[i];

					if (activationFunctions[i] == ATMLSigmoidActivation)
						result.Transform(&SigmoidActivationFloat);
					else if (activationFunctions[i] == ATMLTanhActivation)
						result.Transform(&TanhActivationFloat);
					else if (activationFunctions[i] == ATMLSoftMaxActivation)
						throw runtime_error("Invalid activation in the test");
				}

				INFO("Forward propagating the network");
				LayerMemoryDescription inputMemoryDesc = network.InputMemoryDescriptions()[0];
				inputDataDesc = network.InputDataDescriptions()[0];

				INFO("Making sure the manually calculated perceptron corresponds to the output dimension of the network");
				auto outputDataDesc = network.OutputDataDescriptions()[0];
				auto outputMemoryDesc = network.OutputMemoryDescriptions()[0];
				CHECK(outputDataDesc.Units == result.RowCount());
				CHECK(outputDataDesc.Width == 1);
				CHECK(outputDataDesc.Height == 1);
				CHECK(outputMemoryDesc.Units == result.RowCount());
				CHECK(outputMemoryDesc.Width == 1);
				CHECK(outputMemoryDesc.Height == 1);
				CHECK(result.ColumnCount() == 1);

				INFO("Assuming no particular memory padding at the moment");
				CHECK(inputMemoryDesc.Units == inputDataDesc.Units);
				CHECK(inputMemoryDesc.Width == inputDataDesc.Width);
				CHECK(inputMemoryDesc.Height == inputDataDesc.Height);

				INFO("Forward propagating the network");
				unique_ptr<float[]> outputPointer(new float[outputDataDesc.Units]);
				network.FeedForward(input.Data, 0, outputPointer.get());


				INFO("Comparing the manually calculated perceptron with the OCL version");
				for (int i = 0; i < outputDataDesc.Units; i++)
				{
					auto absDifference = abs((outputPointer[i] - result.At(i, 0)) / result.At(i, 0));
					CHECK(absDifference < 1E-4);
				}
			}
		}
	}
}

