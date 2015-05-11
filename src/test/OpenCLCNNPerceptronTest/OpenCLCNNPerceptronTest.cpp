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
#include "CNN/PerceptronLayerConfig.h"
#include "CNN/StandardOutputLayerConfig.h"
#include <memory>

using namespace std;
using namespace ATML::MachineLearning;
using namespace ATML::Helper;


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
				unique_ptr<StandardOutputLayerConfig> outputLayerConfig(new StandardOutputLayerConfig());

				INFO("Pushng the perceptron config to the cnn config");
				config->AddToBack(move(perceptronConfig));
				config->SetOutputConfig(move(outputLayerConfig));

				INFO("Initializing the network");
				CNNOpenCL<cl_float> network(move(contextPointer), move(config));
				CHECK(network.Interlocked());

				INFO("Creating a pointer with previusly calculated parameters");
				//We know that the weight matrix is located in memory like an image buffer, with the bias value directly after.
				unique_ptr<cl_float[]> parameters(new cl_float[3]);
				parameters[0] = 1.3860e03f;
				parameters[1] = 776.3274f;
				parameters[2] = -397.6140f;

				INFO("Setting the network parameters to the previosly calculated OR parameters");
				network.SetParameters(parameters.get());

				CHECK(network.GetParameterCount() == 3);

				INFO("Making sure the read parameters correspond to the set parameters");
				unique_ptr<cl_float[]> readParameters(new cl_float[network.GetParameterCount()]);
				network.GetParameters(readParameters.get());

				for (int i = 0; i < 3; i++)
				{
					CHECK(readParameters[i] == parameters[i]);
				}

				CHECK(targetsFloat.size() == inputsFloat.size());
				for (int i = 0; i < targetsFloat.size(); i++)
				{
					cl_float output;
					network.FeedForward(inputsFloat[i].data(), 0, &output);
					auto difference = abs(targetsFloat[i] - output);
					REQUIRE(difference < 0.01);
				}
			}
		}
	}
}


