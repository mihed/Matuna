/*
* StandardOutputLayerTest.cpp
*
*  Created on: May 15, 2015
*      Author: Mikael
*/

#define CATCH_CONFIG_MAIN
#include "catch/catch.hpp"
#include "Matuna.OCLHelper/OCLHelper.h"
#include "Matuna.OCLConvNet/OCLConvNet.h"
#include "Matuna.ConvNet/ConvNetConfig.h"
#include "Matuna.ConvNet/StandardOutputLayerConfig.h"
#include "Matuna.ConvNet/StandardOutputLayerConfig.h"
#include "Matuna.ConvNet/LayerDescriptions.h"
#include "Matuna.Math/Matrix.h"
#include <memory>
#include <random>
#include <cmath>
#include <type_traits>

using namespace std;
using namespace Matuna::MachineLearning;
using namespace Matuna::Helper;
using namespace Matuna::Math;



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

SCENARIO("Creating a ConvNet with a standard output layer. Image input")
{
	auto platformInfos = OCLHelper::GetPlatformInfos();
	random_device device;
	mt19937 mt(device());
	uniform_int_distribution<int> dimensionGenerator(2, 100);
	uniform_int_distribution<int> unitGenerator(1, 10);

	vector<vector<OCLDeviceInfo>> deviceInfos;
	for (auto platformInfo : platformInfos)
		deviceInfos.push_back(OCLHelper::GetDeviceInfos(platformInfo));

	int iterations = 10;

	WHEN("Calculating the CE error output using an image")
	{
		THEN("The CE error must be equal to the manually calculated CE error")
		{
			for (int dummy = 0; dummy < iterations; dummy++)
			{
				for (auto& deviceInfo : deviceInfos)
				{
					unique_ptr<OutputLayerConfig> outputLayerConfig(new StandardOutputLayerConfig(MatunaCrossEntropy));
					LayerDataDescription inputDescription;
					inputDescription.Height = dimensionGenerator(mt);
					inputDescription.Width = dimensionGenerator(mt);
					inputDescription.Units = unitGenerator(mt);
					vector<LayerDataDescription> temp;
					temp.push_back(inputDescription);
					unique_ptr<ConvNetConfig> config(new ConvNetConfig(temp));
					config->SetOutputConfig(move(outputLayerConfig));
					OCLConvNet<float> network(deviceInfo, move(config));

					vector<Matrixf> randomInputs;
					vector<Matrixf> randomTargets;
					unique_ptr<float[]> rawInputData(new float[inputDescription.Width * inputDescription.Height * inputDescription.Units]);
					unique_ptr<float[]> rawTargetData(new float[inputDescription.Width * inputDescription.Height * inputDescription.Units]);
					for (int i = 0; i < inputDescription.Units; i++)
					{
						auto randomInput = Matrix<float>::RandomUniform(inputDescription.Height, inputDescription.Width);
						randomInput =  (1.0f / randomInput.Sum()) * randomInput;
						auto randomTarget = Matrix<float>::RandomUniform(inputDescription.Height, inputDescription.Width);
						randomTarget = (1.0f / randomTarget.Sum()) * randomTarget;
						memcpy(rawTargetData.get() + i * (inputDescription.Height * inputDescription.Width), randomTarget.Data, sizeof(float) * inputDescription.Height * inputDescription.Width);
						memcpy(rawInputData.get() + i * (inputDescription.Height * inputDescription.Width), randomInput.Data, sizeof(float) * inputDescription.Height * inputDescription.Width);
						randomInputs.push_back(randomInput);
						randomTargets.push_back(randomTarget);
					}

					float result;
					if (network.RequireForwardOutputAlignment(0))
					{
						auto alignedOutput = move(network.AlignToForwardOutput(rawInputData.get(), 0));
						auto alignedTarget = move(network.AlignToForwardOutput(rawTargetData.get(), 0));
						result = network.CalculateErrorFromForwardAligned(alignedOutput.get(), 0, alignedTarget.get());
					}
					else
						result = network.CalculateErrorFromForwardAligned(rawInputData.get(), 0, rawTargetData.get());

					Matrixf manualResultMatrix = Matrixf::Zeros(inputDescription.Height, inputDescription.Width);
					for(int i = 0; i < inputDescription.Units; i++)
					{
						auto randomInput = randomInputs[i];
						randomInput.Transform([] (float x) { return log(x);});
						manualResultMatrix -= randomTargets[i] % randomInput;
					}

					auto manualResult = manualResultMatrix.Sum();

					if (manualResult == numeric_limits<float>::infinity())
						manualResult = numeric_limits<float>::max();

					float absDifference = result > manualResult ? (result - manualResult) : (manualResult - result);
					if (manualResult > 1E-8)
						absDifference = absDifference / manualResult;
					printf("Abs difference: %f \n", absDifference);
					CHECK(absDifference < 1E-4f);
				}
			}
		}
	}

	WHEN("Calculating the MSE error output using an image")
	{
		THEN("The MSE error must be equal to the manually calculated MSE error")
		{
			for (int dummy = 0; dummy < iterations; dummy++)
			{
				for (auto& deviceInfo : deviceInfos)
				{
					unique_ptr<OutputLayerConfig> outputLayerConfig(new StandardOutputLayerConfig(MatunaMeanSquareError));
					LayerDataDescription inputDescription;
					inputDescription.Height = dimensionGenerator(mt);
					inputDescription.Width = dimensionGenerator(mt);
					inputDescription.Units = unitGenerator(mt);
					vector<LayerDataDescription> temp;
					temp.push_back(inputDescription);
					unique_ptr<ConvNetConfig> config(new ConvNetConfig(temp));
					config->SetOutputConfig(move(outputLayerConfig));
					OCLConvNet<float> network(deviceInfo, move(config));

					vector<Matrixf> randomInputs;
					vector<Matrixf> randomTargets;
					unique_ptr<float[]> rawInputData(new float[inputDescription.Width * inputDescription.Height * inputDescription.Units]);
					unique_ptr<float[]> rawTargetData(new float[inputDescription.Width * inputDescription.Height * inputDescription.Units]);
					for (int i = 0; i < inputDescription.Units; i++)
					{
						auto randomInput = Matrix<float>::RandomUniform(inputDescription.Height, inputDescription.Width);
						randomInput =  (1.0f / randomInput.Sum()) * randomInput;
						auto randomTarget = Matrix<float>::RandomUniform(inputDescription.Height, inputDescription.Width);
						randomTarget = (1.0f / randomTarget.Sum()) * randomTarget;
						memcpy(rawTargetData.get() + i * (inputDescription.Height * inputDescription.Width), randomTarget.Data, sizeof(float) * inputDescription.Height * inputDescription.Width);
						memcpy(rawInputData.get() + i * (inputDescription.Height * inputDescription.Width), randomInput.Data, sizeof(float) * inputDescription.Height * inputDescription.Width);
						randomInputs.push_back(randomInput);
						randomTargets.push_back(randomTarget);
					}

					float result;
					if (network.RequireForwardOutputAlignment(0))
					{
						auto alignedOutput = move(network.AlignToForwardOutput(rawInputData.get(), 0));
						auto alignedTarget = move(network.AlignToForwardOutput(rawTargetData.get(), 0));
						result = network.CalculateErrorFromForwardAligned(alignedOutput.get(), 0, alignedTarget.get());
					}
					else
						result = network.CalculateErrorFromForwardAligned(rawInputData.get(), 0, rawTargetData.get());

					float manualResult = 0;
					for(int i = 0; i < inputDescription.Units; i++)
						manualResult += (randomTargets[i] - randomInputs[i]).Norm2Square();

					manualResult = 0.5f * manualResult;

					if (manualResult == numeric_limits<float>::infinity())
						manualResult = numeric_limits<float>::max();

					float absDifference = result > manualResult ? (result - manualResult) : (manualResult - result);
					if (manualResult > 1E-8)
						absDifference = absDifference / manualResult;
					printf("Abs difference: %f \n", absDifference);
					CHECK(absDifference < 1E-4f);
				}
			}
		}
	}


	WHEN("Creating a float ConvNet with CE and linear activation with normal math using an image")
	{
		THEN("The back propagation must be equal to the manually calcualted values")
		{
			for (int dummy = 0; dummy < iterations; dummy++)
			{
				for (auto& deviceInfo : deviceInfos)
				{
					unique_ptr<OutputLayerConfig> outputLayerConfig(new StandardOutputLayerConfig(MatunaCrossEntropy));
					LayerDataDescription inputDescription;
					inputDescription.Height = dimensionGenerator(mt);
					inputDescription.Width = dimensionGenerator(mt);
					inputDescription.Units = unitGenerator(mt);
					vector<LayerDataDescription> temp;
					temp.push_back(inputDescription);
					unique_ptr<ConvNetConfig> config(new ConvNetConfig(temp));
					config->SetOutputConfig(move(outputLayerConfig));
					OCLConvNet<float> network(deviceInfo, move(config));

					vector<Matrixf> randomInputs;
					vector<Matrixf> randomTargets;
					unique_ptr<float[]> rawInputData(new float[inputDescription.Width * inputDescription.Height * inputDescription.Units]);
					unique_ptr<float[]> rawTargetData(new float[inputDescription.Width * inputDescription.Height * inputDescription.Units]);
					float targetSum = 0;
					float inputSum = 0;
					for (int i = 0; i < inputDescription.Units; i++)
					{
						auto randomInput = Matrix<float>::RandomUniform(inputDescription.Height, inputDescription.Width);
						auto randomTarget = Matrix<float>::RandomUniform(inputDescription.Height, inputDescription.Width);
						targetSum += randomTarget.Sum();
						inputSum += randomInput.Sum();
						randomInputs.push_back(randomInput);
						randomTargets.push_back(randomTarget);
					}

					for (int i = 0; i < inputDescription.Units; i++)
					{
						auto randomTarget = randomTargets[i];
						auto randomInput = randomInputs[i];
						randomTarget = (1.0f / targetSum) * randomTarget;
						randomInput = (1.0f / inputSum) * randomInput;
						randomTargets[i] = randomTarget;
						randomInputs[i] = randomInput;
						memcpy(rawTargetData.get() + i * (inputDescription.Height * inputDescription.Width), randomTarget.Data, sizeof(float) * inputDescription.Height * inputDescription.Width);
						memcpy(rawInputData.get() + i * (inputDescription.Height * inputDescription.Width), randomInput.Data, sizeof(float) * inputDescription.Height * inputDescription.Width);
					}

					auto result = network.BackPropUnaligned(rawInputData.get(), 0, rawTargetData.get());

					vector<Matrixf> manualBackPropMatrices;
					for (int i = 0; i < inputDescription.Units; i++)
					{
						Matrixf tempInput = randomInputs[i];
						tempInput.Transform([](float x) { return -1.0f / x;});
						auto tempResult = randomTargets[i] % tempInput;
						manualBackPropMatrices.push_back(tempResult);
					}

					for (int i = 0; i < inputDescription.Units; i++)
					{
						Matrixf tempResult(inputDescription.Height, inputDescription.Width);
						memcpy(tempResult.Data, result.get() + i * inputDescription.Width * inputDescription.Height, sizeof(float) *  inputDescription.Width * inputDescription.Height);
						auto difference = (tempResult - manualBackPropMatrices[i]).Norm2Square() / tempResult.ElementCount();
						printf("Difference: %f \n", difference);
						CHECK(difference < 1E-4);
					}
				}
			}
		}
	}

	WHEN("Creating a float ConvNet with MSE and linear activation with normal math using an image")
	{
		THEN("The back propagation must be equal to difference between target and input")
		{
			for (int dummy = 0; dummy < iterations; dummy++)
			{
				for (auto& deviceInfo : deviceInfos)
				{
					unique_ptr<OutputLayerConfig> outputLayerConfig(new StandardOutputLayerConfig(MatunaMeanSquareError));
					LayerDataDescription inputDescription;
					inputDescription.Height = dimensionGenerator(mt);
					inputDescription.Width = dimensionGenerator(mt);
					inputDescription.Units = unitGenerator(mt);
					vector<LayerDataDescription> temp;
					temp.push_back(inputDescription);
					unique_ptr<ConvNetConfig> config(new ConvNetConfig(temp));
					config->SetOutputConfig(move(outputLayerConfig));
					OCLConvNet<float> network(deviceInfo, move(config));

					vector<Matrixf> randomInputs;
					vector<Matrixf> randomTargets;
					unique_ptr<float[]> rawInputData(new float[inputDescription.Width * inputDescription.Height * inputDescription.Units]);
					unique_ptr<float[]> rawTargetData(new float[inputDescription.Width * inputDescription.Height * inputDescription.Units]);
					for (int i = 0; i < inputDescription.Units; i++)
					{
						auto randomInput = Matrix<float>::RandomUniform(inputDescription.Height, inputDescription.Width);
						auto randomTarget = Matrix<float>::RandomUniform(inputDescription.Height, inputDescription.Width);
						randomInputs.push_back(randomInput);
						randomTargets.push_back(randomTarget);
					}

					for (int i = 0; i < inputDescription.Units; i++)
					{
						auto& randomTarget = randomTargets[i];
						auto& randomInput = randomInputs[i];
						memcpy(rawTargetData.get() + i * (inputDescription.Height * inputDescription.Width), randomTarget.Data, sizeof(float) * inputDescription.Height * inputDescription.Width);
						memcpy(rawInputData.get() + i * (inputDescription.Height * inputDescription.Width), randomInput.Data, sizeof(float) * inputDescription.Height * inputDescription.Width);
					}

					auto result = network.BackPropUnaligned(rawInputData.get(), 0, rawTargetData.get());

					vector<Matrixf> manualBackPropMatrices;
					for (int i = 0; i < inputDescription.Units; i++)
					{
						auto tempResult = randomInputs[i] - randomTargets[i];
						manualBackPropMatrices.push_back(tempResult);
					}

					for (int i = 0; i < inputDescription.Units; i++)
					{
						Matrixf tempResult(inputDescription.Height, inputDescription.Width);
						memcpy(tempResult.Data, result.get() + i * inputDescription.Width * inputDescription.Height, sizeof(float) *  inputDescription.Width * inputDescription.Height);
						auto difference = (tempResult - manualBackPropMatrices[i]).Norm2Square() / tempResult.ElementCount();
						printf("Difference: %f \n", difference);
						CHECK(difference < 1E-4);
					}
				}
			}
		}
	}
}

SCENARIO("Creating a ConvNet with a standard output layer. No image inputs")
{
	auto platformInfos = OCLHelper::GetPlatformInfos();
	random_device device;
	mt19937 mt(device());
	uniform_int_distribution<int> dimensionGenerator(1, 100);

	vector<vector<OCLDeviceInfo>> deviceInfos;
	for (auto platformInfo : platformInfos)
		deviceInfos.push_back(OCLHelper::GetDeviceInfos(platformInfo));

	int iterations = 10;

	WHEN("Calculating the CE error output")
	{
		THEN("The CE error must be equal to the manually calculated CE error")
		{
			for (int dummy = 0; dummy < iterations; dummy++)
			{
				for (auto& deviceInfo : deviceInfos)
				{
					unique_ptr<OutputLayerConfig> outputLayerConfig(new StandardOutputLayerConfig(MatunaCrossEntropy));
					LayerDataDescription inputDescription;
					inputDescription.Height = 1;
					inputDescription.Width = 1;
					inputDescription.Units = dimensionGenerator(mt);
					vector<LayerDataDescription> temp;
					temp.push_back(inputDescription);
					unique_ptr<ConvNetConfig> config(new ConvNetConfig(temp));
					config->SetOutputConfig(move(outputLayerConfig));
					OCLConvNet<float> network(deviceInfo, move(config));

					auto randomInputs = Matrix<float>::RandomUniform(inputDescription.Units, 1);
					randomInputs =  (1.0f / randomInputs.Sum()) * randomInputs;
					auto randomTargets = Matrix<float>::RandomUniform(inputDescription.Units, 1);
					randomTargets = (1.0f / randomTargets.Sum()) * randomTargets;

					float result;
					if (network.RequireForwardOutputAlignment(0))
					{
						auto alignedOutput = move(network.AlignToForwardOutput(randomInputs.Data, 0));
						auto alignedTarget = move(network.AlignToForwardOutput(randomTargets.Data, 0));
						result = network.CalculateErrorFromForwardAligned(alignedOutput.get(), 0, alignedTarget.get());
					}
					else
						result = network.CalculateErrorFromForwardAligned(randomInputs.Data, 0, randomTargets.Data);

					float manualResult = 0;
					if (inputDescription.Units != 1)
						for(int i = 0; i < inputDescription.Units; i++)
							manualResult -= randomTargets.Data[i] * log(randomInputs.Data[i]);
					else
						manualResult = -( randomTargets.Data[0] == 0 ? 0 : (randomTargets.Data[0] * log(randomInputs.Data[0])) + 
						randomTargets.Data[0] == 1 ? 0 : ((1 - randomTargets.Data[0]) * log(1 - randomInputs.Data[0])));

					if (manualResult == numeric_limits<float>::infinity())
						manualResult = numeric_limits<float>::max();

					float absDifference = result > manualResult ? (result - manualResult) : (manualResult - result);
					if (manualResult > 1E-8)
						absDifference = absDifference / manualResult;
					printf("Abs difference: %f \n", absDifference);
					CHECK(absDifference < 1E-6f);
				}
			}
		}
	}

	WHEN("Calculating the MSE error output")
	{
		THEN("The MSE error must be equal to the vector norm of the difference")
		{
			for (int dummy = 0; dummy < iterations; dummy++)
			{
				for (auto& deviceInfo : deviceInfos)
				{
					unique_ptr<OutputLayerConfig> outputLayerConfig(new StandardOutputLayerConfig());
					LayerDataDescription inputDescription;
					inputDescription.Height = 1;
					inputDescription.Width = 1;
					inputDescription.Units = dimensionGenerator(mt);
					vector<LayerDataDescription> temp;
					temp.push_back(inputDescription);
					unique_ptr<ConvNetConfig> config(new ConvNetConfig(temp));
					config->SetOutputConfig(move(outputLayerConfig));
					OCLConvNet<float> network(deviceInfo, move(config));
					auto randomInputs = Matrix<float>::RandomNormal(inputDescription.Units, 1);
					auto randomTargets = Matrix<float>::RandomNormal(inputDescription.Units, 1);

					float result;
					if (network.RequireForwardOutputAlignment(0))
					{
						auto alignedOutput = move(network.AlignToForwardOutput(randomInputs.Data, 0));
						auto alignedTarget = move(network.AlignToForwardOutput(randomTargets.Data, 0));
						result = network.CalculateErrorFromForwardAligned(alignedOutput.get(), 0, alignedTarget.get());
					}
					else
						result = network.CalculateErrorFromForwardAligned(randomInputs.Data, 0, randomTargets.Data);

					auto difference = randomInputs - randomTargets;
					auto manualResult = 0.5f * difference.Norm2Square();
					auto absDifference = abs(result - manualResult);
					if (manualResult > 1E-8)
						absDifference = absDifference / manualResult;
					CHECK(absDifference < 1E-6f);
				}
			}
		}
	}



	WHEN("Creating a float ConvNet with MSE and Linear activation with normal math")
	{
		THEN("The back propagation must be equal to difference between target and input")
		{
			for (int dummy = 0; dummy < iterations; dummy++)
			{
				for (auto& deviceInfo : deviceInfos)
				{
					unique_ptr<OutputLayerConfig> outputLayerConfig(new StandardOutputLayerConfig());
					LayerDataDescription inputDescription;
					inputDescription.Height = 1;
					inputDescription.Width = 1;
					inputDescription.Units = dimensionGenerator(mt);
					vector<LayerDataDescription> temp;
					temp.push_back(inputDescription);
					unique_ptr<ConvNetConfig> config(new ConvNetConfig(temp));
					config->SetOutputConfig(move(outputLayerConfig));
					OCLConvNet<float> network(deviceInfo, move(config));
					auto randomInputs = Matrix<float>::RandomNormal(inputDescription.Units, 1);
					auto randomTargets = Matrix<float>::RandomNormal(inputDescription.Units, 1);

					auto result = network.BackPropUnaligned(randomInputs.Data, 0, randomTargets.Data);
					auto manualDifference = randomInputs - randomTargets;
					for (int i = 0; i < inputDescription.Units; i++)
					{
						auto absDifference = (manualDifference.Data[i] - result[i]);
						if (manualDifference.Data[i] > 1E-8)
							absDifference /=  manualDifference.Data[i];
						CHECK(absDifference < 1E-7);
					}
				}
			}
		}
	}

	WHEN("Creating a float ConvNet with MSE and Linear activation and relaxed math")
	{
		THEN("The back propagation must be equal to difference between target and input")
		{
			for (int dummy = 0; dummy < iterations; dummy++)
			{
				for (auto& deviceInfo : deviceInfos)
				{
					unique_ptr<OutputLayerConfig> outputLayerConfig(new StandardOutputLayerConfig(MatunaMeanSquareError, true));
					LayerDataDescription inputDescription;
					inputDescription.Height = 1;
					inputDescription.Width = 1;
					inputDescription.Units = dimensionGenerator(mt);
					vector<LayerDataDescription> temp;
					temp.push_back(inputDescription);
					unique_ptr<ConvNetConfig> config(new ConvNetConfig(temp));
					config->SetOutputConfig(move(outputLayerConfig));
					auto randomInputs = Matrix<float>::RandomNormal(inputDescription.Units, 1);
					auto randomTargets = Matrix<float>::RandomNormal(inputDescription.Units, 1);

					OCLConvNet<float> network(deviceInfo, move(config));

					auto result = network.BackPropUnaligned(randomInputs.Data, 0, randomTargets.Data);
					auto manualDifference = randomInputs - randomTargets;
					for (int i = 0; i < inputDescription.Units; i++)
					{
						auto absDifference = (manualDifference.Data[i] - result[i]);
						if (manualDifference.Data[i] > 1E-8)
							absDifference /= manualDifference.Data[i];
						CHECK(absDifference < 1E-7);
					}
				}
			}
		}
	}
}


