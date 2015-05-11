/*
 * PerceptronLayer.cpp
 *
 *  Created on: May 5, 2015
 *      Author: Mikael
 */

#include "PerceptronLayer.h"
#include "CNN/InterlockHelper.h"
#include <stdexcept>
#include <type_traits>
#include <chrono>
#include <random>

namespace ATML
{
namespace MachineLearning
{

template class PerceptronLayer<cl_float> ;
template class PerceptronLayer<cl_double> ;

template<class T>
PerceptronLayer<T>::PerceptronLayer(shared_ptr<OpenCLContext> context,
		const vector<LayerDataDescription>& inputLayerDescriptions,
		const PerceptronLayerConfig* config) :
		OpenCLForwardBackPropLayer<T>(context, inputLayerDescriptions, config), config(
				*config)
{
	if (inputLayerDescriptions.size() == 0)
		throw invalid_argument(
				"There's no input data descriptions for the perceptron layer.");

	//In a perceptron layer, we cannot have multiple input descriptions for the same network
	//since it will correspond to a different weight matrix.
	if (inputLayerDescriptions.size() > 1)
	{
		auto count = inputLayerDescriptions.size();
		for (int i = 1; i < count; i++)
			if (!InterlockHelper::DataEquals(inputLayerDescriptions[i - 1],
					inputLayerDescriptions[i]))
				throw invalid_argument(
						"We cannot have multiple different input descriptions for a perceptron layer");
	}

	for (auto& layerDescription : inputLayerDescriptions)
	{
		LayerMemoryDescription inForwardMemProp;
		inForwardMemProp.Height = layerDescription.Height;
		inForwardMemProp.Width = layerDescription.Width;
		inForwardMemProp.Units = layerDescription.Units;
		inForwardMemProp.HeightOffset = 0;
		inForwardMemProp.UnitOffset = 0;
		inForwardMemProp.WidthOffset = 0;
		this->inForwardPropMemoryProposals.push_back(inForwardMemProp);
		this->outBackPropMemoryProposals.push_back(inForwardMemProp);

		LayerDataDescription outForwardDataDesc;
		outForwardDataDesc.Height = 1;
		outForwardDataDesc.Width = 1;
		outForwardDataDesc.Units = config->Units();
		this->outForwardPropDataDescriptions.push_back(outForwardDataDesc);

		this->inBackPropDataDescriptions = this->outForwardPropDataDescriptions;

		LayerMemoryDescription outForwardMemProp;
		outForwardMemProp.Height = 1;
		outForwardMemProp.Width = 1;
		outForwardMemProp.Units = config->Units();
		outForwardMemProp.HeightOffset = 0;
		outForwardMemProp.UnitOffset = 0;
		outForwardMemProp.WidthOffset = 0;
		this->outForwardPropMemoryProposals.push_back(outForwardMemProp);

		this->inBackPropMemoryProposals.push_back(outForwardMemProp);
	}

	auto inputDataDescriptions = this->InForwardPropDataDescription();
	inputDescription = inputDataDescriptions[0];
}

template<class T>
PerceptronLayer<T>::~PerceptronLayer()
{
	for (auto& deviceAndKernel : deviceAndKernels)
	{
		auto& kernelProgram = deviceAndKernel.second;
		this->context->RemoveKernel(kernelProgram.get());
		this->context->RemoveProgram(kernelProgram.get());
	}
}

template<class T>
PerceptronLayerConfig PerceptronLayer<T>::GetConfig() const
{
	return config;
}

template<class T>
void PerceptronLayer<T>::InterlockFinalized()
{
	auto inputMemoryDescriptions = this->InForwardPropMemoryDescription();
	auto& firstMemory = inputMemoryDescriptions[0];

	//IF the memory descriptions doesn't contain any padding or offsets, we may use the standard forward prop kernel.
	if (firstMemory.HeightOffset == 0 && firstMemory.UnitOffset == 0
			&& firstMemory.WidthOffset == 0
			&& firstMemory.Width == inputDescription.Width
			&& firstMemory.Height == inputDescription.Height
			&& firstMemory.Units == inputDescription.Units)
		InitializeNormalPerceptron();
	else
		InitializeImagePerceptron();
}

template<class T>
void PerceptronLayer<T>::InitializeNormalPerceptron()
{
	auto outputDataDescriptions = this->outForwardPropDataDescriptions;
	auto& firstOutputData = outputDataDescriptions[0];

	int biasCount = firstOutputData.Width * firstOutputData.Height
			* firstOutputData.Units;

	vector<OpenCLDevice*> devices = this->context->GetDevices();

	InitializeParameters();

	for (auto device : devices)
	{
		auto deviceInfo = device->DeviceInfo();

		//Make sure the type we want to execute is supported on the device.
		if (is_same<cl_double, T>::value)
		{
			if (deviceInfo.PreferredDoubleVectorWidth() == 0)
				throw invalid_argument(
						"The template argument is not supported on the chosen devices");
		}
		else if (is_same<cl_float, T>::value)
		{
			if (deviceInfo.PreferredFloatVectorWidth() == 0)
				throw invalid_argument(
						"The template argument is not supported on the chosen devices");
		}
		else
			throw runtime_error(
					"The template argument does not match the supported arguments");

		unique_ptr<ForwardPerceptronKernel<T>> kernel(
				new ForwardPerceptronKernel<T>(
						inputDescription.Width * inputDescription.Height
								* inputDescription.Units,
						firstOutputData.Width * firstOutputData.Height
								* firstOutputData.Units));

		//Now, let us query the device if we have enough memory to use constant weights / inputs / biases etc...
		auto maximumConstantBufferSize = deviceInfo.MaxConstantBufferSize();
		auto biasBytes = sizeof(T) * biasCount;
		if (maximumConstantBufferSize > weights->ByteSize())
		{
			kernel->SetUseConstantWeights(true);
			maximumConstantBufferSize -= weights->ByteSize();
		}
		if (maximumConstantBufferSize > biasBytes)
		{
			kernel->SetUseConstantInput(true);
			maximumConstantBufferSize -= biasBytes;
		}
		if (maximumConstantBufferSize > biasBytes)
		{
			kernel->SetUseConstantBiases(true);
			maximumConstantBufferSize -= biasBytes;
		}

		kernel->SetUseRelaxedMath(config.UseRelaxedMath());
		kernel->SetComputationPrecision(config.ComputationPrecision());
		kernel->SetActivationFunction(config.ActivationFunction());
		kernel->SetWeights(weights.get());
		kernel->SetBiases(biases.get());
		kernel->InitializeCompilerOptions();
		vector<OpenCLDevice*> oneDeviceVector;
		oneDeviceVector.push_back(device);
		this->context->AddProgramFromSource(kernel.get(), oneDeviceVector);
		this->context->AddKernel(kernel.get());
		kernel->InitializeArguments();
		deviceAndKernels.insert(make_pair(device, move(kernel)));
	}
}

template<class T>
void PerceptronLayer<T>::InitializeParameters()
{
	auto outputDataDescriptions = this->outForwardPropDataDescriptions;
	auto& firstOutputData = outputDataDescriptions[0];

	int weightCount = inputDescription.Width * inputDescription.Height
			* inputDescription.Units * firstOutputData.Width
			* firstOutputData.Height * firstOutputData.Units;

	int biasCount = firstOutputData.Width * firstOutputData.Height
			* firstOutputData.Units;

	weights = move(
			this->context->CreateMemory(CL_MEM_READ_WRITE,
					sizeof(T) * weightCount));

	biases = move(
			this->context->CreateMemory(CL_MEM_READ_WRITE,
					sizeof(T) * biasCount));

	auto seed = chrono::system_clock::now().time_since_epoch().count();
	default_random_engine generator(seed);

	uniform_real_distribution<T> uniformDistribution(0, 0.1);

	vector<T> initialWeightValues;
	initialWeightValues.resize(weightCount);
	for (int i = 0; i < weightCount; i++)
		initialWeightValues[i] = uniformDistribution(generator);

	vector<T> initialBiasValues;
	initialBiasValues.resize(biasCount);
	for (int i = 0; i < biasCount; i++)
		initialBiasValues[i] = uniformDistribution(generator);

	//Since this is initialization, we don't really care about which device and device queue we are using
	OpenCLDevice* device = this->context->GetDevices()[0];
	device->WriteMemory(weights.get(), sizeof(T) * initialWeightValues.size(),
			initialWeightValues.data(), 0, false);
	device->WriteMemory(biases.get(), sizeof(T) * initialBiasValues.size(),
			initialBiasValues.data(), 0, false);
	device->WaitForDeviceQueue(0);
}

template<class T>
void PerceptronLayer<T>::InitializeImagePerceptron()
{
	throw runtime_error("Not implemented");
}

template<class T>
void PerceptronLayer<T>::EnqueueForwardPropagation(OpenCLDevice* device,
		int queueIndex, OpenCLMemory* previousInput, OpenCLMemory* output,
		bool blocking)
{
	auto& kernel = deviceAndKernels[device];
	kernel->SetInput(previousInput);
	kernel->SetOutput(output);
	device->ExecuteKernel(kernel.get(), queueIndex, blocking);
}

template<class T>
void PerceptronLayer<T>::EnqueueBackPropagation(OpenCLDevice* device,
		int queueIndex, OpenCLMemory* previousInput, OpenCLMemory* delta,
		OpenCLMemory* deltaOutput, bool blocking)
{
	throw runtime_error("Not implemented");
}

template<class T>
void PerceptronLayer<T>::GetParameters(T* parameters, OpenCLDevice* device,
		int queueIndex, bool blocking)
{
	device->ReadMemory(weights.get(), weights->ByteSize(), parameters,
			queueIndex, blocking);
	auto biasPosition = parameters
			+ config.Units() * inputDescription.Width * inputDescription.Height
					* inputDescription.Units;
	device->ReadMemory(biases.get(), biases->ByteSize(), biasPosition,
			queueIndex, blocking);
}

template<class T>
void PerceptronLayer<T>::SetParameters(T* parameters,
		OpenCLDevice* device, int queueIndex, bool blocking)
{
	device->WriteMemory(weights.get(), weights->ByteSize(), parameters,
			queueIndex, blocking);
	auto biasPosition = parameters
			+ config.Units() * inputDescription.Width * inputDescription.Height
					* inputDescription.Units;
	device->WriteMemory(biases.get(), biases->ByteSize(), biasPosition,
			queueIndex, blocking);
}

template<class T>
size_t PerceptronLayer<T>::GetParameterCount()
{
	return inputDescription.Height * inputDescription.Width
			* inputDescription.Units * config.Units() + config.Units();
}

} /* namespace MachineLearning */
} /* namespace ATML */
