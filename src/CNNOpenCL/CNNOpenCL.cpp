/*
 * CNNOpenCL.cpp
 *
 *  Created on: May 5, 2015
 *      Author: Mikael
 */

#include "CNNOpenCL.h"
#include "CNNOpenCLFactoryVisitor.h"
#include <CL/cl.h>
#include <stdexcept>

namespace ATML
{
namespace MachineLearning
{
template class CNNOpenCL<cl_float> ;
template class CNNOpenCL<cl_double> ;

template<class T>
CNNOpenCL<T>::CNNOpenCL(unique_ptr<OpenCLContext> context,
		unique_ptr<CNNConfig> config) :
		TrainableCNN<T>(*config)
{
	this->context = move(context);

	for (auto device : this->context->GetDevices())
	{
		auto deviceInfo = device->DeviceInfo();
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
	}

	CNNOpenCLFactoryVisitor<T> factory(this->context, this);
	config->Accept(&factory);
	auto createdLayers = factory.GetLayers();
	auto createdOutputLayer = factory.GetOutputLayer();

	//Transfer ownership to this class
	for (auto& layer : createdLayers)
	{
		auto temp = dynamic_cast<OpenCLForwardBackPropLayer<T>*>(layer.get());
		if (!temp)
			throw runtime_error("The factory does not yield correct layers");
		layers.push_back(unique_ptr<OpenCLForwardBackPropLayer<T>>(temp));
		layer.release();
	}

	auto temp2 = dynamic_cast<StandardOutputLayer<T>*>(createdOutputLayer.get());
	outputLayer = unique_ptr<StandardOutputLayer<T>>(temp2);
	if (!temp2)
		throw runtime_error(
				"The factory does not yield correct layers. Every CNN must have an output layer");
	createdOutputLayer.release();
}

template<class T>
CNNOpenCL<T>::~CNNOpenCL()
{

}

template<class T>
unique_ptr<T[]> CNNOpenCL<T>::FeedForwardAligned(T* input, int formatIndex)
{
	auto devices = context->GetDevices();
	auto device = devices[0];
	LayerMemoryDescription inputMemoryDescription =
			this->InputMemoryDescriptions()[formatIndex];

	unique_ptr<OpenCLMemory> inputMemory = context->CreateMemory(
	CL_MEM_READ_ONLY, sizeof(T) * inputMemoryDescription.TotalMemory());
	device->WriteMemory(inputMemory.get(), inputMemory->ByteSize(), input, 0,
			true);

	//We need some synchronization in order not to use blocking calls. 
	//We could probably yield some better performance in that case.
	for (auto& layer : layers)
	{
		inputMemoryDescription =
				layer->OutForwardPropMemoryDescriptions()[formatIndex];

		unique_ptr<OpenCLMemory> outputMemory = context->CreateMemory(
		CL_MEM_READ_WRITE, sizeof(T) * inputMemoryDescription.TotalMemory());

		layer->EnqueueForwardPropagation(device, 0, inputMemory.get(),
				outputMemory.get(), true);
		inputMemory.reset();
		inputMemory = move(outputMemory);
	}

	unique_ptr<T[]> output(new T[inputMemoryDescription.TotalMemory()]);
	device->ReadMemory(inputMemory.get(), inputMemory->ByteSize(), output.get(), 0,
			true);

	return move(output);
}

template<class T>
T CNNOpenCL<T>::CalculateErrorAligned(T* propagatedValue, int formatIndex,
		T* target)
{
	return T();
}

template<class T>
unique_ptr<T[]> CNNOpenCL<T>::CalculateGradientAligned(T* input, int formatIndex)
{
	return nullptr;
}

template<class T>
unique_ptr<T[]> CNNOpenCL<T>::GetParameters()
{
	auto devices = context->GetDevices();
	auto device = devices[0];
	unique_ptr<T[]> parameters(new T[GetParameterCount()]);
	auto pointerPosition = parameters.get();
	for (auto& layer : layers)
	{
		layer->GetParameters(pointerPosition, device, 0, false);
		pointerPosition += layer->GetParameterCount();
	}

	device->WaitForDeviceQueue(0);

	return move(parameters);
}

template<class T>
void CNNOpenCL<T>::SetParameters(T* parameters)
{
	auto devices = context->GetDevices();
	auto device = devices[0];
	auto pointerPosition = parameters;
	for (auto& layer : layers)
	{
		layer->SetParameters(pointerPosition, device, 0, false);
		pointerPosition += layer->GetParameterCount();
	}

	device->WaitForDeviceQueue(0);
}

template<class T>
size_t CNNOpenCL<T>::GetParameterCount()
{
	size_t result = 0;
	for (auto& layer : layers)
		result += layer->GetParameterCount();

	return result;
}

template<class T>
void CNNOpenCL<T>::TrainNetwork(unique_ptr<CNNTrainer<T>> trainer,
		unique_ptr<IAlgorithmConfig> algorithm)
{

}

template<class T>
vector<OpenCLForwardBackPropLayer<T>*> CNNOpenCL<T>::GetLayers()
{
	vector<OpenCLForwardBackPropLayer<T>*> result;
	for (auto& layer : layers)
		result.push_back(layer.get());

	return result;
}

template<class T>
StandardOutputLayer<T>* CNNOpenCL<T>::GetOutputLayer()
{
	return outputLayer.get();
}

} /* namespace MachineLearning */
} /* namespace ATML */
