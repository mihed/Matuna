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
		throw runtime_error("The factory does not yield correct layers");
	createdOutputLayer.release();
}

template<class T>
CNNOpenCL<T>::~CNNOpenCL()
{

}

template<class T>
void CNNOpenCL<T>::FeedForward(const T* input, int formatIndex, T* output)
{

}

template<class T>
T CNNOpenCL<T>::CalculateError(const T* propagatedValue, int formatIndex,
		const T* target)
{
	return T();
}

template<class T>
void CNNOpenCL<T>::CalculateGradient(const T* input, int formatIndex, T* output)
{

}

template<class T>
void CNNOpenCL<T>::GetParameters(T* parameters)
{

}

template<class T>
size_t CNNOpenCL<T>::GetParameterCount()
{
	return 0;
}

template<class T>
void CNNOpenCL<T>::TrainNetwork(unique_ptr<CNNTrainer<T>> trainer,
		unique_ptr<IAlgorithmConfig> algorithm)
{

}

} /* namespace MachineLearning */
} /* namespace ATML */
