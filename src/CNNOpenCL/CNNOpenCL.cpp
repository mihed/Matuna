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
	CNNOpenCLFactoryVisitor factory(this->context, this);
	config->Accept(&factory);
	auto createdLayers = factory.GetLayers();
	auto createdOutputLayer = factory.GetOutputLayer();

	//Transfer ownership to this class
	for (auto& layer : createdLayers)
	{
		auto temp = dynamic_cast<OpenCLForwardBackPropLayer*>(layer.get());
		if (!temp)
			throw runtime_error("The factory does not yield correct layers");
		layers.push_back(unique_ptr<OpenCLForwardBackPropLayer>(temp));
		layer.release();
	}

	auto temp2 = dynamic_cast<StandardOutputLayer*>(createdOutputLayer.get());
	outputLayer = unique_ptr<StandardOutputLayer>(temp2);
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
