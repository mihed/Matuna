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

CNNOpenCL::CNNOpenCL(unique_ptr<OpenCLContext> context, CNNConfig config) :
		CNN(config)
{
	this->context = move(context);
	CNNOpenCLFactoryVisitor factory(this->context, this);
	config.Accept(&factory);
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

	//TODO: Make sure the factory interlocks the network that creates it as well.
}

CNNOpenCL::~CNNOpenCL()
{

}

template void CNNOpenCL::FeedForward<cl_float>(const cl_float* input,
		int formatIndex, cl_float* output);
template void CNNOpenCL::FeedForward<cl_double>(const cl_double* input,
		int formatIndex, cl_double* output);

template<class T>
void CNNOpenCL::FeedForward(const T* input, int formatIndex, T* output)
{

}

template cl_float CNNOpenCL::CalculateError<cl_float>(
		const cl_float* propagatedValue, int formatIndex,
		const cl_float* target);
template cl_double CNNOpenCL::CalculateError<cl_double>(
		const cl_double* propagatedValue, int formatIndex,
		const cl_double* target);

template<class T>
T CNNOpenCL::CalculateError(const T* propagatedValue, int formatIndex,
		const T* target)
{
	return T();
}

template void CNNOpenCL::CalculateGradient<cl_float>(const cl_float* input,
		int formatIndex, cl_float* output);
template void CNNOpenCL::CalculateGradient<cl_double>(const cl_double* input,
		int formatIndex, cl_double* output);

template<class T>
void CNNOpenCL::CalculateGradient(const T* input, int formatIndex, T* output)
{

}

template void CNNOpenCL::GetParameters<cl_float>(cl_float* parameters);
template void CNNOpenCL::GetParameters<cl_double>(cl_double* parameters);

template<class T>
void CNNOpenCL::GetParameters(T* parameters)
{

}

} /* namespace MachineLearning */
} /* namespace ATML */
