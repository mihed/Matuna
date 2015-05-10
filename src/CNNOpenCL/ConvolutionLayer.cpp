/*
 * ConvolutionLayer.cpp
 *
 *  Created on: May 5, 2015
 *      Author: Mikael
 */

#include "ConvolutionLayer.h"
#include <CL/cl.h>

namespace ATML
{
namespace MachineLearning
{

template class ConvolutionLayer<cl_float> ;
template class ConvolutionLayer<cl_double> ;

template<class T>
ConvolutionLayer<T>::ConvolutionLayer(shared_ptr<OpenCLContext> context,
		const vector<LayerDataDescription>& inputLayerDescriptions,
		const ConvolutionLayerConfig* config) :
		OpenCLForwardBackPropLayer<T>(context, inputLayerDescriptions, config)
{

}

template<class T>
ConvolutionLayer<T>::~ConvolutionLayer()
{

}

template<class T>
void ConvolutionLayer<T>::InterlockFinalized()
{

}

template<class T>
void ConvolutionLayer<T>::EnqueueForwardPropagation(
		shared_ptr<OpenCLMemory> previousInput, shared_ptr<OpenCLMemory> output)
{

}

template<class T>
void ConvolutionLayer<T>::EnqueueBackPropagation(
		shared_ptr<OpenCLMemory> previousInput, shared_ptr<OpenCLMemory> delta,
		shared_ptr<OpenCLMemory> deltaOutput)
{

}

} /* namespace MachineLearning */
} /* namespace ATML */
