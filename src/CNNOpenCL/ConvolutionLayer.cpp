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
void ConvolutionLayer<T>::EnqueueForwardPropagation(OpenCLDevice* device, int queueIndex,
		OpenCLMemory* previousInput, OpenCLMemory* output, bool blocking)
{

}

template<class T>
void ConvolutionLayer<T>::EnqueueBackPropagation(OpenCLDevice* device, int queueIndex,
		OpenCLMemory* previousInput, OpenCLMemory* delta,
		OpenCLMemory* deltaOutput, bool blocking)
{

}

} /* namespace MachineLearning */
} /* namespace ATML */
