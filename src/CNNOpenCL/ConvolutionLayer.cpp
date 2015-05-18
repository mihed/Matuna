/*
 * ConvolutionLayer.cpp
 *
 *  Created on: May 5, 2015
 *      Author: Mikael
 */

#include "ConvolutionLayer.h"
#include <CL/cl.h>

namespace ATML {
namespace MachineLearning {

template<class T>
ConvolutionLayer<T>::ConvolutionLayer(shared_ptr<OpenCLContext> context,
		const vector<LayerDataDescription>& inputLayerDescriptions,
		ATMLActivationFunction backPropActivation,
		const ConvolutionLayerConfig* config) :
		OpenCLForwardBackPropLayer<T>(context, inputLayerDescriptions,
				backPropActivation, config) {

}

template<class T>
ConvolutionLayer<T>::~ConvolutionLayer() {

}

template<class T>
void ConvolutionLayer<T>::InterlockFinalized() {

}

template<class T>
void ConvolutionLayer<T>::EnqueueForwardPropagation(OpenCLDevice* device,
		int queueIndex, OpenCLMemory* previousInput, OpenCLMemory* output,
		bool blocking) {

}

template<class T>
void ConvolutionLayer<T>::EnqueueBackPropagation(OpenCLDevice* device,
		int queueIndex, OpenCLMemory* previousInput, OpenCLMemory* delta,
		OpenCLMemory* deltaOutput, bool blocking) {

}

template<class T>
void ConvolutionLayer<T>::EnqueueCalculateGradient(OpenCLDevice* device,
		int queueIndex, OpenCLMemory* previousInput, OpenCLMemory* delta,
		OpenCLMemory* gradient, bool blocking) {

}

template<class T>
vector<tuple<OpenCLMemory*, int>> ConvolutionLayer<T>::GetParameters() {
	return vector<tuple<OpenCLMemory*, int> >();
}

template<class T>
void ConvolutionLayer<T>::GetParameters(T* parameters, OpenCLDevice* device,
		int queueIndex, bool blocking) {

}

template<class T>
void ConvolutionLayer<T>::SetParameters(T* parameters, OpenCLDevice* device,
		int queueIndex, bool blocking) {

}

template<class T>
size_t ConvolutionLayer<T>::GetParameterCount() {
	return 0;
}


template class ConvolutionLayer<cl_float> ;
template class ConvolutionLayer<cl_double> ;

} /* namespace MachineLearning */
} /* namespace ATML */
