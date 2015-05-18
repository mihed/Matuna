/*
 * OpenCLForwardBackPropLayer.cpp
 *
 *  Created on: May 5, 2015
 *      Author: Mikael
 */

#include "OpenCLForwardBackPropLayer.h"
#include <stdexcept>
#include <CL/cl.h>

namespace ATML {
namespace MachineLearning {

template<class T>
OpenCLForwardBackPropLayer<T>::OpenCLForwardBackPropLayer(
		shared_ptr<OpenCLContext> context,
		const vector<LayerDataDescription>& inputLayerDescriptions,
		ATMLActivationFunction backPropActivation,
		const ForwardBackPropLayerConfig* config) :
		ForwardBackPropLayer(inputLayerDescriptions, backPropActivation,
				config), context(context) {

}

template<class T>
OpenCLForwardBackPropLayer<T>::~OpenCLForwardBackPropLayer() {

}

template class OpenCLForwardBackPropLayer<cl_float> ;
template class OpenCLForwardBackPropLayer<cl_double> ;

} /* namespace MachineLearning */
} /* namespace ATML */
