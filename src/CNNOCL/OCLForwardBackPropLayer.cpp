/*
 * OCLForwardBackPropLayer.cpp
 *
 *  Created on: May 5, 2015
 *      Author: Mikael
 */

#include "OCLForwardBackPropLayer.h"
#include <stdexcept>

namespace Matuna {
namespace MachineLearning {

template<class T>
OCLForwardBackPropLayer<T>::OCLForwardBackPropLayer(
		shared_ptr<OCLContext> context,
		const vector<LayerDataDescription>& inputLayerDescriptions,
		MatunaActivationFunction backPropActivation,
		const ForwardBackPropLayerConfig* config) :
		ForwardBackPropLayer(inputLayerDescriptions, backPropActivation,
				config), context(context) {

}

template<class T>
OCLForwardBackPropLayer<T>::~OCLForwardBackPropLayer() {

}

template class OCLForwardBackPropLayer<cl_float> ;
template class OCLForwardBackPropLayer<cl_double> ;

} /* namespace MachineLearning */
} /* namespace Matuna */
