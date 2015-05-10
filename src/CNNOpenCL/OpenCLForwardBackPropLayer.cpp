/*
 * OpenCLForwardBackPropLayer.cpp
 *
 *  Created on: May 5, 2015
 *      Author: Mikael
 */

#include "OpenCLForwardBackPropLayer.h"
#include <stdexcept>
#include <CL/cl.h>

namespace ATML
{
namespace MachineLearning
{

template class OpenCLForwardBackPropLayer<cl_float>;
template class OpenCLForwardBackPropLayer<cl_double>;

template<class T>
OpenCLForwardBackPropLayer<T>::OpenCLForwardBackPropLayer(
		shared_ptr<OpenCLContext> context,
		const vector<LayerDataDescription>& inputLayerDescriptions,
		const ForwardBackPropLayerConfig* config) :
		ForwardBackPropLayer(inputLayerDescriptions, config), context(context)
{

}

template<class T>
OpenCLForwardBackPropLayer<T>::~OpenCLForwardBackPropLayer()
{
	// TODO Auto-generated destructor stub
}

} /* namespace MachineLearning */
} /* namespace ATML */
