/*
 * OpenCLForwardBackPropLayer.cpp
 *
 *  Created on: May 5, 2015
 *      Author: Mikael
 */

#include "OpenCLForwardBackPropLayer.h"
#include <stdexcept>

namespace ATML
{
namespace MachineLearning
{

OpenCLForwardBackPropLayer::OpenCLForwardBackPropLayer(
		shared_ptr<OpenCLContext> context,
		const vector<LayerDataDescription>& inputLayerDescriptions,
		const ForwardBackPropLayerConfig* config) :
		ForwardBackPropLayer(inputLayerDescriptions, config), context(context)
{

}

OpenCLForwardBackPropLayer::~OpenCLForwardBackPropLayer()
{
	// TODO Auto-generated destructor stub
}

} /* namespace MachineLearning */
} /* namespace ATML */
