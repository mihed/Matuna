/*
 * OpenCLForwardBackPropLayer.cpp
 *
 *  Created on: May 5, 2015
 *      Author: Mikael
 */

#include "OpenCLForwardBackPropLayer.h"

namespace ATML
{
namespace MachineLearning
{

OpenCLForwardBackPropLayer::OpenCLForwardBackPropLayer(
		const LayerDataDescription& inputLayerDescription,
		const ForwardBackPropLayerConfig* config) :
		ForwardBackPropLayer(inputLayerDescription, config)
{

}

OpenCLForwardBackPropLayer::~OpenCLForwardBackPropLayer()
{
	// TODO Auto-generated destructor stub
}

} /* namespace MachineLearning */
} /* namespace ATML */
