/*
 * ForthBackPropLayer.cpp
 *
 *  Created on: May 2, 2015
 *      Author: Mikael
 */

#include "ForwardBackPropLayer.h"
#include "InterlockHelper.h"
#include <stdexcept>

using namespace std;

namespace ATML
{
namespace MachineLearning
{

ForwardBackPropLayer::ForwardBackPropLayer(
		const LayerDataDescription& inputLayerDescription) :
		BackPropLayer(inputLayerDescription)
{
	outputInterlocked = false;
}

ForwardBackPropLayer::~ForwardBackPropLayer()
{

}

LayerMemoryDescription ForwardBackPropLayer::OutForwardPropMemoryDescription() const
{
	if (!outputInterlocked)
		throw runtime_error("The forward-prop out layer is not interlocked");

	return outForwardPropMemoryDescription;
}

void ForwardBackPropLayer::InterlockForwardPropOutput(
		const LayerMemoryDescription& outputDescription)
{
	if (outputInterlocked)
		throw runtime_error(
				"The forward-prop out layer is already interlocked");

	if (!InterlockHelper::IsCompatible(outForwardPropMemoryProposal,
			outputDescription))
		throw runtime_error(
				"The out forward memory description is incompatible");

	outForwardPropMemoryDescription = outputDescription;
	outputInterlocked = true;
}

} /* namespace MachineLearning */
} /* namespace ATML */
