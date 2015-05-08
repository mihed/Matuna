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
		const vector<LayerDataDescription>& inputLayerDescriptions,
		const ForwardBackPropLayerConfig* config) :
		BackPropLayer(inputLayerDescriptions)
{
	outputInterlocked = false;
}

ForwardBackPropLayer::~ForwardBackPropLayer()
{

}

vector<LayerMemoryDescription> ForwardBackPropLayer::OutForwardPropMemoryDescription() const
{
	if (!outputInterlocked)
		throw runtime_error("The forward-prop out layer is not interlocked");

	return outForwardPropMemoryDescriptions;
}

void ForwardBackPropLayer::InterlockForwardPropOutput(
		const vector<LayerMemoryDescription>& outputDescriptions)
{
	if (outputInterlocked)
		throw runtime_error(
				"The forward-prop out layer is already interlocked");

	if (!InterlockHelper::IsCompatible(outForwardPropMemoryProposals,
			outputDescriptions))
		throw runtime_error(
				"The out forward memory description is incompatible");

	outForwardPropMemoryDescriptions = outputDescriptions;
	outputInterlocked = true;
}

} /* namespace MachineLearning */
} /* namespace ATML */
