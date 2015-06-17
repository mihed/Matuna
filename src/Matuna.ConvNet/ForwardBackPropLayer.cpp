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

namespace Matuna
{
namespace MachineLearning
{

ForwardBackPropLayer::ForwardBackPropLayer(
		const vector<LayerDataDescription>& inputLayerDescriptions,
		MatunaActivationFunction backPropActivation,
		const ForwardBackPropLayerConfig*) :
		BackPropLayer(inputLayerDescriptions, backPropActivation)
{
	outputInterlocked = false;
}

ForwardBackPropLayer::~ForwardBackPropLayer()
{

}

bool ForwardBackPropLayer::Interlocked() const
{
	return outputInterlocked && BackPropLayer::Interlocked();
}

vector<LayerMemoryDescription> ForwardBackPropLayer::OutForwardPropMemoryDescriptions() const
{
	if (!outputInterlocked)
		throw runtime_error("The forward-prop out layer is not interlocked");

	return outForwardPropMemoryDescriptions;
}

vector<LayerDataDescription> ForwardBackPropLayer::OutForwardPropDataDescriptions() const
{
	return outForwardPropDataDescriptions;
}

vector<LayerMemoryDescription> ForwardBackPropLayer::OutForwardPropMemoryProposals() const
{
	return outForwardPropMemoryProposals;
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
} /* namespace Matuna */
