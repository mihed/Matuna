/*
 * BackPropLayer.cpp
 *
 *  Created on: May 2, 2015
 *      Author: Mikael
 */

#include "BackPropLayer.h"
#include "InterlockHelper.h"
#include <stdexcept>

using namespace std;

namespace ATML
{
namespace MachineLearning
{

BackPropLayer::BackPropLayer(
		const vector<LayerDataDescription>& inputLayerDescriptions) :
		inForwardPropDataDescriptions(inputLayerDescriptions)
{
	inputInterlocked = false;
	outputInterlocked = false;
	forwardInputInterlocked = false;
}

BackPropLayer::~BackPropLayer()
{

}

bool BackPropLayer::Interlocked() const
{
	return inputInterlocked && outputInterlocked && forwardInputInterlocked;
}

vector<LayerMemoryDescription> BackPropLayer::InForwardPropMemoryDescription() const
{
	if (!forwardInputInterlocked)
		throw runtime_error("The forward-prop in layer is not interlocked");

	return inForwardPropMemoryDescriptions;
}

vector<LayerMemoryDescription> BackPropLayer::InBackPropMemoryDescription() const
{
	if (!inputInterlocked)
		throw runtime_error("The back-prop in layer is not interlocked");

	return inBackPropMemoryDescriptions;
}

vector<LayerMemoryDescription> BackPropLayer::OutBackPropMemoryDescription() const
{
	if (!outputInterlocked)
		throw runtime_error("The back-prop out layer is not interlocked");

	return outBackPropMemoryDescriptions;
}

void BackPropLayer::InterlockForwardPropInput(
		const vector<LayerMemoryDescription>& inputDescriptions)
{
	if (forwardInputInterlocked)
		throw runtime_error("The forward-prop in is already interlocked");

	if (!InterlockHelper::IsCompatible(inForwardPropMemoryProposals,
			inputDescriptions))
		throw runtime_error(
				"The in forward memory description is incompatible");

	inForwardPropMemoryDescriptions = inputDescriptions;
	forwardInputInterlocked = true;
}

void BackPropLayer::InterlockBackPropInput(
		const vector<LayerMemoryDescription>& inputDescriptions)
{
	if (inputInterlocked)
		throw runtime_error("The layer is already interlocked");

	if (!InterlockHelper::IsCompatible(inBackPropMemoryProposals,
			inputDescriptions))
		throw runtime_error(
				"The in backward memory description is incompatible");

	inBackPropMemoryDescriptions = inputDescriptions;
	inputInterlocked = true;
}

void BackPropLayer::InterlockBackPropOutput(
		const vector<LayerMemoryDescription>& outputDescriptions)
{
	if (outputInterlocked)
		throw runtime_error("The layer is already interlocked");

	if (!InterlockHelper::IsCompatible(outBackPropMemoryProposals,
			outputDescriptions))
		throw runtime_error(
				"The out backward memory description is incompatible");

	outBackPropMemoryDescriptions = outputDescriptions;
	outputInterlocked = true;
}

vector<LayerMemoryDescription> BackPropLayer::InForwardPropMemoryProposal() const
{
	return inForwardPropMemoryProposals;
}

vector<LayerDataDescription> BackPropLayer::InForwardPropDataDescription() const
{
	return inForwardPropDataDescriptions;
}

vector<LayerDataDescription> BackPropLayer::InBackPropDataDescription() const
{
	return inBackPropDataDescriptions;
}

vector<LayerDataDescription> BackPropLayer::OutBackPropDataDescription() const
{
	return inForwardPropDataDescriptions; //Must be equal by definition
}

vector<LayerMemoryDescription> BackPropLayer::InBackPropMemoryProposal() const
{
	return inBackPropMemoryProposals;
}

vector<LayerMemoryDescription> BackPropLayer::OutBackPropMemoryProposal() const
{
	return outBackPropMemoryProposals;
}

} /* namespace MachineLearning */
} /* namespace ATML */
