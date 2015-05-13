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

vector<LayerMemoryDescription> BackPropLayer::InForwardPropMemoryDescriptions() const
{
	if (!forwardInputInterlocked)
		throw runtime_error("The forward-prop in layer is not interlocked");

	return inForwardPropMemoryDescriptions;
}

vector<LayerMemoryDescription> BackPropLayer::InBackPropMemoryDescriptions() const
{
	if (!inputInterlocked)
		throw runtime_error("The back-prop in layer is not interlocked");

	return inBackPropMemoryDescriptions;
}

vector<LayerMemoryDescription> BackPropLayer::OutBackPropMemoryDescriptions() const
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

vector<LayerMemoryDescription> BackPropLayer::InForwardPropMemoryProposals() const
{
	return inForwardPropMemoryProposals;
}

vector<LayerDataDescription> BackPropLayer::InForwardPropDataDescriptions() const
{
	return inForwardPropDataDescriptions;
}

vector<LayerDataDescription> BackPropLayer::InBackPropDataDescriptions() const
{
	return inBackPropDataDescriptions;
}

vector<LayerDataDescription> BackPropLayer::OutBackPropDataDescriptions() const
{
	return inForwardPropDataDescriptions; //Must be equal by definition
}

vector<LayerMemoryDescription> BackPropLayer::InBackPropMemoryProposals() const
{
	return inBackPropMemoryProposals;
}

vector<LayerMemoryDescription> BackPropLayer::OutBackPropMemoryProposals() const
{
	return outBackPropMemoryProposals;
}

} /* namespace MachineLearning */
} /* namespace ATML */
