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

BackPropLayer::BackPropLayer(const LayerDataDescription& inputLayerDescription) :
		inForwardPropDataDescription(inputLayerDescription)
{
	inputInterlocked = false;
	outputInterlocked = false;
	forwardInputInterlocked = false;
}

BackPropLayer::~BackPropLayer()
{

}

LayerMemoryDescription BackPropLayer::InForwardPropMemoryDescription() const
{
	if (!forwardInputInterlocked)
		throw runtime_error("The forward-prop in layer is not interlocked");

	return inForwardPropMemoryDescription;
}

void BackPropLayer::InterlockForwardPropInput(
		const LayerMemoryDescription& inputDescription)
{
	if (forwardInputInterlocked)
		throw runtime_error("The forward-prop in is already interlocked");

	if (!InterlockHelper::IsCompatible(inForwardPropMemoryProposal,
			inputDescription))
		throw runtime_error(
				"The in forward memory description is incompatible");

	inForwardPropMemoryDescription = inputDescription;
	forwardInputInterlocked = true;
}

LayerMemoryDescription BackPropLayer::InBackPropMemoryDescription() const
{
	if (!inputInterlocked)
		throw runtime_error("The back-prop in layer is not interlocked");

	return inBackPropMemoryDescription;
}

LayerMemoryDescription BackPropLayer::OutBackPropMemoryDescription() const
{
	if (!outputInterlocked)
		throw runtime_error("The back-prop out layer is not interlocked");

	return outBackPropMemoryDescription;
}
;

void BackPropLayer::InterlockBackPropInput(
		const LayerMemoryDescription& inputDescription)
{
	if (inputInterlocked)
		throw runtime_error("The layer is already interlocked");

	if (!InterlockHelper::IsCompatible(inBackPropMemoryProposal,
			inputDescription))
		throw runtime_error(
				"The in backward memory description is incompatible");

	inBackPropMemoryDescription = inputDescription;
	inputInterlocked = true;
}

void BackPropLayer::InterlockBackPropOutput(
		const LayerMemoryDescription& outputDescription)
{
	if (outputInterlocked)
		throw runtime_error("The layer is already interlocked");

	if (!InterlockHelper::IsCompatible(outBackPropMemoryProposal,
			outputDescription))
		throw runtime_error(
				"The out backward memory description is incompatible");

	outBackPropMemoryDescription = outputDescription;
	outputInterlocked = true;
}

} /* namespace MachineLearning */
} /* namespace ATML */
