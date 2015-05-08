/*
 * CNN.cpp
 *
 *  Created on: May 2, 2015
 *      Author: Mikael
 */

#include "CNN.h"
#include <stdexcept>

namespace ATML
{
namespace MachineLearning
{

CNN::CNN(const CNNConfig& config)
{
	inputDataDescriptions = config.InputDataDescription();
	inputMemoryProposals = config.InputMemoryProposal();
	inputInterlocked = false;

	//It is completely forbidding to call CNNFactoryVisitor here.
	//However, it will work in any base classes since this constructor is instantiated first.
}

CNN::~CNN()
{

}

void CNN::InterlockForwardPropInput(
		const vector<LayerMemoryDescription>& inputDescriptions)
{
	if (inputInterlocked)
		throw runtime_error("The input is already interlocked");

	inputMemoryDescriptions = inputDescriptions;
	inputInterlocked = true;
}

bool CNN::Interlocked() const
{
	return inputInterlocked;
}

vector<LayerDataDescription> CNN::InputDataDescriptions() const
{
	return inputDataDescriptions;
}

vector<LayerMemoryDescription> CNN::InputMemoryDescriptions() const
{
	return inputMemoryDescriptions;
}

vector<LayerMemoryDescription> CNN::InputMemoryProposals() const
{
	return inputMemoryProposals;
}

} /* namespace MachineLearning */
} /* namespace ATML */
