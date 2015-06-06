/*
 * ConvNet.cpp
 *
 *  Created on: May 2, 2015
 *      Author: Mikael
 */

#include "ConvNet.h"
#include <stdexcept>

namespace Matuna
{
namespace MachineLearning
{

ConvNet::ConvNet(const ConvNetConfig& config)
{
	inputForwardDataDescriptions = config.InputDataDescription();
	inputInterlocked = false;

	outputInterlocked = false;
	outputDataInterlocked = false;

	outputBackInterlocked = false;
	outputBackDataInterlocked = false;

	//It is completely forbidding to call ConvNetFactoryVisitor here.
	//However, it will work in any base classes since this constructor is instantiated first.
}

ConvNet::~ConvNet()
{

}

void ConvNet::InterlockForwardPropInput(
		const vector<LayerMemoryDescription>& inputDescriptions)
{
	if (inputInterlocked)
		throw runtime_error("The input is already interlocked");

	inputForwardMemoryDescriptions = inputDescriptions;
	inputInterlocked = true;
}

void ConvNet::InterlockForwardPropOutput(
		const vector<LayerMemoryDescription>& outputDescriptions)
{
	if (outputInterlocked)
		throw runtime_error("The output is already interlocked");

	outputForwardMemoryDescriptions = outputDescriptions;
	outputInterlocked = true;
}

void ConvNet::InterlockBackPropOutput(
	const vector<LayerMemoryDescription>& outputDescriptions)
{
	if (outputBackInterlocked)
		throw runtime_error("The output data is already interlocked");

	outputBackMemoryDescriptions = outputDescriptions;
	outputBackInterlocked = true;
}

void ConvNet::InterlockForwardPropDataOutput(
		const vector<LayerDataDescription>& outputDescriptions)
{
	if (outputDataInterlocked)
		throw runtime_error("The output memory is already interlocked");

	outputForwardDataDescriptions = outputDescriptions;
	outputDataInterlocked = true;
}

void ConvNet::InterlockBackPropDataOutput(
	const vector<LayerDataDescription>& outputDescriptions)
{
	if (outputBackDataInterlocked)
		throw runtime_error("The output data is already interlocked");

	outputBackDataDescriptions = outputDescriptions;
	outputBackDataInterlocked = true;
}

bool ConvNet::Interlocked() const
{
	return inputInterlocked && outputDataInterlocked && outputInterlocked && outputBackInterlocked && outputBackDataInterlocked;
}

vector<LayerDataDescription> ConvNet::InputForwardDataDescriptions() const
{
	return inputForwardDataDescriptions;
}

vector<LayerMemoryDescription> ConvNet::InputForwardMemoryDescriptions() const
{
	return inputForwardMemoryDescriptions;
}

vector<LayerDataDescription> ConvNet::OutputForwardDataDescriptions() const
{
	return outputForwardDataDescriptions;
}
vector<LayerMemoryDescription> ConvNet::OutputForwardMemoryDescriptions() const
{
	return outputForwardMemoryDescriptions;
}

vector<LayerMemoryDescription> ConvNet::OutputBackMemoryDescriptions() const
{
	return outputBackMemoryDescriptions;
}

vector<LayerDataDescription>ConvNet::OutputBackDataDescriptions() const
{
	return outputBackDataDescriptions;
}


} /* namespace MachineLearning */
} /* namespace Matuna */
