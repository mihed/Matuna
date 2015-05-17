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
	inputForwardDataDescriptions = config.InputDataDescription();
	inputInterlocked = false;

	outputInterlocked = false;
	outputDataInterlocked = false;

	outputBackInterlocked = false;
	outputBackDataInterlocked = false;

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

	inputForwardMemoryDescriptions = inputDescriptions;
	inputInterlocked = true;
}

void CNN::InterlockForwardPropOutput(
		const vector<LayerMemoryDescription>& outputDescriptions)
{
	if (outputInterlocked)
		throw runtime_error("The output is already interlocked");

	outputForwardMemoryDescriptions = outputDescriptions;
	outputInterlocked = true;
}

void CNN::InterlockBackPropOutput(
	const vector<LayerMemoryDescription>& outputDescriptions)
{
	if (outputBackInterlocked)
		throw runtime_error("The output data is already interlocked");

	outputBackMemoryDescriptions = outputDescriptions;
	outputBackInterlocked = true;
}

void CNN::InterlockForwardPropDataOutput(
		const vector<LayerDataDescription>& outputDescriptions)
{
	if (outputDataInterlocked)
		throw runtime_error("The output memory is already interlocked");

	outputForwardDataDescriptions = outputDescriptions;
	outputDataInterlocked = true;
}

void CNN::InterlockBackPropDataOutput(
	const vector<LayerDataDescription>& outputDescriptions)
{
	if (outputBackDataInterlocked)
		throw runtime_error("The output data is already interlocked");

	outputBackDataDescriptions = outputDescriptions;
	outputBackDataInterlocked = true;
}

bool CNN::Interlocked() const
{
	return inputInterlocked && outputDataInterlocked && outputInterlocked && outputBackInterlocked && outputBackDataInterlocked;
}

vector<LayerDataDescription> CNN::InputForwardDataDescriptions() const
{
	return inputForwardDataDescriptions;
}

vector<LayerMemoryDescription> CNN::InputForwardMemoryDescriptions() const
{
	return inputForwardMemoryDescriptions;
}

vector<LayerDataDescription> CNN::OutputForwardDataDescriptions() const
{
	return outputForwardDataDescriptions;
}
vector<LayerMemoryDescription> CNN::OutputForwardMemoryDescriptions() const
{
	return outputForwardMemoryDescriptions;
}

vector<LayerMemoryDescription> CNN::OutputBackMemoryDescriptions() const
{
	return outputBackMemoryDescriptions;
}

vector<LayerDataDescription>CNN::OutputBackDataDescriptions() const
{
	return outputBackDataDescriptions;
}


} /* namespace MachineLearning */
} /* namespace ATML */
