/*
 * PerceptronLayer.cpp
 *
 *  Created on: May 5, 2015
 *      Author: Mikael
 */

#include "PerceptronLayer.h"
#include "CNN/InterlockHelper.h"
#include <stdexcept>

namespace ATML
{
namespace MachineLearning
{

template class PerceptronLayer<cl_float>;
template class PerceptronLayer<cl_double>;

template<class T>
PerceptronLayer<T>::PerceptronLayer(shared_ptr<OpenCLContext> context,
		const vector<LayerDataDescription>& inputLayerDescriptions,
		const PerceptronLayerConfig* config) :
		OpenCLForwardBackPropLayer<T>(context, inputLayerDescriptions, config)
{
	//In a perceptron layer, we cannot have multiple input descriptions for the same network
	//since it will correspond to a different weight matrix.
	if (inputLayerDescriptions.size() > 1)
	{
		auto count = inputLayerDescriptions.size();
		for (int i = 1; i < count; i++)
			if (!InterlockHelper::DataEquals(inputLayerDescriptions[i - 1],
					inputLayerDescriptions[i]))
				throw invalid_argument(
						"We cannot have multiple different input descriptions for a perceptron layer");
	}

	for (auto& layerDescription : inputLayerDescriptions)
	{
		LayerMemoryDescription inForwardMemProp;
		inForwardMemProp.Height = layerDescription.Height;
		inForwardMemProp.Width = layerDescription.Width;
		inForwardMemProp.Units = layerDescription.Units;
		inForwardMemProp.HeightOffset = 0;
		inForwardMemProp.UnitOffset = 0;
		inForwardMemProp.WidthOffset = 0;
		this->inForwardPropMemoryProposals.push_back(inForwardMemProp);
		this->outBackPropMemoryProposals.push_back(inForwardMemProp);

		LayerDataDescription outForwardDataDesc;
		outForwardDataDesc.Height = 1;
		outForwardDataDesc.Width = 1;
		outForwardDataDesc.Units = config->Units();
		this->outForwardPropDataDescriptions.push_back(outForwardDataDesc);

		this->inBackPropDataDescriptions = this->outForwardPropDataDescriptions;

		LayerMemoryDescription outForwardMemProp;
		outForwardMemProp.Height = 1;
		outForwardMemProp.Width = 1;
		outForwardMemProp.Units = config->Units();
		outForwardMemProp.HeightOffset = 0;
		outForwardMemProp.UnitOffset = 0;
		outForwardMemProp.WidthOffset = 0;
		this->outForwardPropMemoryProposals.push_back(outForwardMemProp);

		this->inBackPropMemoryProposals.push_back(outForwardMemProp);
	}

	//We are not using any automatic tuning atm.
	if (inputLayerDescriptions.size() == 0)
		throw invalid_argument(
				"There's no input data descriptions for the perceptron layer.");
}

template<class T>
PerceptronLayer<T>::~PerceptronLayer()
{
	if (forwardPerceptronKernel.get())
		this->context->RemoveProgram(forwardPerceptronKernel->ProgramName());
}

template<class T>
void PerceptronLayer<T>::InterlockFinalized()
{
	auto inputMemoryDescriptions = this->InForwardPropMemoryDescription();
	auto inputDataDescriptions = this->InForwardPropDataDescription();
	auto& firstMemory = inputMemoryDescriptions[0];
	auto& firstData = inputDataDescriptions[0];

	//IF the memory descriptions doesn't contain any padding or offsets, we may use the standard forward prop kernel.
	if (firstMemory.HeightOffset == 0 && firstMemory.UnitOffset == 0
			&& firstMemory.WidthOffset == 0
			&& firstMemory.Width == firstData.Width
			&& firstMemory.Height == firstData.Height
			&& firstMemory.Units == firstData.Units)
		InitializeNormalPerceptron();
	else
		InitializeImagePerceptron();
}

template<class T>
void PerceptronLayer<T>::InitializeNormalPerceptron()
{
	//FIXME: We cannot use the context interface here. Since we need to evaluate pre-processors options.

	auto inputDataDescriptions = this->InForwardPropDataDescription();
	auto& firstData = inputDataDescriptions[0];

}

template<class T>
void PerceptronLayer<T>::InitializeImagePerceptron()
{
	throw runtime_error("Not implemented");
}

template<class T>
void PerceptronLayer<T>::EnqueueForwardPropagation(
		shared_ptr<OpenCLMemory> previousInput, shared_ptr<OpenCLMemory> output)
{
//FIXME: I don't know which device to enqueue to.
}

template<class T>
void PerceptronLayer<T>::EnqueueBackPropagation(
		shared_ptr<OpenCLMemory> previousInput, shared_ptr<OpenCLMemory> delta,
		shared_ptr<OpenCLMemory> deltaOutput)
{
	throw runtime_error("Not implemented");
}

} /* namespace MachineLearning */
} /* namespace ATML */
