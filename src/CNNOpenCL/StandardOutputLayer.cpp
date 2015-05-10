/*
 * StandardOutputLayer.cpp
 *
 *  Created on: May 5, 2015
 *      Author: Mikael
 */

#include "StandardOutputLayer.h"

namespace ATML
{
namespace MachineLearning
{

template class StandardOutputLayer<cl_float>;
template class StandardOutputLayer<cl_double>;

template<class T>
StandardOutputLayer<T>::StandardOutputLayer(shared_ptr<OpenCLContext> context,
		const vector<LayerDataDescription>& inputLayerDescriptions,
		const OutputLayerConfig* outputLayerConfig) :
		OutputLayer(inputLayerDescriptions, outputLayerConfig), context(context)
{
	//The targets must have the same data descriptions as the inputs
	inBackPropDataDescriptions = inputLayerDescriptions;

	for (auto& inputDescription : inputLayerDescriptions)
	{
		LayerMemoryDescription inBackPropMemProp;
		inBackPropMemProp.Height = inputDescription.Height;
		inBackPropMemProp.Width = inputDescription.Width;
		inBackPropMemProp.Units = inputDescription.Units;
		inBackPropMemProp.UnitOffset = 0;
		inBackPropMemProp.WidthOffset = 0;
		inBackPropMemProp.HeightOffset = 0;
		inBackPropMemoryProposals.push_back(inBackPropMemProp);
		outBackPropMemoryProposals.push_back(inBackPropMemProp);
		inForwardPropMemoryProposals.push_back(inBackPropMemProp);
	}
}

template<class T>
StandardOutputLayer<T>::~StandardOutputLayer()
{

}

template<class T>
void StandardOutputLayer<T>::InterlockFinalized()
{

}

template<class T>
void StandardOutputLayer<T>::EnqueueBackPropagation(
		shared_ptr<OpenCLMemory> previousInput, shared_ptr<OpenCLMemory> target,
		shared_ptr<OpenCLMemory> deltaOutput)
{

}

} /* namespace MachineLearning */
} /* namespace ATML */
