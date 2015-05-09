/*
 * PerceptronLayer.cpp
 *
 *  Created on: May 5, 2015
 *      Author: Mikael
 */

#include "PerceptronLayer.h"

namespace ATML
{
namespace MachineLearning
{

PerceptronLayer::PerceptronLayer(shared_ptr<OpenCLContext> context,
		const vector<LayerDataDescription>& inputLayerDescriptions,
		const PerceptronLayerConfig* config) :
		OpenCLForwardBackPropLayer(context, inputLayerDescriptions, config)
{
	for (auto& layerDescription : inputLayerDescriptions)
	{
		LayerMemoryDescription inForwardMemProp;
		inForwardMemProp.Height = layerDescription.Height;
		inForwardMemProp.Width = layerDescription.Width;
		inForwardMemProp.Units = layerDescription.Units;
		inForwardMemProp.HeightOffset = 0;
		inForwardMemProp.UnitOffset = 0;
		inForwardMemProp.WidthOffset = 0;
		inForwardPropMemoryProposals.push_back(inForwardMemProp);
		outBackPropMemoryProposals.push_back(inForwardMemProp);

		LayerDataDescription outForwardDataDesc;
		outForwardDataDesc.Height = 1;
		outForwardDataDesc.Width = 1;
		outForwardDataDesc.Units = config->Units();
		outForwardPropDataDescriptions.push_back(outForwardDataDesc);

		inBackPropDataDescriptions = outForwardPropDataDescriptions;

		LayerMemoryDescription outForwardMemProp;
		outForwardMemProp.Height = 1;
		outForwardMemProp.Width = 1;
		outForwardMemProp.Units = config->Units();
		outForwardMemProp.HeightOffset = 0;
		outForwardMemProp.UnitOffset = 0;
		outForwardMemProp.WidthOffset = 0;
		outForwardPropMemoryProposals.push_back(outForwardMemProp);

		inBackPropMemoryProposals.push_back(outForwardMemProp);
	}
}

PerceptronLayer::~PerceptronLayer()
{

}

void PerceptronLayer::EnqueueForwardPropagation(
		shared_ptr<OpenCLMemory> previousInput, shared_ptr<OpenCLMemory> output)
{

}

void PerceptronLayer::EnqueueBackPropagation(
		shared_ptr<OpenCLMemory> previousInput, shared_ptr<OpenCLMemory> delta,
		shared_ptr<OpenCLMemory> deltaOutput)
{

}

} /* namespace MachineLearning */
} /* namespace ATML */
