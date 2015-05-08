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
