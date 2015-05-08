/*
 * ConvolutionLayer.cpp
 *
 *  Created on: May 5, 2015
 *      Author: Mikael
 */

#include "ConvolutionLayer.h"

namespace ATML
{
namespace MachineLearning
{

ConvolutionLayer::ConvolutionLayer(shared_ptr<OpenCLContext> context,
		const vector<LayerDataDescription>& inputLayerDescriptions,
		const ConvolutionLayerConfig* config) :
		OpenCLForwardBackPropLayer(context, inputLayerDescriptions, config)
{

}

ConvolutionLayer::~ConvolutionLayer()
{

}

void ConvolutionLayer::EnqueueForwardPropagation(
		shared_ptr<OpenCLMemory> previousInput, shared_ptr<OpenCLMemory> output)
{

}

void ConvolutionLayer::EnqueueBackPropagation(
		shared_ptr<OpenCLMemory> previousInput, shared_ptr<OpenCLMemory> delta,
		shared_ptr<OpenCLMemory> deltaOutput)
{

}

} /* namespace MachineLearning */
} /* namespace ATML */
