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

StandardOutputLayer::StandardOutputLayer(shared_ptr<OpenCLContext> context,
		const vector<LayerDataDescription>& inputLayerDescriptions,
		const OutputLayerConfig* outputLayerConfig) :
		OutputLayer(inputLayerDescriptions, outputLayerConfig), context(context)
{

}

StandardOutputLayer::~StandardOutputLayer()
{

}

void StandardOutputLayer::EnqueueBackPropagation(
		shared_ptr<OpenCLMemory> previousInput, shared_ptr<OpenCLMemory> target,
		shared_ptr<OpenCLMemory> deltaOutput)
{

}

} /* namespace MachineLearning */
} /* namespace ATML */
