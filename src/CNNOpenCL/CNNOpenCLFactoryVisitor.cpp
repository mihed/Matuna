/*
 * CNNOpenCLFactoryVisitor.cpp
 *
 *  Created on: May 5, 2015
 *      Author: Mikael
 */

#include "CNNOpenCLFactoryVisitor.h"
#include "PerceptronLayer.h"
#include "ConvolutionLayer.h"
#include "StandardOutputLayer.h"

#include "CNN/PerceptronLayerConfig.h"
#include "CNN/StandardOutputLayerConfig.h"
#include "CNN/ConvolutionLayerConfig.h"
#include "CNN/CNNConfig.h"
#include "CNN/InterlockHelper.h"

namespace ATML
{
namespace MachineLearning
{

CNNOpenCLFactoryVisitor::CNNOpenCLFactoryVisitor(
		shared_ptr<OpenCLContext> context) :
		context(context)
{

}

CNNOpenCLFactoryVisitor::~CNNOpenCLFactoryVisitor()
{

}

void CNNOpenCLFactoryVisitor::Visit(const CNNConfig* const cnnConfig)
{
	auto inputData = cnnConfig->InputDataDescription();
	auto inputMemory = cnnConfig->InputMemoryProposal();
	if (!InterlockHelper::IsCompatible(inputData, inputMemory))
		throw runtime_error("Invalid cnn config memory and data description");

	forwardInputProposals = inputMemory;
	backOutputProposals = inputMemory;
	inputDataDescriptions = inputData;
}

void CNNOpenCLFactoryVisitor::Visit(
		const PerceptronLayerConfig* const perceptronConfig)
{
	unique_ptr<ForwardBackPropLayer> layer(
			new PerceptronLayer(context, inputDataDescriptions,
					perceptronConfig));

	//The forwardInputProposal, backOutputProposal and inputDataDescription are set here.
	InterlockLayer(layer.get());

	layers.push_back(move(layer));
}

void CNNOpenCLFactoryVisitor::Visit(
		const ConvolutionLayerConfig* const convolutionConfig)
{
	unique_ptr<ForwardBackPropLayer> layer(
			new ConvolutionLayer(context, inputDataDescriptions,
					convolutionConfig));

	//The forwardInputProposal, backOutputProposal and inputDataDescription are set here.
	InterlockLayer(layer.get());

	layers.push_back(move(layer));
}

void CNNOpenCLFactoryVisitor::Visit(
		const StandardOutputLayerConfig* const outputConfig)
{
	unique_ptr<OutputLayer> layer(
			new StandardOutputLayer(context, inputDataDescriptions,
					outputConfig));

	InterlockLayer(layer.get());

	//Since this is an output layer, this will define the targets.
	//We could potentially have some value in the config if we want to do something about this.
	layer->InterlockBackPropInput(layer->InBackPropMemoryProposal());

	if (!layer->Interlocked())
		throw runtime_error("The output layer is not interlocked");

	outputLayer = move(layer);
}

} /* namespace MachineLearning */
} /* namespace ATML */
