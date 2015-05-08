/*
 * CNNFactoryVisitorTest.cpp
 *
 *  Created on: May 5, 2015
 *      Author: Mikael
 */

#include "CNN/PerceptronLayerConfig.h"
#include "CNN/StandardOutputLayerConfig.h"
#include "CNN/ConvolutionLayerConfig.h"
#include "CNN/CNNConfig.h"
#include "CNN/InterlockHelper.h"
#include "CNNFactoryVisitorTest.h"
#include "ForthBackPropLayerTest.h"
#include "OutputLayerTest.h"

CNNFactoryVisitorTest::CNNFactoryVisitorTest(CNN* network)
	: CNNFactoryVisitor(network)
{

}

CNNFactoryVisitorTest::~CNNFactoryVisitorTest()
{

}

void CNNFactoryVisitorTest::Visit(const CNNConfig* const cnnConfig)
{
	auto inputData = cnnConfig->InputDataDescription();
	auto inputMemory = cnnConfig->InputMemoryProposal();
	if (!InterlockHelper::IsCompatible(inputData, inputMemory))
		throw runtime_error("Invalid cnn config memory and data description");

	forwardInputProposals = inputMemory;
	backOutputProposals = inputMemory;
	inputDataDescriptions = inputData;
}

void CNNFactoryVisitorTest::Visit(
		const PerceptronLayerConfig* const perceptronConfig)
{
	unique_ptr<ForwardBackPropLayer> layer(
			new ForthBackPropLayerTest(inputDataDescriptions, perceptronConfig));

	InterlockLayer(layer.get());

	layers.push_back(move(layer));
}

void CNNFactoryVisitorTest::Visit(
		const ConvolutionLayerConfig* const convolutionConfig)
{
	unique_ptr<ForwardBackPropLayer> layer(
			new ForthBackPropLayerTest(inputDataDescriptions,
					convolutionConfig));

	InterlockLayer(layer.get());

	layers.push_back(move(layer));
}

void CNNFactoryVisitorTest::Visit(
		const StandardOutputLayerConfig* const convolutionConfig)
{
	unique_ptr<OutputLayer> layer(
			new OutputLayerTest(inputDataDescriptions, convolutionConfig));

	InterlockLayer(layer.get());

	//Since this is an output layer, this will define the targets.
	//We could potentially have some value in the config if we want to do something about this.
	layer->InterlockBackPropInput(layer->InBackPropMemoryProposal());

	if (!layer->Interlocked())
		throw runtime_error("The output layer is not interlocked");

	outputLayer = move(layer);
}

