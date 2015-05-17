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
#include "CNN/CNN.h"
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

	//Initialization. We assume that our data is as slim as possible
	vector<LayerMemoryDescription> inputMemory;
	for (auto& data : inputData)
	{
		LayerMemoryDescription memory;
		memory.Width = data.Width;
		memory.Height = data.Height;
		memory.Units = data.Units;
		memory.HeightOffset = 0;
		memory.WidthOffset = 0;
		memory.UnitOffset = 0;
		inputMemory.push_back(memory);
	}

	forwardInputProposals = inputMemory;
	backOutputProposals = inputMemory;
	inputDataDescriptions = inputData;
}

void CNNFactoryVisitorTest::Visit(
		const PerceptronLayerConfig* const perceptronConfig)
{
	unique_ptr<ForwardBackPropLayer> layer(
			new ForthBackPropLayerTest(inputDataDescriptions, backPropActivation, perceptronConfig));

	backPropActivation = perceptronConfig->ActivationFunction();

	InterlockLayer(layer.get());

	layers.push_back(move(layer));
}

void CNNFactoryVisitorTest::Visit(
		const ConvolutionLayerConfig* const convolutionConfig)
{
	unique_ptr<ForwardBackPropLayer> layer(
			new ForthBackPropLayerTest(inputDataDescriptions,
			backPropActivation,
					convolutionConfig));

	backPropActivation = convolutionConfig->ActivationFunction();

	InterlockLayer(layer.get());

	layers.push_back(move(layer));
}

void CNNFactoryVisitorTest::Visit(
		const StandardOutputLayerConfig* const convolutionConfig)
{
	unique_ptr<OutputLayer> layer(
		new OutputLayerTest(inputDataDescriptions, backPropActivation, convolutionConfig));

	InterlockLayer(layer.get());

	//Since this is an output layer, this will define the targets.
	//We could potentially have some value in the config if we want to do something about this.
	layer->InterlockBackPropInput(layer->InBackPropMemoryProposals());

	if (!layer->Interlocked())
		throw runtime_error("The output layer is not interlocked");

	network->InterlockForwardPropDataOutput(layer->InBackPropDataDescriptions());
	network->InterlockForwardPropOutput(layer->InBackPropMemoryDescriptions());

	//FIXME: This is hacky, need fixing!
	if (layers.size() != 0)
	{
		auto& firstLayer = layers[0];
		network->InterlockBackPropOutput(firstLayer->InBackPropMemoryDescriptions());
		network->InterlockBackPropDataOutput(firstLayer->InBackPropDataDescriptions());
	}
	else
	{
		network->InterlockBackPropOutput(layer->InBackPropMemoryDescriptions());
		network->InterlockBackPropDataOutput(layer->InBackPropDataDescriptions());
	}

	if (!network->Interlocked())
		throw runtime_error("The network is not interlocked");

	outputLayer = move(layer);
}

