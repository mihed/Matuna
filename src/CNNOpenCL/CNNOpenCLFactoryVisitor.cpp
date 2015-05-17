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

#include "CNN/CNN.h"
#include "CNN/PerceptronLayerConfig.h"
#include "CNN/StandardOutputLayerConfig.h"
#include "CNN/ConvolutionLayerConfig.h"
#include "CNN/CNNConfig.h"
#include "CNN/InterlockHelper.h"

namespace ATML
{
namespace MachineLearning
{

template class CNNOpenCLFactoryVisitor<cl_float> ;
template class CNNOpenCLFactoryVisitor<cl_double> ;

template<class T>
CNNOpenCLFactoryVisitor<T>::CNNOpenCLFactoryVisitor(
		shared_ptr<OpenCLContext> context, CNN* network) :
		CNNFactoryVisitor(network), context(context)
{

}

template<class T>
CNNOpenCLFactoryVisitor<T>::~CNNOpenCLFactoryVisitor()
{

}

template<class T>
void CNNOpenCLFactoryVisitor<T>::Visit(const CNNConfig* const cnnConfig)
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

template<class T>
void CNNOpenCLFactoryVisitor<T>::Visit(
		const PerceptronLayerConfig* const perceptronConfig)
{
	unique_ptr<ForwardBackPropLayer> layer(
			new PerceptronLayer<T>(context, inputDataDescriptions, backPropActivation,
					perceptronConfig));

	backPropActivation = perceptronConfig->ActivationFunction();

	//The forwardInputProposal, backOutputProposal and inputDataDescription are set here.
	InterlockLayer(layer.get());

	layers.push_back(move(layer));
}

template<class T>
void CNNOpenCLFactoryVisitor<T>::Visit(
		const ConvolutionLayerConfig* const convolutionConfig)
{
	unique_ptr<ForwardBackPropLayer> layer(
			new ConvolutionLayer<T>(context, inputDataDescriptions, backPropActivation,
					convolutionConfig));

	backPropActivation = convolutionConfig->ActivationFunction();

	//The forwardInputProposal, backOutputProposal and inputDataDescription are set here.
	InterlockLayer(layer.get());

	layers.push_back(move(layer));
}
template<class T>
void CNNOpenCLFactoryVisitor<T>::Visit(
		const StandardOutputLayerConfig* const outputConfig)
{
	unique_ptr<OutputLayer> layer(
			new StandardOutputLayer<T>(context, inputDataDescriptions, backPropActivation,
					outputConfig));

	//TODO: If we end up with more configs after this call we should throw an exception

	InterlockLayer(layer.get());

	//Since this is an output layer, this will define the targets.
	//We could potentially have some value in the config if we want to do something about this.
	layer->InterlockBackPropInput(layer->InBackPropMemoryProposals());

	if (!layer->Interlocked())
		throw runtime_error("The output layer is not interlocked");

	layer->InterlockFinalized();

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

} /* namespace MachineLearning */
} /* namespace ATML */
