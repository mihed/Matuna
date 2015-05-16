/*
 * CNNFactoryVisitor.cpp
 *
 *  Created on: May 5, 2015
 *      Author: Mikael
 */

#include "CNNFactoryVisitor.h"
#include "InterlockHelper.h"
#include "CNN.h"
#include <stdexcept>
#include <type_traits>

namespace ATML
{
namespace MachineLearning
{

CNNFactoryVisitor::CNNFactoryVisitor(CNN* network) :
		network(network)
{
	backPropActivation = ATMLLinearActivation;
}

CNNFactoryVisitor::~CNNFactoryVisitor()
{

}

void CNNFactoryVisitor::InterlockLayer(BackPropLayer* layer)
{
	if (!InterlockHelper::IsCompatible(layer->InForwardPropDataDescriptions(),
			layer->InForwardPropMemoryProposals()))
		throw runtime_error(
				"We have incompatible data with the memory proposal");

	if (!InterlockHelper::IsCompatible(layer->InBackPropDataDescriptions(),
			layer->InBackPropMemoryProposals()))
		throw runtime_error(
				"We have incompatible data with the memory proposal");

	if (!InterlockHelper::IsCompatible(layer->OutBackPropDataDescriptions(),
			layer->OutBackPropMemoryProposals()))
		throw runtime_error(
				"We have incompatible data with the memory proposal");

	if (!InterlockHelper::DataEquals(inputDataDescriptions,
			layer->OutBackPropDataDescriptions()))
		throw runtime_error("Invalid data description");
	if (!InterlockHelper::DataEquals(inputDataDescriptions,
			layer->InForwardPropDataDescriptions()))
		throw runtime_error("Invalid data description");

	if (!InterlockHelper::IsCompatible(layer->OutBackPropMemoryProposals(),
			backOutputProposals))
		throw runtime_error("Invalid memory description");

	if (!InterlockHelper::IsCompatible(layer->InForwardPropMemoryProposals(),
			forwardInputProposals))
		throw runtime_error("Invalid memory description");

	auto backPropOutputMemory = InterlockHelper::CalculateCompatibleMemory(
			layer->OutBackPropMemoryProposals(), backOutputProposals);

	auto forwardPropInput = InterlockHelper::CalculateCompatibleMemory(
			layer->InForwardPropMemoryProposals(), forwardInputProposals);

	layer->InterlockBackPropOutput(backPropOutputMemory);
	layer->InterlockForwardPropInput(forwardPropInput);

	//Finally we need to perform interlocking on the previous layer's output units
	if (layers.size() != 0)
	{
		auto& forwardPropLayer = layers[layers.size() - 1];

		if (!InterlockHelper::IsCompatible(
				forwardPropLayer->OutForwardPropDataDescriptions(),
				forwardPropLayer->OutForwardPropMemoryProposals()))
			throw runtime_error(
					"We have incompatible data with the memory proposal");

		if (!InterlockHelper::DataEquals(
				forwardPropLayer->InBackPropDataDescriptions(),
				forwardPropLayer->OutForwardPropDataDescriptions()))
			throw runtime_error(
					"The backprop input and forward output units are different");

		if (!InterlockHelper::IsCompatible(
				forwardPropLayer->InBackPropDataDescriptions(),
				forwardPropLayer->InBackPropMemoryProposals()))
			throw runtime_error(
					"We have incompatible data with the memory proposal");

		if (!InterlockHelper::DataEquals(
				forwardPropLayer->OutForwardPropDataDescriptions(),
				layer->InForwardPropDataDescriptions()))
			throw runtime_error(
					"The previous left output description doesn't match the right input description");

		if (!InterlockHelper::DataEquals(
				forwardPropLayer->InBackPropDataDescriptions(),
				layer->OutBackPropDataDescriptions()))
			throw runtime_error(
					"The previous left output description doesn't match the right input description");

		auto forwardPropOutputMemory =
				InterlockHelper::CalculateCompatibleMemory(
						forwardPropLayer->OutForwardPropMemoryProposals(),
						layer->InForwardPropMemoryProposals());

		auto backPropInputMemory = InterlockHelper::CalculateCompatibleMemory(
				forwardPropLayer->InBackPropMemoryProposals(),
				layer->OutBackPropMemoryProposals());

		forwardPropLayer->InterlockForwardPropOutput(forwardPropOutputMemory);
		forwardPropLayer->InterlockBackPropInput(backPropInputMemory);

		if (!forwardPropLayer->Interlocked())
			throw runtime_error(
					"The back-forward prop layer is not interlocked");

		//Need to call this to indicate to the layer that the interlock has been finalized
		forwardPropLayer->InterlockFinalized();
	}
	else //Interlock the CNN here
	{
		if (!InterlockHelper::DataEquals(network->InputForwardDataDescriptions(),
				layer->OutBackPropDataDescriptions()))
			throw runtime_error(
					"The previous left output description doesn't match the right input description");

		if (!InterlockHelper::DataEquals(network->InputForwardDataDescriptions(),
			layer->InForwardPropDataDescriptions()))
			throw runtime_error(
			"The previous left output description doesn't match the right input description");

		network->InterlockForwardPropInput(layer->InForwardPropMemoryDescriptions());
		network->InterlockBackPropOutput(layer->OutBackPropMemoryDescriptions());
	}
}

void CNNFactoryVisitor::InterlockLayer(ForwardBackPropLayer* layer)
{
	static_assert(is_base_of<BackPropLayer, ForwardBackPropLayer>::value,
			"ForwardBackPropLayer is not a sub class of BackPropLayer");

	InterlockLayer(dynamic_cast<BackPropLayer*>(layer));

	forwardInputProposals = layer->OutForwardPropMemoryProposals();
	backOutputProposals = layer->InBackPropMemoryProposals();
	inputDataDescriptions = layer->OutForwardPropDataDescriptions();
}

vector<unique_ptr<ForwardBackPropLayer>> CNNFactoryVisitor::GetLayers()
{
	vector<unique_ptr<ForwardBackPropLayer>> result;

	for (auto& layer : layers)
		result.push_back(move(layer));

	layers.clear();

	return result;
}

unique_ptr<OutputLayer> CNNFactoryVisitor::GetOutputLayer()
{
	return move(outputLayer);
}

} /* namespace MachineLearning */
} /* namespace ATML */
