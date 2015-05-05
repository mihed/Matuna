/*
 * CNNFactoryVisitor.cpp
 *
 *  Created on: May 5, 2015
 *      Author: Mikael
 */

#include "CNNFactoryVisitor.h"
#include "InterlockHelper.h"
#include <stdexcept>

namespace ATML
{
namespace MachineLearning
{

CNNFactoryVisitor::CNNFactoryVisitor()
{

}

CNNFactoryVisitor::~CNNFactoryVisitor()
{

}

void CNNFactoryVisitor::InterlockLayer(BackPropLayer* layer)
{
	if (!InterlockHelper::IsCompatible(layer->InForwardPropDataDescription(),
			layer->InForwardPropMemoryProposal()))
		throw runtime_error(
				"We have incompatible data with the memory proposal");

	if (!InterlockHelper::IsCompatible(layer->InBackPropDataDescription(),
			layer->InBackPropMemoryProposal()))
		throw runtime_error(
				"We have incompatible data with the memory proposal");

	if (!InterlockHelper::IsCompatible(layer->OutBackPropDataDescription(),
			layer->OutBackPropMemoryProposal()))
		throw runtime_error(
				"We have incompatible data with the memory proposal");

	if (!InterlockHelper::DataEquals(inputDataDescription,
			layer->OutBackPropDataDescription()))
		throw runtime_error("Invalid data description");
	if (!InterlockHelper::DataEquals(inputDataDescription,
			layer->InForwardPropDataDescription()))
		throw runtime_error("Invalid data description");

	if (!InterlockHelper::IsCompatible(layer->OutBackPropMemoryProposal(),
			backOutputProposal))
		throw runtime_error("Invalid memory description");

	if (!InterlockHelper::IsCompatible(layer->InForwardPropMemoryProposal(),
			forwardInputProposal))
		throw runtime_error("Invalid memory description");

	auto backPropOutputMemory = InterlockHelper::CalculateCompatibleMemory(
			layer->OutBackPropMemoryProposal(), backOutputProposal);

	auto forwardPropInput = InterlockHelper::CalculateCompatibleMemory(
			layer->InForwardPropMemoryProposal(), forwardInputProposal);

	layer->InterlockBackPropOutput(backPropOutputMemory);
	layer->InterlockForwardPropInput(forwardPropInput);

	//Finally we need to perform interlocking on the previous layer's output units
	if (layers.size() != 0)
	{
		auto& forwardPropLayer = layers[layers.size() - 1];

		if (!InterlockHelper::IsCompatible(
				forwardPropLayer->OutForwardPropDataDescription(),
				forwardPropLayer->OutForwardPropMemoryProposal()))
			throw runtime_error(
					"We have incompatible data with the memory proposal");

		if (!InterlockHelper::DataEquals(
				forwardPropLayer->InBackPropDataDescription(),
				forwardPropLayer->OutForwardPropDataDescription()))
			throw runtime_error(
					"The backprop input and forward output units are different");

		if (!InterlockHelper::IsCompatible(
				forwardPropLayer->InBackPropDataDescription(),
				forwardPropLayer->InBackPropMemoryProposal()))
			throw runtime_error(
					"We have incompatible data with the memory proposal");

		if (!InterlockHelper::DataEquals(
				forwardPropLayer->OutForwardPropDataDescription(),
				layer->InForwardPropDataDescription()))
			throw runtime_error(
					"The previous left output description doesn't match the right input description");

		if (!InterlockHelper::DataEquals(
				forwardPropLayer->InBackPropDataDescription(),
				layer->OutBackPropDataDescription()))
			throw runtime_error(
					"The previous left output description doesn't match the right input description");

		auto forwardPropOutputMemory =
				InterlockHelper::CalculateCompatibleMemory(
						forwardPropLayer->OutForwardPropMemoryProposal(),
						layer->InForwardPropMemoryProposal());

		auto backPropInputMemory = InterlockHelper::CalculateCompatibleMemory(
				forwardPropLayer->InBackPropMemoryProposal(),
				layer->OutBackPropMemoryProposal());

		forwardPropLayer->InterlockForwardPropOutput(forwardPropOutputMemory);
		forwardPropLayer->InterlockBackPropInput(backPropInputMemory);

		if (!forwardPropLayer->Interlocked())
			throw runtime_error(
					"The back-forward prop layer is not interlocked");
	}
}

void CNNFactoryVisitor::InterlockLayer(ForwardBackPropLayer* layer)
{
	InterlockLayer(dynamic_cast<BackPropLayer*>(layer));

	forwardInputProposal = layer->OutForwardPropMemoryProposal();
	backOutputProposal = layer->InBackPropMemoryProposal();
	inputDataDescription = layer->OutForwardPropDataDescription();
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
