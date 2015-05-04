/*
 * FactoryVisitorTest.cpp
 *
 *  Created on: May 3, 2015
 *      Author: Mikael
 */

#include "FactoryVisitorTest.h"
#include "CNN/CNNConfig.h"
#include "CNN/InterlockHelper.h"

#include "OutputLayerConfigTest.h"
#include "OutputLayerTest.h"
#include "ForthBackPropLayerConfigTest.h"
#include "ForthBackPropLayerTest.h"
#include <stdexcept>

void FactoryVisitorTest::InterlockLayer(BackPropLayer* layer)
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
		auto& previousLayer = layers[layers.size() - 1];
		auto forwardPropLayer =
				dynamic_cast<ForwardBackPropLayer*>(previousLayer.get());

		if (!forwardPropLayer)
			throw runtime_error(
					"Invalid configuration. There's a back-prop only layer that is not located in the back");

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

void FactoryVisitorTest::InterlockLayer(ForwardBackPropLayer* layer)
{
	InterlockLayer(dynamic_cast<BackPropLayer*>(layer));

	forwardInputProposal = layer->OutForwardPropMemoryProposal();
	backOutputProposal = layer->InBackPropMemoryProposal();
	inputDataDescription = layer->OutForwardPropDataDescription();
}

FactoryVisitorTest::FactoryVisitorTest()
{

}

FactoryVisitorTest::~FactoryVisitorTest()
{

}

void FactoryVisitorTest::Visit(const CNNConfig* const config)
{
	auto inputData = config->InputDataDescription();
	auto inputMemory = config->InputMemoryProposal();
	if (!InterlockHelper::IsCompatible(inputData, inputMemory))
		throw runtime_error("Invalid cnn config memory and data description");

	forwardInputProposal = inputMemory;
	backOutputProposal = inputMemory;
	inputDataDescription = inputData;
}

void FactoryVisitorTest::Visit(const OutputLayerConfigTest* const config)
{
	unique_ptr<OutputLayerTest> layer(
			new OutputLayerTest(inputDataDescription, *config));

	InterlockLayer(layer.get());

	//Since this is an output layer, this will define the targets.
	//We could potentially have some value in the config if we want to do something about this.
	layer->InterlockBackPropInput(layer->InBackPropMemoryProposal());

	if (!layer->Interlocked())
		throw runtime_error("The output layer is not interlocked");

	layers.push_back(move(layer));
}

void FactoryVisitorTest::Visit(const ForthBackPropLayerConfigTest* const config)
{
	unique_ptr<ForthBackPropLayerTest> layer(
			new ForthBackPropLayerTest(inputDataDescription, *config));

	InterlockLayer(dynamic_cast<ForwardBackPropLayer*>(layer.get()));

	layers.push_back(move(layer));
}

vector<unique_ptr<BackPropLayer>> FactoryVisitorTest::GetLayers()
{
	vector<unique_ptr<BackPropLayer>> result;

	for (auto& layer : layers)
		result.push_back(move(layer));

	layers.clear();

	return result;
}

