/*
 * CNNConfig.cpp
 *
 *  Created on: May 2, 2015
 *      Author: Mikael
 */

#include "CNNConfig.h"
#include "ILayerConfigVisitor.h"

namespace ATML
{
namespace MachineLearning
{

CNNConfig::CNNConfig(const vector<LayerDataDescription>& dataDescriptions) :
		inputDataDescriptions(dataDescriptions)
{
	for (auto& dataDescription : dataDescriptions)
	{
		LayerMemoryDescription inputMemoryProposal;
		inputMemoryProposal.Height = dataDescription.Height;
		inputMemoryProposal.Width = dataDescription.Width;
		inputMemoryProposal.Units = dataDescription.Units;
		inputMemoryProposal.WidthOffset = 0;
		inputMemoryProposal.HeightOffset = 0;
		inputMemoryProposal.UnitOffset = 0;
		inputMemoryProposals.push_back(inputMemoryProposal);
	}
}

CNNConfig::CNNConfig(const vector<LayerDataDescription>& dataDescription,
		const vector<LayerMemoryDescription>& memoryProposal) :
		inputMemoryProposals(memoryProposal), inputDataDescriptions(
				dataDescription)
{

}

CNNConfig::~CNNConfig()
{

}

size_t CNNConfig::LayerCount() const
{
	return forwardBackConfigs.size();
}
;

bool CNNConfig::HasOutputLayer() const
{
	if (outputConfig.get())
		return true;
	else
		return false;
}

void CNNConfig::SetOutputConfig(unique_ptr<OutputLayerConfig> config)
{
	outputConfig = move(config);
}

void CNNConfig::RemoveOutputConfig()
{
	outputConfig.reset(nullptr);
}

void CNNConfig::AddToBack(unique_ptr<ForwardBackPropLayerConfig> config)
{
	forwardBackConfigs.push_back(move(config));
}

void CNNConfig::AddToFront(unique_ptr<ForwardBackPropLayerConfig> config)
{
	forwardBackConfigs.insert(forwardBackConfigs.begin(), move(config));
}

void CNNConfig::InsertAt(unique_ptr<ForwardBackPropLayerConfig> config,
		size_t index)
{
	forwardBackConfigs.insert(forwardBackConfigs.begin() + index, move(config));
}

void CNNConfig::RemoveAt(size_t index)
{
	forwardBackConfigs.erase(forwardBackConfigs.begin() + index);
}

vector<LayerMemoryDescription> CNNConfig::InputMemoryProposal() const
{
	return inputMemoryProposals;
}

vector<LayerDataDescription> CNNConfig::InputDataDescription() const
{
	return inputDataDescriptions;
}

void CNNConfig::Accept(ILayerConfigVisitor* visitor)
{
	visitor->Visit(this);

	for (auto& config : forwardBackConfigs)
		config->Accept(visitor);

	if (outputConfig.get())
		outputConfig->Accept(visitor);
}

} /* namespace MachineLearning */
} /* namespace ATML */
