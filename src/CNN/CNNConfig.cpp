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

CNNConfig::CNNConfig(const LayerDataDescription& dataDescription) :
		inputDataDescription(dataDescription)
{
	inputMemoryProposal.Height = dataDescription.Height;
	inputMemoryProposal.Width = dataDescription.Width;
	inputMemoryProposal.Units = dataDescription.Units;
	inputMemoryProposal.WidthOffset = 0;
	inputMemoryProposal.HeightOffset = 0;
	inputMemoryProposal.UnitOffset = 0;
}

CNNConfig::CNNConfig(const LayerDataDescription& dataDescription,
		const LayerMemoryDescription& memoryProposal) :
		inputMemoryProposal(memoryProposal), inputDataDescription(
				dataDescription)
{

}

CNNConfig::~CNNConfig()
{

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
