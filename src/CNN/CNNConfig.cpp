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

void CNNConfig::AddToBack(unique_ptr<ILayerConfig> config)
{
	configs.push_back(move(config));
}

void CNNConfig::AddToFront(unique_ptr<ILayerConfig> config)
{
	configs.insert(configs.begin(), move(config));
}

void CNNConfig::InsertAt(unique_ptr<ILayerConfig> config, size_t index)
{
	configs.insert(configs.begin() + index, move(config));
}

void CNNConfig::RemoveAt(size_t index)
{

}

void CNNConfig::Accept(ILayerConfigVisitor* visitor)
{
	visitor->Visit(this);

	for (auto& config : configs)
		config->Accept(visitor);
}

} /* namespace MachineLearning */
} /* namespace ATML */
