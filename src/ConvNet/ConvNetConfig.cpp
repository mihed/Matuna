/*
 * ConvNetConfig.cpp
 *
 *  Created on: May 2, 2015
 *      Author: Mikael
 */

#include "ConvNetConfig.h"
#include "ILayerConfigVisitor.h"

namespace Matuna
{
namespace MachineLearning
{

ConvNetConfig::ConvNetConfig(const vector<LayerDataDescription>& dataDescriptions) :
		inputDataDescriptions(dataDescriptions)
{

}

ConvNetConfig::~ConvNetConfig()
{

}

size_t ConvNetConfig::LayerCount() const
{
	return forwardBackConfigs.size();
}
;

bool ConvNetConfig::HasOutputLayer() const
{
	if (outputConfig.get())
		return true;
	else
		return false;
}

void ConvNetConfig::SetOutputConfig(unique_ptr<OutputLayerConfig> config)
{
	outputConfig = move(config);
}

void ConvNetConfig::RemoveOutputConfig()
{
	outputConfig.reset(nullptr);
}

void ConvNetConfig::AddToBack(unique_ptr<ForwardBackPropLayerConfig> config)
{
	forwardBackConfigs.push_back(move(config));
}

void ConvNetConfig::AddToFront(unique_ptr<ForwardBackPropLayerConfig> config)
{
	forwardBackConfigs.insert(forwardBackConfigs.begin(), move(config));
}

void ConvNetConfig::InsertAt(unique_ptr<ForwardBackPropLayerConfig> config,
		size_t index)
{
	forwardBackConfigs.insert(forwardBackConfigs.begin() + index, move(config));
}

void ConvNetConfig::RemoveAt(size_t index)
{
	forwardBackConfigs.erase(forwardBackConfigs.begin() + index);
}

vector<LayerDataDescription> ConvNetConfig::InputDataDescription() const
{
	return inputDataDescriptions;
}

void ConvNetConfig::Accept(ILayerConfigVisitor* visitor)
{
	visitor->Visit(this);

	for (auto& config : forwardBackConfigs)
		config->Accept(visitor);

	if (outputConfig.get())
		outputConfig->Accept(visitor);
}

} /* namespace MachineLearning */
} /* namespace Matuna */
