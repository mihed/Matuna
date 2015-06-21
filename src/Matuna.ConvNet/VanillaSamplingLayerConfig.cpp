/*
 * VanillaSamplingLayerConfig.cpp
 *
 *  Created on: Jun 21, 2015
 *      Author: Mikael
 */

#include "VanillaSamplingLayerConfig.h"

namespace Matuna
{
namespace MachineLearning
{

VanillaSamplingLayerConfig::VanillaSamplingLayerConfig(int samplingSize) :
	samplingSize(samplingSize)
{

}

VanillaSamplingLayerConfig::~VanillaSamplingLayerConfig()
{

}

int VanillaSamplingLayerConfig::SamplingSize()
{
	return samplingSize;
}

void VanillaSamplingLayerConfig::Accept(ILayerConfigVisitor* visitor)
{
	visitor->Visit(this);
}

} /* namespace MachineLearning */
} /* namespace Matuna */
