/*
 * StandardOutputLayerConfig.cpp
 *
 *  Created on: May 5, 2015
 *      Author: Mikael
 */

#include "StandardOutputLayerConfig.h"

namespace ATML
{
namespace MachineLearning
{

StandardOutputLayerConfig::StandardOutputLayerConfig(
		ATMLErrorFunction errorFunction) :
		errorFunction(errorFunction)
{

}

StandardOutputLayerConfig::~StandardOutputLayerConfig()
{

}

void StandardOutputLayerConfig::Accept(ILayerConfigVisitor* visitor)
{
	visitor->Visit(this);
}

} /* namespace MachineLearning */
} /* namespace ATML */