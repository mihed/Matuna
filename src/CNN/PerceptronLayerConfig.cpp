/*
 * PerceptronLayerConfig.cpp
 *
 *  Created on: May 5, 2015
 *      Author: Mikael
 */

#include "PerceptronLayerConfig.h"

namespace ATML
{
namespace MachineLearning
{

PerceptronLayerConfig::PerceptronLayerConfig(
		ATMLActivationFunction activationFunction,
		ATMLConnectionType connectionType) :
		activationFunction(activationFunction), connectionType(connectionType)

{

}

PerceptronLayerConfig::~PerceptronLayerConfig()
{

}

void PerceptronLayerConfig::Accept(ILayerConfigVisitor* visitor)
{
	visitor->Visit(this);
}

} /* namespace MachineLearning */
} /* namespace ATML */
