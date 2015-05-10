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

PerceptronLayerConfig::PerceptronLayerConfig(int units,
		ATMLActivationFunction activationFunction,
		ATMLConnectionType connectionType, bool useRelaxedMath,
		ATMLComputationPrecision computationPrecision) :
		ForwardBackPropLayerConfig(useRelaxedMath, computationPrecision), units(
				units), activationFunction(activationFunction), connectionType(
				connectionType)

{

}

PerceptronLayerConfig::~PerceptronLayerConfig()
{

}

int PerceptronLayerConfig::Units() const
{
	return units;
}

ATMLActivationFunction PerceptronLayerConfig::ActivationFunction() const
{
	return activationFunction;
}

ATMLConnectionType PerceptronLayerConfig::ConnectionType() const
{
	return connectionType;
}

void PerceptronLayerConfig::Accept(ILayerConfigVisitor* visitor)
{
	visitor->Visit(this);
}

} /* namespace MachineLearning */
} /* namespace ATML */
