/*
 * PerceptronLayerConfig.cpp
 *
 *  Created on: May 5, 2015
 *      Author: Mikael
 */

#include "PerceptronLayerConfig.h"

namespace Matuna
{
namespace MachineLearning
{

PerceptronLayerConfig::PerceptronLayerConfig(int units,
		MatunaActivationFunction activationFunction,
		MatunaConnectionType connectionType, bool useRelaxedMath,
		MatunaComputationPrecision computationPrecision) :
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

MatunaActivationFunction PerceptronLayerConfig::ActivationFunction() const
{
	return activationFunction;
}

MatunaConnectionType PerceptronLayerConfig::ConnectionType() const
{
	return connectionType;
}

void PerceptronLayerConfig::Accept(ILayerConfigVisitor* visitor)
{
	visitor->Visit(this);
}

} /* namespace MachineLearning */
} /* namespace Matuna */
