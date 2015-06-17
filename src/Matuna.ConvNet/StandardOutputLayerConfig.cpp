/*
 * StandardOutputLayerConfig.cpp
 *
 *  Created on: May 5, 2015
 *      Author: Mikael
 */

#include "StandardOutputLayerConfig.h"

namespace Matuna
{
namespace MachineLearning
{

StandardOutputLayerConfig::StandardOutputLayerConfig(
		MatunaErrorFunction errorFunction, bool useRelaxedMath,
		MatunaComputationPrecision computationPrecision) :
		OutputLayerConfig(useRelaxedMath, computationPrecision)
{
	this->errorFunction = errorFunction;
}

StandardOutputLayerConfig::~StandardOutputLayerConfig()
{

}

MatunaErrorFunction StandardOutputLayerConfig::ErrorFunction() const
{
	return errorFunction;
}

void StandardOutputLayerConfig::Accept(ILayerConfigVisitor* visitor)
{
	visitor->Visit(this);
}

} /* namespace MachineLearning */
} /* namespace Matuna */
