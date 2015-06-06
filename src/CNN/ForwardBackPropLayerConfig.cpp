/*
 * ForwardBackPropLayerConfig.cpp
 *
 *  Created on: May 5, 2015
 *      Author: Mikael
 */

#include "ForwardBackPropLayerConfig.h"

namespace Matuna
{
namespace MachineLearning
{

ForwardBackPropLayerConfig::ForwardBackPropLayerConfig(bool useRelaxedMath,
		MatunaComputationPrecision computationPrecision) :
		useRelaxedMath(useRelaxedMath), computationPrecision(
				computationPrecision)
{

}

ForwardBackPropLayerConfig::~ForwardBackPropLayerConfig()
{

}

bool ForwardBackPropLayerConfig::UseRelaxedMath() const
{
	return useRelaxedMath;
}
MatunaComputationPrecision ForwardBackPropLayerConfig::ComputationPrecision() const
{
	return computationPrecision;
}

} /* namespace MachineLearning */
} /* namespace Matuna */
