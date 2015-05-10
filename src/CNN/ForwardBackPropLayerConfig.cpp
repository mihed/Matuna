/*
 * ForwardBackPropLayerConfig.cpp
 *
 *  Created on: May 5, 2015
 *      Author: Mikael
 */

#include "ForwardBackPropLayerConfig.h"

namespace ATML
{
namespace MachineLearning
{

ForwardBackPropLayerConfig::ForwardBackPropLayerConfig(bool useRelaxedMath,
		ATMLComputationPrecision computationPrecision) :
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
ATMLComputationPrecision ForwardBackPropLayerConfig::ComputationPrecision() const
{
	return computationPrecision;
}

} /* namespace MachineLearning */
} /* namespace ATML */
