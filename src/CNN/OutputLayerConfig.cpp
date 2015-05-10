/*
 * OutputLayerConfig.cpp
 *
 *  Created on: May 5, 2015
 *      Author: Mikael
 */

#include "OutputLayerConfig.h"

namespace ATML
{
namespace MachineLearning
{

OutputLayerConfig::OutputLayerConfig(bool useRelaxedMath,
		ATMLComputationPrecision computationPrecision) :
		useRelaxedMath(useRelaxedMath), computationPrecision(
				computationPrecision)
{

}

OutputLayerConfig::~OutputLayerConfig()
{

}

bool OutputLayerConfig::UseRelaxedMath() const
{
	return useRelaxedMath;
}
ATMLComputationPrecision OutputLayerConfig::ComputationPrecision() const
{
	return computationPrecision;
}

} /* namespace MachineLearning */
} /* namespace ATML */
