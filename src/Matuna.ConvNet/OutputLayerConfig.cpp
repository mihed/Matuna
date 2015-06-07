/*
 * OutputLayerConfig.cpp
 *
 *  Created on: May 5, 2015
 *      Author: Mikael
 */

#include "OutputLayerConfig.h"

namespace Matuna
{
namespace MachineLearning
{

OutputLayerConfig::OutputLayerConfig(bool useRelaxedMath,
		MatunaComputationPrecision computationPrecision) :
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
MatunaComputationPrecision OutputLayerConfig::ComputationPrecision() const
{
	return computationPrecision;
}

} /* namespace MachineLearning */
} /* namespace Matuna */
