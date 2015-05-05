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

PerceptronLayerConfig::PerceptronLayerConfig()
{
	// TODO Auto-generated constructor stub

}

PerceptronLayerConfig::~PerceptronLayerConfig()
{
	// TODO Auto-generated destructor stub
}

void PerceptronLayerConfig::Accept(ILayerConfigVisitor* visitor)
{
	visitor->Visit(this);
}

} /* namespace MachineLearning */
} /* namespace ATML */
