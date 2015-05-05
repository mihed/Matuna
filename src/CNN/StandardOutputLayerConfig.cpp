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

StandardOutputLayerConfig::StandardOutputLayerConfig()
{
	// TODO Auto-generated constructor stub

}

StandardOutputLayerConfig::~StandardOutputLayerConfig()
{
	// TODO Auto-generated destructor stub
}

void StandardOutputLayerConfig::Accept(ILayerConfigVisitor* visitor)
{
	visitor->Visit(this);
}

} /* namespace MachineLearning */
} /* namespace ATML */
