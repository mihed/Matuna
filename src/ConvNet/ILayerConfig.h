/*
 * ILayerConfig.h
 *
 *  Created on: May 3, 2015
 *      Author: Mikael
 */

#ifndef MATUNA_CONVNET_ILAYERCONFIG_H_
#define MATUNA_CONVNET_ILAYERCONFIG_H_

#include "ILayerConfigVisitor.h"

namespace Matuna
{
namespace MachineLearning
{

class ILayerConfig
{

public:
	virtual ~ILayerConfig()
	{
	}
	;
	virtual void Accept(ILayerConfigVisitor* visitor) = 0;

};

} /* Matuna */
} /* MachineLearning */

#endif /* MATUNA_CONVNET_ILAYERCONFIG_H_ */
