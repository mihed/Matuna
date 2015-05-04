/*
 * ILayerConfig.h
 *
 *  Created on: May 3, 2015
 *      Author: Mikael
 */

#ifndef ATML_CNN_ILAYERCONFIG_H_
#define ATML_CNN_ILAYERCONFIG_H_

namespace ATML
{
namespace MachineLearning
{

class ILayerConfigVisitor;

class ILayerConfig
{

public:
	virtual ~ILayerConfig()
	{
	}
	;
	virtual void Accept(ILayerConfigVisitor* visitor) = 0;

};

} /* ATML */
} /* MachineLearning */

#endif /* ATML_CNN_ILAYERCONFIG_H_ */
