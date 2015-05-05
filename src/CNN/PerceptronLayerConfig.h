/*
 * PerceptronLayerConfig.h
 *
 *  Created on: May 5, 2015
 *      Author: Mikael
 */

#ifndef ATML_CNN_PERCEPTRONLAYERCONFIG_H_
#define ATML_CNN_PERCEPTRONLAYERCONFIG_H_

#include "ForwardBackPropLayerConfig.h"

namespace ATML
{
namespace MachineLearning
{

class PerceptronLayerConfig: public ForwardBackPropLayerConfig
{
public:
	PerceptronLayerConfig();
	~PerceptronLayerConfig();

	virtual void Accept(ILayerConfigVisitor* visitor) override;
};

} /* namespace MachineLearning */
} /* namespace ATML */

#endif /* ATML_CNN_PERCEPTRONLAYERCONFIG_H_ */
