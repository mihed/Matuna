/*
 * VanillaSamplingLayerConfig.h
 *
 *  Created on: Jun 21, 2015
 *      Author: Mikael
 */

#ifndef MATUNA_MATUNA_CONVNET_VANILLASAMPLINGLAYERCONFIG_H_
#define MATUNA_MATUNA_CONVNET_VANILLASAMPLINGLAYERCONFIG_H_

#include "ForwardBackPropLayerConfig.h"

namespace Matuna
{
namespace MachineLearning
{

class VanillaSamplingLayerConfig: public ForwardBackPropLayerConfig
{

private:
	int samplingSizeWidth;
	int samplingSizeHeight;

public:
	VanillaSamplingLayerConfig(int samplingSizeWidth, int samplingSizeHeight);
	~VanillaSamplingLayerConfig();


	int SamplingSizeWidth() const;
	int SamplingSizeHeight() const;
	virtual void Accept(ILayerConfigVisitor* visitor) override;
};

} /* namespace MachineLearning */
} /* namespace Matuna */

#endif /* MATUNA_MATUNA_CONVNET_VANILLASAMPLINGLAYERCONFIG_H_ */
