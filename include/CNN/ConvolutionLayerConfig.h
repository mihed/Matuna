/*
 * ConvolutionLayerConfig.h
 *
 *  Created on: May 5, 2015
 *      Author: Mikael
 */

#ifndef ATML_CNN_CONVOLUTIONLAYERCONFIG_H_
#define ATML_CNN_CONVOLUTIONLAYERCONFIG_H_

#include "ForwardBackPropLayerConfig.h"

namespace ATML
{
namespace MachineLearning
{

class ConvolutionLayerConfig: public ForwardBackPropLayerConfig
{
public:
	ConvolutionLayerConfig();
	~ConvolutionLayerConfig();

	virtual void Accept(ILayerConfigVisitor* visitor) override;
};

} /* namespace MachineLearning */
} /* namespace ATML */

#endif /* ATML_CNN_CONVOLUTIONLAYERCONFIG_H_ */
