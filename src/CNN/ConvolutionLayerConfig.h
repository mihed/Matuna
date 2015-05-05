/*
 * ConvolutionLayerConfig.h
 *
 *  Created on: May 5, 2015
 *      Author: Mikael
 */

#ifndef ATML_CNN_CONVOLUTIONLAYERCONFIG_H_
#define ATML_CNN_CONVOLUTIONLAYERCONFIG_H_

#include "ForwardBackPropLayerConfig.h"
#include "ATMLConnectionTypeEnum.h"
#include "ATMLActivationFunctionEnum.h"

namespace ATML
{
namespace MachineLearning
{

class ConvolutionLayerConfig: public ForwardBackPropLayerConfig
{
private:
	ATMLActivationFunction activationFunction;
	ATMLConnectionType connectionType;

public:
	ConvolutionLayerConfig(ATMLActivationFunction activationFunction =
			ATMLSigmoidActivation, ATMLConnectionType connectionType =
			ATMLFullConnection);
	~ConvolutionLayerConfig();

	virtual void Accept(ILayerConfigVisitor* visitor) override;

	ATMLActivationFunction ActivationFunction() const
	{
		return activationFunction;
	}
	;

	ATMLConnectionType ConnectionType() const
	{
		return connectionType;
	}
	;
};

} /* namespace MachineLearning */
} /* namespace ATML */

#endif /* ATML_CNN_CONVOLUTIONLAYERCONFIG_H_ */