/*
 * PerceptronLayerConfig.h
 *
 *  Created on: May 5, 2015
 *      Author: Mikael
 */

#ifndef ATML_CNN_PERCEPTRONLAYERCONFIG_H_
#define ATML_CNN_PERCEPTRONLAYERCONFIG_H_

#include "ForwardBackPropLayerConfig.h"
#include "ATMLActivationFunctionEnum.h"
#include "ATMLConnectionTypeEnum.h"

namespace ATML
{
namespace MachineLearning
{

class PerceptronLayerConfig: public ForwardBackPropLayerConfig
{
private:
	ATMLActivationFunction activationFunction;
	ATMLConnectionType connectionType;

public:
	PerceptronLayerConfig(ATMLActivationFunction activationFunction =
			ATMLSigmoidActivation, ATMLConnectionType connectionType =
			ATMLFullConnection);
	~PerceptronLayerConfig();

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

	virtual void Accept(ILayerConfigVisitor* visitor) override;
};

} /* namespace MachineLearning */
} /* namespace ATML */

#endif /* ATML_CNN_PERCEPTRONLAYERCONFIG_H_ */
