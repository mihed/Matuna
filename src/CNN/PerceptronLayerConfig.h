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
	int units;

public:
	PerceptronLayerConfig(int units, ATMLActivationFunction activationFunction =
			ATMLSigmoidActivation, ATMLConnectionType connectionType =
			ATMLFullConnection, bool useRelaxedMath = false,
			ATMLComputationPrecision computationPrecision = ATMLNormalPrecision);
	~PerceptronLayerConfig();

	int Units() const;
	ATMLActivationFunction ActivationFunction() const;
	ATMLConnectionType ConnectionType() const;
	virtual void Accept(ILayerConfigVisitor* visitor) override;
};

} /* namespace MachineLearning */
} /* namespace ATML */

#endif /* ATML_CNN_PERCEPTRONLAYERCONFIG_H_ */
