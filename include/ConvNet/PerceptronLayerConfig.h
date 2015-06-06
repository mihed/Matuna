/*
 * PerceptronLayerConfig.h
 *
 *  Created on: May 5, 2015
 *      Author: Mikael
 */

#ifndef MATUNA_CONVNET_PERCEPTRONLAYERCONFIG_H_
#define MATUNA_CONVNET_PERCEPTRONLAYERCONFIG_H_

#include "ForwardBackPropLayerConfig.h"
#include "MatunaActivationFunctionEnum.h"
#include "MatunaConnectionTypeEnum.h"

namespace Matuna
{
namespace MachineLearning
{

class PerceptronLayerConfig: public ForwardBackPropLayerConfig
{
private:
	MatunaActivationFunction activationFunction;
	MatunaConnectionType connectionType;
	int units;

public:
	PerceptronLayerConfig(int units, MatunaActivationFunction activationFunction =
			MatunaSigmoidActivation, MatunaConnectionType connectionType =
			MatunaFullConnection, bool useRelaxedMath = false,
			MatunaComputationPrecision computationPrecision = MatunaNormalPrecision);
	~PerceptronLayerConfig();

	int Units() const;
	MatunaActivationFunction ActivationFunction() const;
	MatunaConnectionType ConnectionType() const;
	virtual void Accept(ILayerConfigVisitor* visitor) override;
};

} /* namespace MachineLearning */
} /* namespace Matuna */

#endif /* MATUNA_CONVNET_PERCEPTRONLAYERCONFIG_H_ */
