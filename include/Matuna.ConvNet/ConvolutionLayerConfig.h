/*
 * ConvolutionLayerConfig.h
 *
 *  Created on: May 5, 2015
 *      Author: Mikael
 */

#ifndef MATUNA_MATUNA_CONVNET_CONVOLUTIONLAYERCONFIG_H_
#define MATUNA_MATUNA_CONVNET_CONVOLUTIONLAYERCONFIG_H_

#include "ForwardBackPropLayerConfig.h"
#include "MatunaConnectionTypeEnum.h"
#include "MatunaActivationFunctionEnum.h"

namespace Matuna
{
namespace MachineLearning
{

class ConvolutionLayerConfig: public ForwardBackPropLayerConfig
{
private:
	MatunaActivationFunction activationFunction;
	MatunaConnectionType connectionType;
	int filterCount;
	int filterWidth;
	int filterHeight;

public:
	ConvolutionLayerConfig(int filterCount, int filterWidth, int filterHeight,
			MatunaActivationFunction activationFunction = MatunaSigmoidActivation,
			MatunaConnectionType connectionType = MatunaFullConnection,
			bool useRelaxedMath = false,
			MatunaComputationPrecision computationPrecision = MatunaNormalPrecision);
	virtual ~ConvolutionLayerConfig();

	virtual void Accept(ILayerConfigVisitor* visitor) override;

	int FilterCount() const;
	int FilterHeight() const;
	int FilterWidth() const;
	MatunaActivationFunction ActivationFunction() const;
	MatunaConnectionType ConnectionType() const;
};

} /* namespace MachineLearning */
} /* namespace Matuna */

#endif /* MATUNA_MATUNA_CONVNET_CONVOLUTIONLAYERCONFIG_H_ */
