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
	int filterCount;
	int filterWidth;
	int filterHeight;

public:
	ConvolutionLayerConfig(int filterCount, int filterWidth, int filterHeight,
			ATMLActivationFunction activationFunction = ATMLSigmoidActivation,
			ATMLConnectionType connectionType = ATMLFullConnection,
			bool useRelaxedMath = false,
			ATMLComputationPrecision computationPrecision = ATMLNormalPrecision);
	virtual ~ConvolutionLayerConfig();

	virtual void Accept(ILayerConfigVisitor* visitor) override;

	int FilterCount() const;
	int FilterHeight() const;
	int FilterWidth() const;
	ATMLActivationFunction ActivationFunction() const;
	ATMLConnectionType ConnectionType() const;
};

} /* namespace MachineLearning */
} /* namespace ATML */

#endif /* ATML_CNN_CONVOLUTIONLAYERCONFIG_H_ */
