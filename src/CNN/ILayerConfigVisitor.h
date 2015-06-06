/*
 * ILayerConfigVisitor.h
 *
 *  Created on: May 3, 2015
 *      Author: Mikael
 */

#ifndef MATUNA_CNN_ILAYERCONFIGVISITOR_H_
#define MATUNA_CNN_ILAYERCONFIGVISITOR_H_

namespace Matuna
{
namespace MachineLearning
{

class CNNConfig;
class PerceptronLayerConfig;
class ConvolutionLayerConfig;
class StandardOutputLayerConfig;

class ILayerConfigVisitor
{

public:

	virtual ~ILayerConfigVisitor()
	{
	}
	;

	virtual void Visit(const CNNConfig* const cnnConfig) = 0;
	virtual void Visit(const PerceptronLayerConfig* const perceptronConfig) = 0;
	virtual void Visit(const ConvolutionLayerConfig* const convolutionConfig) = 0;
	virtual void Visit(const StandardOutputLayerConfig* const convolutionConfig) = 0;
};

} /* Matuna */
} /* MachineLearning */

#endif /* MATUNA_CNN_ILAYERCONFIGVISITOR_H_ */
