/*
 * ILayerConfigVisitor.h
 *
 *  Created on: May 3, 2015
 *      Author: Mikael
 */

#ifndef MATUNA_CONVNET_ILAYERCONFIGVISITOR_H_
#define MATUNA_CONVNET_ILAYERCONFIGVISITOR_H_

namespace Matuna
{
namespace MachineLearning
{

//In order to avoid circular dependencies
class ConvNetConfig;
class PerceptronLayerConfig;
class ConvolutionLayerConfig;
class StandardOutputLayerConfig;
class VanillaSamplingLayerConfig;

class ILayerConfigVisitor
{

public:

	virtual ~ILayerConfigVisitor()
	{
	}
	;

	virtual void Visit(const ConvNetConfig* const cnnConfig) = 0;
	virtual void Visit(const PerceptronLayerConfig* const perceptronConfig) = 0;
	virtual void Visit(const ConvolutionLayerConfig* const convolutionConfig) = 0;
	virtual void Visit(const StandardOutputLayerConfig* const convolutionConfig) = 0;
	virtual void Visit(const VanillaSamplingLayerConfig* const vanillaConfig) = 0;
};

} /* Matuna */
} /* MachineLearning */

#endif /* MATUNA_CONVNET_ILAYERCONFIGVISITOR_H_ */
