/*
 * ConvolutionLayerConfig.cpp
 *
 *  Created on: May 5, 2015
 *      Author: Mikael
 */

#include "ConvolutionLayerConfig.h"

namespace Matuna
{
namespace MachineLearning
{

ConvolutionLayerConfig::ConvolutionLayerConfig(int filterCount, int filterWidth,
		int filterHeight, MatunaActivationFunction activationFunction,
		MatunaConnectionType connectionType, bool useRelaxedMath,
		MatunaComputationPrecision computationPrecision) :
		ForwardBackPropLayerConfig(useRelaxedMath, computationPrecision)
{
	 this->filterCount = filterCount;
	 this->filterWidth = filterWidth;
	 this->filterHeight = filterHeight;
	 this->activationFunction = activationFunction;
	 this->connectionType = connectionType;
}

ConvolutionLayerConfig::~ConvolutionLayerConfig()
{

}

int ConvolutionLayerConfig::FilterCount() const
{
	return filterCount;
}

int ConvolutionLayerConfig::FilterHeight() const
{
	return filterHeight;
}

int ConvolutionLayerConfig::FilterWidth() const
{
	return filterWidth;
}

MatunaActivationFunction ConvolutionLayerConfig::ActivationFunction() const
{
	return activationFunction;
}

MatunaConnectionType ConvolutionLayerConfig::ConnectionType() const
{
	return connectionType;
}

void ConvolutionLayerConfig::Accept(ILayerConfigVisitor* visitor)
{
	visitor->Visit(this);
}

} /* namespace MachineLearning */
} /* namespace Matuna */
