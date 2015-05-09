/*
 * ConvolutionLayerConfig.cpp
 *
 *  Created on: May 5, 2015
 *      Author: Mikael
 */

#include "ConvolutionLayerConfig.h"

namespace ATML
{
	namespace MachineLearning
	{

		ConvolutionLayerConfig::ConvolutionLayerConfig(int filterCount,
			int filterWidth,
			int filterHeight,
			ATMLActivationFunction activationFunction,
			ATMLConnectionType connectionType) :
			filterCount(filterCount), filterWidth(filterWidth), filterHeight(filterHeight),
			activationFunction(activationFunction), connectionType(connectionType)
		{

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

		ATMLActivationFunction ConvolutionLayerConfig::ActivationFunction() const
		{
			return activationFunction;
		}

		ATMLConnectionType ConvolutionLayerConfig::ConnectionType() const
		{
			return connectionType;
		}

		void ConvolutionLayerConfig::Accept(ILayerConfigVisitor* visitor)
		{
			visitor->Visit(this);
		}

	} /* namespace MachineLearning */
} /* namespace ATML */
