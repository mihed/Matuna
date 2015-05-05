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

ConvolutionLayerConfig::ConvolutionLayerConfig()
{

}

ConvolutionLayerConfig::~ConvolutionLayerConfig()
{

}

void ConvolutionLayerConfig::Accept(ILayerConfigVisitor* visitor)
{
	visitor->Visit(this);
}

} /* namespace MachineLearning */
} /* namespace ATML */
