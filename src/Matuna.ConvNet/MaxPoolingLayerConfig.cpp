/*
* MaxPoolingLayerConfig.cpp
*
*  Created on: Jun 23, 2015
*      Author: Mikael
*/

#include "MaxPoolingLayerConfig.h"

namespace Matuna
{
	namespace MachineLearning
	{

		MaxPoolingLayerConfig::MaxPoolingLayerConfig(int samplingSizeWidth, int samplingSizeHeight)
		{
			this->samplingSizeWidth = samplingSizeWidth;
			this->samplingSizeHeight = samplingSizeHeight;
		}

		MaxPoolingLayerConfig::~MaxPoolingLayerConfig()
		{

		}

		int MaxPoolingLayerConfig::SamplingSizeWidth() const
		{
			return samplingSizeWidth;
		}
		int MaxPoolingLayerConfig::SamplingSizeHeight() const
		{
			return samplingSizeHeight;
		}

		void MaxPoolingLayerConfig::Accept(ILayerConfigVisitor* visitor)
		{
			visitor->Visit(this);
		}

	} /* namespace MachineLearning */
} /* namespace Matuna */
