/*
* VanillaSamplingLayerConfig.cpp
*
*  Created on: Jun 21, 2015
*      Author: Mikael
*/

#include "VanillaSamplingLayerConfig.h"

namespace Matuna
{
	namespace MachineLearning
	{

		VanillaSamplingLayerConfig::VanillaSamplingLayerConfig(int samplingSizeWidth, int samplingSizeHeight)
		{
			this->samplingSizeHeight = samplingSizeHeight;
			this->samplingSizeWidth = samplingSizeWidth;
		}

		VanillaSamplingLayerConfig::~VanillaSamplingLayerConfig()
		{

		}

		int VanillaSamplingLayerConfig::SamplingSizeWidth() const
		{
			return samplingSizeWidth;
		}

		int VanillaSamplingLayerConfig::SamplingSizeHeight() const
		{
			return samplingSizeHeight;
		}

		void VanillaSamplingLayerConfig::Accept(ILayerConfigVisitor* visitor)
		{
			visitor->Visit(this);
		}

	} /* namespace MachineLearning */
} /* namespace Matuna */
