/*
* MaxPoolingLayerConfig.h
*
*  Created on: Jun 23, 2015
*      Author: Mikael
*/

#ifndef MATUNA_MATUNA_CONVNET_MAXPOOLINGLAYERCONFIG_H_
#define MATUNA_MATUNA_CONVNET_MAXPOOLINGLAYERCONFIG_H_

#include "ForwardBackPropLayerConfig.h"

namespace Matuna
{
	namespace MachineLearning
	{

		class MaxPoolingLayerConfig: public ForwardBackPropLayerConfig
		{
		private:
			int samplingSizeWidth;
			int samplingSizeHeight;

		public:
			MaxPoolingLayerConfig(int samplingSizeWidth, int samplingSizeHeight);
			~MaxPoolingLayerConfig();


			int SamplingSizeWidth() const;
			int SamplingSizeHeight() const;
			virtual void Accept(ILayerConfigVisitor* visitor) override;
		};

	} /* namespace MachineLearning */
} /* namespace Matuna */

#endif /* MATUNA_MATUNA_CONVNET_MAXPOOLINGLAYERCONFIG_H_ */
