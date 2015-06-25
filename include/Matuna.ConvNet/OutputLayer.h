/*
 * OutputLayer.h
 *
 *  Created on: May 2, 2015
 *      Author: Mikael
 */

#ifndef MATUNA_MATUNA_CONVNET_OUTPUTLAYER_H_
#define MATUNA_MATUNA_CONVNET_OUTPUTLAYER_H_

#include "BackPropLayer.h"
#include "LayerDescriptions.h"
#include "OutputLayerConfig.h"

namespace Matuna
{
namespace MachineLearning
{

class OutputLayer: public BackPropLayer
{
public:
	OutputLayer(const vector<LayerDataDescription>& inputLayerDescriptions,
			MatunaActivationFunction backPropActivation,
			const OutputLayerConfig* outputLayerConfig);
	virtual ~OutputLayer();
};

} /* namespace MachineLearning */
} /* namespace Matuna */

#endif /* MATUNA_MATUNA_CONVNET_OUTPUTLAYER_H_ */
