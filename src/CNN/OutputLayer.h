/*
 * OutputLayer.h
 *
 *  Created on: May 2, 2015
 *      Author: Mikael
 */

#ifndef ATML_CNN_OUTPUTLAYER_H_
#define ATML_CNN_OUTPUTLAYER_H_

#include "BackPropLayer.h"
#include "LayerDescriptions.h"
#include "OutputLayerConfig.h"

namespace ATML
{
namespace MachineLearning
{

class OutputLayer: public BackPropLayer
{
public:
	OutputLayer(const vector<LayerDataDescription>& inputLayerDescriptions,
			ATMLActivationFunction backPropActivation,
			const OutputLayerConfig* outputLayerConfig);
	virtual ~OutputLayer();
};

} /* namespace MachineLearning */
} /* namespace ATML */

#endif /* ATML_CNN_OUTPUTLAYER_H_ */
