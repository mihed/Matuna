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

namespace ATML
{
namespace MachineLearning
{

class OutputLayer: public BackPropLayer
{
public:
	OutputLayer(const LayerDataDescription& inputLayerDescription);
	virtual ~OutputLayer();
};

} /* namespace MachineLearning */
} /* namespace ATML */

#endif /* ATML_CNN_OUTPUTLAYER_H_ */
