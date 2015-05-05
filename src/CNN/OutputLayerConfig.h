/*
 * OutputLayerConfig.h
 *
 *  Created on: May 5, 2015
 *      Author: Mikael
 */

#ifndef ATML_CNN_OUTPUTLAYERCONFIG_H_
#define ATML_CNN_OUTPUTLAYERCONFIG_H_

#include "ILayerConfig.h"

namespace ATML
{
namespace MachineLearning
{

class OutputLayerConfig: public ILayerConfig
{
public:
	OutputLayerConfig();
	virtual ~OutputLayerConfig();
};

} /* namespace MachineLearning */
} /* namespace ATML */

#endif /* ATML_CNN_OUTPUTLAYERCONFIG_H_ */
