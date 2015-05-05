/*
 * StandardOutputLayerConfig.h
 *
 *  Created on: May 5, 2015
 *      Author: Mikael
 */

#ifndef ATML_CNN_STANDARDOUTPUTLAYERCONFIG_H_
#define ATML_CNN_STANDARDOUTPUTLAYERCONFIG_H_

#include "OutputLayerConfig.h"

namespace ATML
{
namespace MachineLearning
{

class StandardOutputLayerConfig: public OutputLayerConfig
{
public:
	StandardOutputLayerConfig();
	~StandardOutputLayerConfig();

	virtual void Accept(ILayerConfigVisitor* visitor) override;
};

} /* namespace MachineLearning */
} /* namespace ATML */

#endif /* ATML_CNN_STANDARDOUTPUTLAYERCONFIG_H_ */
