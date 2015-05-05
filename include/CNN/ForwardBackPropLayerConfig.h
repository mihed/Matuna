/*
 * ForwardBackPropLayerConfig.h
 *
 *  Created on: May 5, 2015
 *      Author: Mikael
 */

#ifndef ATML_CNN_FORWARDBACKPROPLAYERCONFIG_H_
#define ATML_CNN_FORWARDBACKPROPLAYERCONFIG_H_

#include "ILayerConfig.h"

namespace ATML
{
namespace MachineLearning
{

class ForwardBackPropLayerConfig: public ILayerConfig
{
public:
	ForwardBackPropLayerConfig();
	virtual ~ForwardBackPropLayerConfig();
};

} /* namespace MachineLearning */
} /* namespace ATML */

#endif /* ATML_CNN_FORWARDBACKPROPLAYERCONFIG_H_ */
