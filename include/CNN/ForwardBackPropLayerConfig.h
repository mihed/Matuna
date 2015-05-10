/*
 * ForwardBackPropLayerConfig.h
 *
 *  Created on: May 5, 2015
 *      Author: Mikael
 */

#ifndef ATML_CNN_FORWARDBACKPROPLAYERCONFIG_H_
#define ATML_CNN_FORWARDBACKPROPLAYERCONFIG_H_

#include "ILayerConfig.h"
#include "ATMLComputationPrecision.h"

namespace ATML
{
namespace MachineLearning
{

class ForwardBackPropLayerConfig: public ILayerConfig
{
private:
	bool useRelaxedMath;
	ATMLComputationPrecision computationPrecision;

public:
	ForwardBackPropLayerConfig(bool useRelaxedMath = false,
			ATMLComputationPrecision computationPrecision = ATMLNormalPrecision);
	virtual ~ForwardBackPropLayerConfig();

	bool UseRelaxedMath() const;
	ATMLComputationPrecision ComputationPrecision() const;
};

} /* namespace MachineLearning */
} /* namespace ATML */

#endif /* ATML_CNN_FORWARDBACKPROPLAYERCONFIG_H_ */
