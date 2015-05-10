/*
 * OutputLayerConfig.h
 *
 *  Created on: May 5, 2015
 *      Author: Mikael
 */

#ifndef ATML_CNN_OUTPUTLAYERCONFIG_H_
#define ATML_CNN_OUTPUTLAYERCONFIG_H_

#include "ILayerConfig.h"
#include "ATMLComputationPrecision.h"

namespace ATML
{
namespace MachineLearning
{

class OutputLayerConfig: public ILayerConfig
{
private:
	bool useRelaxedMath;
	ATMLComputationPrecision computationPrecision;

public:
	OutputLayerConfig(bool useRelaxedMath = false,
			ATMLComputationPrecision computationPrecision = ATMLNormalPrecision);
	virtual ~OutputLayerConfig();

	bool UseRelaxedMath() const;
	ATMLComputationPrecision ComputationPrecision() const;

};

} /* namespace MachineLearning */
} /* namespace ATML */

#endif /* ATML_CNN_OUTPUTLAYERCONFIG_H_ */
