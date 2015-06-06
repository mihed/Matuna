/*
 * OutputLayerConfig.h
 *
 *  Created on: May 5, 2015
 *      Author: Mikael
 */

#ifndef MATUNA_CNN_OUTPUTLAYERCONFIG_H_
#define MATUNA_CNN_OUTPUTLAYERCONFIG_H_

#include "ILayerConfig.h"
#include "MatunaComputationPrecision.h"

namespace Matuna
{
namespace MachineLearning
{

class OutputLayerConfig: public ILayerConfig
{
private:
	bool useRelaxedMath;
	MatunaComputationPrecision computationPrecision;

public:
	OutputLayerConfig(bool useRelaxedMath = false,
			MatunaComputationPrecision computationPrecision = MatunaNormalPrecision);
	virtual ~OutputLayerConfig();

	bool UseRelaxedMath() const;
	MatunaComputationPrecision ComputationPrecision() const;

};

} /* namespace MachineLearning */
} /* namespace Matuna */

#endif /* MATUNA_CNN_OUTPUTLAYERCONFIG_H_ */
