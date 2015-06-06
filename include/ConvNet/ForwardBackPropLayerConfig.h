/*
 * ForwardBackPropLayerConfig.h
 *
 *  Created on: May 5, 2015
 *      Author: Mikael
 */

#ifndef MATUNA_CONVNET_FORWARDBACKPROPLAYERCONFIG_H_
#define MATUNA_CONVNET_FORWARDBACKPROPLAYERCONFIG_H_

#include "ILayerConfig.h"
#include "MatunaComputationPrecision.h"

namespace Matuna
{
namespace MachineLearning
{

class ForwardBackPropLayerConfig: public ILayerConfig
{
private:
	bool useRelaxedMath;
	MatunaComputationPrecision computationPrecision;

public:
	ForwardBackPropLayerConfig(bool useRelaxedMath = false,
			MatunaComputationPrecision computationPrecision = MatunaNormalPrecision);
	virtual ~ForwardBackPropLayerConfig();

	bool UseRelaxedMath() const;
	MatunaComputationPrecision ComputationPrecision() const;
};

} /* namespace MachineLearning */
} /* namespace Matuna */

#endif /* MATUNA_CONVNET_FORWARDBACKPROPLAYERCONFIG_H_ */
