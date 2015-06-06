/*
 * StandardOutputLayerConfig.h
 *
 *  Created on: May 5, 2015
 *      Author: Mikael
 */

#ifndef MATUNA_CNN_STANDARDOUTPUTLAYERCONFIG_H_
#define MATUNA_CNN_STANDARDOUTPUTLAYERCONFIG_H_

#include "OutputLayerConfig.h"
#include "MatunaErrorFunctionEnum.h"

namespace Matuna
{
namespace MachineLearning
{

class StandardOutputLayerConfig: public OutputLayerConfig
{

private:
	MatunaErrorFunction errorFunction;

public:
	StandardOutputLayerConfig(MatunaErrorFunction errorFunction =
			MatunaMeanSquareError, bool useRelaxedMath = false,
			MatunaComputationPrecision computationPrecision = MatunaNormalPrecision);
	~StandardOutputLayerConfig();

	MatunaErrorFunction ErrorFunction() const;
	virtual void Accept(ILayerConfigVisitor* visitor) override;
};

} /* namespace MachineLearning */
} /* namespace Matuna */

#endif /* MATUNA_CNN_STANDARDOUTPUTLAYERCONFIG_H_ */
