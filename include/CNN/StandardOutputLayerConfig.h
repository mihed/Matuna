/*
 * StandardOutputLayerConfig.h
 *
 *  Created on: May 5, 2015
 *      Author: Mikael
 */

#ifndef ATML_CNN_STANDARDOUTPUTLAYERCONFIG_H_
#define ATML_CNN_STANDARDOUTPUTLAYERCONFIG_H_

#include "OutputLayerConfig.h"
#include "ATMLErrorFunctionEnum.h"

namespace ATML
{
namespace MachineLearning
{

class StandardOutputLayerConfig: public OutputLayerConfig
{

private:
	ATMLErrorFunction errorFunction;

public:
	StandardOutputLayerConfig(ATMLErrorFunction errorFunction =
			ATMLMeanSquareError);
	~StandardOutputLayerConfig();

	ATMLErrorFunction ErrorFunction() const
	{
		return errorFunction;
	}
	;

	virtual void Accept(ILayerConfigVisitor* visitor) override;
};

} /* namespace MachineLearning */
} /* namespace ATML */

#endif /* ATML_CNN_STANDARDOUTPUTLAYERCONFIG_H_ */
