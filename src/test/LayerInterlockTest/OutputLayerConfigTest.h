/*
 * OutputLayerConfig.h
 *
 *  Created on: May 3, 2015
 *      Author: Mikael
 */

#ifndef ATML_TEST_LAYERINTERLOCKTEST_OUTPUTLAYERCONFIG_H_
#define ATML_TEST_LAYERINTERLOCKTEST_OUTPUTLAYERCONFIG_H_

#include "ILayerConfigTest.h"

using namespace ATML::MachineLearning;

class FactoryVisitorTest;

class OutputLayerConfigTest: public ILayerConfigTest
{
public:
	OutputLayerConfigTest();
	~OutputLayerConfigTest();

	virtual void Accept(FactoryVisitorTest* visitor) override;
};

#endif /* ATML_TEST_LAYERINTERLOCKTEST_OUTPUTLAYERCONFIG_H_ */
