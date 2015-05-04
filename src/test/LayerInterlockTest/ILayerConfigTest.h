/*
 * ILayerConfigTest.h
 *
 *  Created on: May 3, 2015
 *      Author: Mikael
 */

#ifndef ATML_TEST_LAYERINTERLOCKTEST_ILAYERCONFIGTEST_H_
#define ATML_TEST_LAYERINTERLOCKTEST_ILAYERCONFIGTEST_H_

#include "CNN/ILayerConfig.h"

using namespace ATML::MachineLearning;

class FactoryVisitorTest;

class ILayerConfigTest: public ILayerConfig
{
public:
	ILayerConfigTest();
	~ILayerConfigTest();

	virtual void Accept(FactoryVisitorTest* visitor) = 0;
	virtual void Accept(ILayerConfigVisitor* visitor) final override;
};

#endif /* ATML_TEST_LAYERINTERLOCKTEST_ILAYERCONFIGTEST_H_ */
