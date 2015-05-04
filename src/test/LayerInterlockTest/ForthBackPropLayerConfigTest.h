/*
 * ForthBackPropLayerConfigTest.h
 *
 *  Created on: May 3, 2015
 *      Author: Mikael
 */

#ifndef ATML_TEST_LAYERINTERLOCKTEST_FORTHBACKPROPLAYERCONFIGTEST_H_
#define ATML_TEST_LAYERINTERLOCKTEST_FORTHBACKPROPLAYERCONFIGTEST_H_

#include "ILayerConfigTest.h"

class ForthBackPropLayerConfigTest: public ILayerConfigTest
{
public:
	ForthBackPropLayerConfigTest();
	~ForthBackPropLayerConfigTest();

	virtual void Accept(FactoryVisitorTest* visitor) override;
};

#endif /* ATML_TEST_LAYERINTERLOCKTEST_FORTHBACKPROPLAYERCONFIGTEST_H_ */
