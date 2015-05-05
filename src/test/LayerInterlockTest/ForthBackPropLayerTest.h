/*
 * ForthBackPropLayerTest.h
 *
 *  Created on: May 3, 2015
 *      Author: Mikael
 */

#ifndef ATML_TEST_LAYERINTERLOCKTEST_FORTHBACKPROPLAYERTEST_H_
#define ATML_TEST_LAYERINTERLOCKTEST_FORTHBACKPROPLAYERTEST_H_

#include "CNN/ForwardBackPropLayer.h"

using namespace ATML::MachineLearning;

class ForthBackPropLayerTest: public ForwardBackPropLayer
{
public:
	ForthBackPropLayerTest(const LayerDataDescription& inputLayerDescription,
			const ForwardBackPropLayerConfig* config);
	~ForthBackPropLayerTest();
};

#endif /* ATML_TEST_LAYERINTERLOCKTEST_FORTHBACKPROPLAYERTEST_H_ */
