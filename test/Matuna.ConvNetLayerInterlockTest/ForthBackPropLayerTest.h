/*
 * ForthBackPropLayerTest.h
 *
 *  Created on: May 3, 2015
 *      Author: Mikael
 */

#ifndef MATUNA_TEST_LAYERINTERLOCKTEST_FORTHBACKPROPLAYERTEST_H_
#define MATUNA_TEST_LAYERINTERLOCKTEST_FORTHBACKPROPLAYERTEST_H_

#include "Matuna.ConvNet/ForwardBackPropLayer.h"

using namespace Matuna::MachineLearning;

class ForthBackPropLayerTest: public ForwardBackPropLayer
{
public:
	ForthBackPropLayerTest(const vector<LayerDataDescription>& inputLayerDescriptions,
			MatunaActivationFunction backPropActivation,
			const ForwardBackPropLayerConfig* config);
	~ForthBackPropLayerTest();

	virtual void InterlockFinalized() override;
};

#endif /* MATUNA_TEST_LAYERINTERLOCKTEST_FORTHBACKPROPLAYERTEST_H_ */
