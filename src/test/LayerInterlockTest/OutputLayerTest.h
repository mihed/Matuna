/*
 * OutputLayerTest.h
 *
 *  Created on: May 3, 2015
 *      Author: Mikael
 */

#ifndef ATML_TEST_LAYERINTERLOCKTEST_OUTPUTLAYERTEST_H_
#define ATML_TEST_LAYERINTERLOCKTEST_OUTPUTLAYERTEST_H_

#include "CNN/OutputLayer.h"

using namespace ATML::MachineLearning;

class OutputLayerTest: public OutputLayer
{
public:
	OutputLayerTest(const vector<LayerDataDescription>& inputLayerDescriptions,
		ATMLActivationFunction backPropActivation,
			const OutputLayerConfig* config);
	~OutputLayerTest();

	virtual void InterlockFinalized() override;
};

#endif /* ATML_TEST_LAYERINTERLOCKTEST_OUTPUTLAYERTEST_H_ */
