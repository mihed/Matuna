/*
 * OutputLayerTest.h
 *
 *  Created on: May 3, 2015
 *      Author: Mikael
 */

#ifndef MATUNA_TEST_LAYERINTERLOCKTEST_OUTPUTLAYERTEST_H_
#define MATUNA_TEST_LAYERINTERLOCKTEST_OUTPUTLAYERTEST_H_

#include "Matuna.ConvNet/OutputLayer.h"

using namespace Matuna::MachineLearning;

class OutputLayerTest: public OutputLayer
{
public:
	OutputLayerTest(const vector<LayerDataDescription>& inputLayerDescriptions,
		MatunaActivationFunction backPropActivation,
			const OutputLayerConfig* config);
	~OutputLayerTest();

	virtual void InterlockFinalized() override;
};

#endif /* MATUNA_TEST_LAYERINTERLOCKTEST_OUTPUTLAYERTEST_H_ */
