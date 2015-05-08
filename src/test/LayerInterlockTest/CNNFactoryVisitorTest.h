/*
 * CNNFactoryVisitorTest.h
 *
 *  Created on: May 5, 2015
 *      Author: Mikael
 */

#ifndef ATML_TEST_LAYERINTERLOCKTEST_CNNFACTORYVISITORTEST_H_
#define ATML_TEST_LAYERINTERLOCKTEST_CNNFACTORYVISITORTEST_H_

#include "CNN/CNNFactoryVisitor.h"

using namespace ATML::MachineLearning;

class CNNFactoryVisitorTest: public CNNFactoryVisitor
{
public:
	CNNFactoryVisitorTest(CNN* network);
	~CNNFactoryVisitorTest();

	virtual void Visit(const CNNConfig* const cnnConfig) override;
	virtual void Visit(const PerceptronLayerConfig* const perceptronConfig) override;
	virtual void Visit(const ConvolutionLayerConfig* const convolutionConfig) override;
	virtual void Visit(const StandardOutputLayerConfig* const convolutionConfig) override;

};

#endif /* ATML_TEST_LAYERINTERLOCKTEST_CNNFACTORYVISITORTEST_H_ */
