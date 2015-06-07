/*
 * ConvNetFactoryVisitorTest.h
 *
 *  Created on: May 5, 2015
 *      Author: Mikael
 */

#ifndef MATUNA_TEST_LAYERINTERLOCKTEST_CONVNETFACTORYVISITORTEST_H_
#define MATUNA_TEST_LAYERINTERLOCKTEST_CONVNETFACTORYVISITORTEST_H_

#include "Matuna.ConvNet/ConvNetFactoryVisitor.h"

using namespace Matuna::MachineLearning;

class ConvNetFactoryVisitorTest: public ConvNetFactoryVisitor
{
public:
	ConvNetFactoryVisitorTest(ConvNet* network);
	~ConvNetFactoryVisitorTest();

	virtual void Visit(const ConvNetConfig* const cnnConfig) override;
	virtual void Visit(const PerceptronLayerConfig* const perceptronConfig) override;
	virtual void Visit(const ConvolutionLayerConfig* const convolutionConfig) override;
	virtual void Visit(const StandardOutputLayerConfig* const convolutionConfig) override;

};

#endif /* MATUNA_TEST_LAYERINTERLOCKTEST_CONVNETFACTORYVISITORTEST_H_ */
