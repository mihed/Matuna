/*
 * ConvNetFactoryVisitorTest.cpp
 *
 *  Created on: May 5, 2015
 *      Author: Mikael
 */

#include "ConvNet/PerceptronLayerConfig.h"
#include "ConvNet/StandardOutputLayerConfig.h"
#include "ConvNet/ConvolutionLayerConfig.h"
#include "ConvNet/ConvNetConfig.h"
#include "ConvNet/ConvNet.h"
#include "ConvNet/InterlockHelper.h"
#include "ConvNetFactoryVisitorTest.h"
#include "ForthBackPropLayerTest.h"
#include "OutputLayerTest.h"

ConvNetFactoryVisitorTest::ConvNetFactoryVisitorTest(ConvNet* network)
	: ConvNetFactoryVisitor(network)
{

}

ConvNetFactoryVisitorTest::~ConvNetFactoryVisitorTest()
{

}

void ConvNetFactoryVisitorTest::Visit(const ConvNetConfig* const cnnConfig)
{
	this->InitializeInterlock(cnnConfig);
}

void ConvNetFactoryVisitorTest::Visit(
		const PerceptronLayerConfig* const perceptronConfig)
{
	unique_ptr<ForwardBackPropLayer> layer(
			new ForthBackPropLayerTest(inputDataDescriptions, backPropActivation, perceptronConfig));

	this->InterlockAndAddLayer(perceptronConfig, move(layer));
}

void ConvNetFactoryVisitorTest::Visit(
		const ConvolutionLayerConfig* const convolutionConfig)
{
	unique_ptr<ForwardBackPropLayer> layer(
			new ForthBackPropLayerTest(inputDataDescriptions,
			backPropActivation,
					convolutionConfig));

	this->InterlockAndAddLayer(convolutionConfig, move(layer));
}

void ConvNetFactoryVisitorTest::Visit(
		const StandardOutputLayerConfig* const outputConfig)
{
	unique_ptr<OutputLayer> layer(
		new OutputLayerTest(inputDataDescriptions, backPropActivation, outputConfig));

	this->InterlockAndAddLayer(outputConfig, move(layer));
}

