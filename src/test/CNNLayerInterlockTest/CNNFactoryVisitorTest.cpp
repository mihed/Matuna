/*
 * CNNFactoryVisitorTest.cpp
 *
 *  Created on: May 5, 2015
 *      Author: Mikael
 */

#include "CNN/PerceptronLayerConfig.h"
#include "CNN/StandardOutputLayerConfig.h"
#include "CNN/ConvolutionLayerConfig.h"
#include "CNN/CNNConfig.h"
#include "CNN/CNN.h"
#include "CNN/InterlockHelper.h"
#include "CNNFactoryVisitorTest.h"
#include "ForthBackPropLayerTest.h"
#include "OutputLayerTest.h"

CNNFactoryVisitorTest::CNNFactoryVisitorTest(CNN* network)
	: CNNFactoryVisitor(network)
{

}

CNNFactoryVisitorTest::~CNNFactoryVisitorTest()
{

}

void CNNFactoryVisitorTest::Visit(const CNNConfig* const cnnConfig)
{
	this->InitializeInterlock(cnnConfig);
}

void CNNFactoryVisitorTest::Visit(
		const PerceptronLayerConfig* const perceptronConfig)
{
	unique_ptr<ForwardBackPropLayer> layer(
			new ForthBackPropLayerTest(inputDataDescriptions, backPropActivation, perceptronConfig));

	this->InterlockAndAddLayer(perceptronConfig, move(layer));
}

void CNNFactoryVisitorTest::Visit(
		const ConvolutionLayerConfig* const convolutionConfig)
{
	unique_ptr<ForwardBackPropLayer> layer(
			new ForthBackPropLayerTest(inputDataDescriptions,
			backPropActivation,
					convolutionConfig));

	this->InterlockAndAddLayer(convolutionConfig, move(layer));
}

void CNNFactoryVisitorTest::Visit(
		const StandardOutputLayerConfig* const outputConfig)
{
	unique_ptr<OutputLayer> layer(
		new OutputLayerTest(inputDataDescriptions, backPropActivation, outputConfig));

	this->InterlockAndAddLayer(outputConfig, move(layer));
}

