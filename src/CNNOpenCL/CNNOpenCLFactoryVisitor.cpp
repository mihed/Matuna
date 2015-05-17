/*
 * CNNOpenCLFactoryVisitor.cpp
 *
 *  Created on: May 5, 2015
 *      Author: Mikael
 */

#include "CNNOpenCLFactoryVisitor.h"
#include "PerceptronLayer.h"
#include "ConvolutionLayer.h"
#include "StandardOutputLayer.h"

#include "CNN/CNN.h"
#include "CNN/PerceptronLayerConfig.h"
#include "CNN/StandardOutputLayerConfig.h"
#include "CNN/ConvolutionLayerConfig.h"
#include "CNN/CNNConfig.h"
#include "CNN/InterlockHelper.h"

namespace ATML
{
namespace MachineLearning
{

template class CNNOpenCLFactoryVisitor<cl_float> ;
template class CNNOpenCLFactoryVisitor<cl_double> ;

template<class T>
CNNOpenCLFactoryVisitor<T>::CNNOpenCLFactoryVisitor(
		shared_ptr<OpenCLContext> context, CNN* network) :
		CNNFactoryVisitor(network), context(context)
{

}

template<class T>
CNNOpenCLFactoryVisitor<T>::~CNNOpenCLFactoryVisitor()
{

}

template<class T>
void CNNOpenCLFactoryVisitor<T>::Visit(const CNNConfig* const cnnConfig)
{
	this->InitializeInterlock(cnnConfig);
}

template<class T>
void CNNOpenCLFactoryVisitor<T>::Visit(
		const PerceptronLayerConfig* const perceptronConfig)
{
	unique_ptr<ForwardBackPropLayer> layer(
			new PerceptronLayer<T>(context, inputDataDescriptions, backPropActivation,
					perceptronConfig));

	this->InterlockAndAddLayer(perceptronConfig, move(layer));
}

template<class T>
void CNNOpenCLFactoryVisitor<T>::Visit(
		const ConvolutionLayerConfig* const convolutionConfig)
{
	unique_ptr<ForwardBackPropLayer> layer(
			new ConvolutionLayer<T>(context, inputDataDescriptions, backPropActivation,
					convolutionConfig));

	this->InterlockAndAddLayer(convolutionConfig, move(layer));
}
template<class T>
void CNNOpenCLFactoryVisitor<T>::Visit(
		const StandardOutputLayerConfig* const outputConfig)
{
	unique_ptr<OutputLayer> layer(
			new StandardOutputLayer<T>(context, inputDataDescriptions, backPropActivation,
					outputConfig));

	this->InterlockAndAddLayer(outputConfig, move(layer));
}

} /* namespace MachineLearning */
} /* namespace ATML */
