/*
 * CNNOpenCLFactoryVisitor.cpp
 *
 *  Created on: May 5, 2015
 *      Author: Mikael
 */

#include "CNNOpenCLFactoryVisitor.h"
#include "CNN/PerceptronLayerConfig.h"
#include "CNN/StandardOutputLayerConfig.h"
#include "CNN/ConvolutionLayerConfig.h"
#include "CNN/CNNConfig.h"
#include "CNN/InterlockHelper.h"

namespace ATML
{
namespace MachineLearning
{

CNNOpenCLFactoryVisitor::CNNOpenCLFactoryVisitor()
{

}

CNNOpenCLFactoryVisitor::~CNNOpenCLFactoryVisitor()
{

}

void CNNOpenCLFactoryVisitor::Visit(const CNNConfig* const cnnConfig)
{
	auto inputData = cnnConfig->InputDataDescription();
	auto inputMemory = cnnConfig->InputMemoryProposal();
	if (!InterlockHelper::IsCompatible(inputData, inputMemory))
		throw runtime_error("Invalid cnn config memory and data description");

	forwardInputProposal = inputMemory;
	backOutputProposal = inputMemory;
	inputDataDescription = inputData;
}

void CNNOpenCLFactoryVisitor::Visit(
		const PerceptronLayerConfig* const perceptronConfig)
{

}

void CNNOpenCLFactoryVisitor::Visit(
		const ConvolutionLayerConfig* const convolutionConfig)
{

}

void CNNOpenCLFactoryVisitor::Visit(
		const StandardOutputLayerConfig* const convolutionConfig)
{

}

} /* namespace MachineLearning */
} /* namespace ATML */
