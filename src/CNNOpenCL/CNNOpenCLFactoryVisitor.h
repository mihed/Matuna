/*
 * CNNOpenCLFactoryVisitor.h
 *
 *  Created on: May 5, 2015
 *      Author: Mikael
 */

#ifndef MATUNA_CNNOPENCL_CNNOPENCLFACTORYVISITOR_H_
#define MATUNA_CNNOPENCL_CNNOPENCLFACTORYVISITOR_H_

#include "CNN/CNNFactoryVisitor.h"
#include "OpenCLHelper/OpenCLContext.h"

#include <memory>

using namespace Matuna::Helper;

namespace Matuna
{
namespace MachineLearning
{

class CNN;

template<class T>
class CNNOpenCLFactoryVisitor: public CNNFactoryVisitor
{
private:
	shared_ptr<OpenCLContext> context;
public:
	CNNOpenCLFactoryVisitor(shared_ptr<OpenCLContext> context, CNN* network);
	virtual ~CNNOpenCLFactoryVisitor();

	virtual void Visit(const CNNConfig* const cnnConfig) override;
	virtual void Visit(const PerceptronLayerConfig* const perceptronConfig)
			override;
	virtual void Visit(const ConvolutionLayerConfig* const convolutionConfig)
			override;
	virtual void Visit(const StandardOutputLayerConfig* const convolutionConfig)
			override;
};

} /* namespace MachineLearning */
} /* namespace Matuna */

#endif /* MATUNA_CNNOPENCL_CNNOPENCLFACTORYVISITOR_H_ */
