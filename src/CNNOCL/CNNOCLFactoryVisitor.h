/*
 * CNNOCLFactoryVisitor.h
 *
 *  Created on: May 5, 2015
 *      Author: Mikael
 */

#ifndef MATUNA_CNNOCL_CNNOCLFACTORYVISITOR_H_
#define MATUNA_CNNOCL_CNNOCLFACTORYVISITOR_H_

#include "CNN/CNNFactoryVisitor.h"
#include "OCLHelper/OCLContext.h"

#include <memory>

using namespace Matuna::Helper;

namespace Matuna
{
namespace MachineLearning
{

class CNN;

template<class T>
class CNNOCLFactoryVisitor: public CNNFactoryVisitor
{
private:
	shared_ptr<OCLContext> context;
public:
	CNNOCLFactoryVisitor(shared_ptr<OCLContext> context, CNN* network);
	virtual ~CNNOCLFactoryVisitor();

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

#endif /* MATUNA_CNNOCL_CNNOCLFACTORYVISITOR_H_ */
