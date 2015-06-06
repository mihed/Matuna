/*
 * ConvNetOCLFactoryVisitor.h
 *
 *  Created on: May 5, 2015
 *      Author: Mikael
 */

#ifndef MATUNA_ConvNetOCL_ConvNetOCLFACTORYVISITOR_H_
#define MATUNA_ConvNetOCL_ConvNetOCLFACTORYVISITOR_H_

#include "ConvNet/ConvNetFactoryVisitor.h"
#include "OCLHelper/OCLContext.h"

#include <memory>

using namespace Matuna::Helper;

namespace Matuna
{
namespace MachineLearning
{

class ConvNet;

template<class T>
class ConvNetOCLFactoryVisitor: public ConvNetFactoryVisitor
{
private:
	shared_ptr<OCLContext> context;
public:
	ConvNetOCLFactoryVisitor(shared_ptr<OCLContext> context, ConvNet* network);
	virtual ~ConvNetOCLFactoryVisitor();

	virtual void Visit(const ConvNetConfig* const cnnConfig) override;
	virtual void Visit(const PerceptronLayerConfig* const perceptronConfig)
			override;
	virtual void Visit(const ConvolutionLayerConfig* const convolutionConfig)
			override;
	virtual void Visit(const StandardOutputLayerConfig* const convolutionConfig)
			override;
};

} /* namespace MachineLearning */
} /* namespace Matuna */

#endif /* MATUNA_ConvNetOCL_ConvNetOCLFACTORYVISITOR_H_ */
