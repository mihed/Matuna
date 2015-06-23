/*
* OCLConvNetFactoryVisitor.h
*
*  Created on: May 5, 2015
*      Author: Mikael
*/

#ifndef MATUNA_OCLCONVNET_OCLCONVNETFACTORYVISITOR_H_
#define MATUNA_OCLCONVNET_OCLCONVNETFACTORYVISITOR_H_

#include "Matuna.ConvNet/ConvNetFactoryVisitor.h"
#include "Matuna.OCLHelper/OCLContext.h"

#include <memory>

using namespace Matuna::Helper;

namespace Matuna
{
	namespace MachineLearning
	{

		class ConvNet;

		template<class T>
		class OCLConvNetFactoryVisitor: public ConvNetFactoryVisitor
		{
		private:
			shared_ptr<OCLContext> context;
		public:
			OCLConvNetFactoryVisitor(shared_ptr<OCLContext> context, ConvNet* network);
			virtual ~OCLConvNetFactoryVisitor();

			virtual void Visit(const ConvNetConfig* const cnnConfig) override;
			virtual void Visit(const PerceptronLayerConfig* const perceptronConfig)
				override;
			virtual void Visit(const ConvolutionLayerConfig* const convolutionConfig)
				override;
			virtual void Visit(const StandardOutputLayerConfig* const convolutionConfig)
				override;
			virtual void Visit(const VanillaSamplingLayerConfig* const vanillaConfig)
				override;
			virtual void Visit(const MaxPoolingLayerConfig* const maxPoolingConfig)
				override;
		};

	} /* namespace MachineLearning */
} /* namespace Matuna */

#endif /* MATUNA_OCLOCLCONVNET_OCLCONVNETFACTORYVISITOR_H_ */
