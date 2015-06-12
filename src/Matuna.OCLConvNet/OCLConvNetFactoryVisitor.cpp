/*
* OCLConvNetFactoryVisitor.cpp
*
*  Created on: May 5, 2015
*      Author: Mikael
*/

#include "OCLConvNetFactoryVisitor.h"
#include "PerceptronLayer.h"
//#include "ConvolutionLayer.h"
#include "StandardOutputLayer.h"

#include "Matuna.ConvNet/ConvNet.h"
#include "Matuna.ConvNet/PerceptronLayerConfig.h"
#include "Matuna.ConvNet/StandardOutputLayerConfig.h"
#include "Matuna.ConvNet/ConvolutionLayerConfig.h"
#include "Matuna.ConvNet/ConvNetConfig.h"
#include "Matuna.ConvNet/InterlockHelper.h"

namespace Matuna {
	namespace MachineLearning {

		template<class T>
		OCLConvNetFactoryVisitor<T>::OCLConvNetFactoryVisitor(
			shared_ptr<OCLContext> context, ConvNet* network) :
		ConvNetFactoryVisitor(network), context(context) {

		}

		template<class T>
		OCLConvNetFactoryVisitor<T>::~OCLConvNetFactoryVisitor() {

		}

		template<class T>
		void OCLConvNetFactoryVisitor<T>::Visit(const ConvNetConfig* const cnnConfig) {
			this->InitializeInterlock(cnnConfig);
		}

		template<class T>
		void OCLConvNetFactoryVisitor<T>::Visit(
			const PerceptronLayerConfig* const perceptronConfig) {

				if (backPropActivation == MatunaSoftMaxActivation)
					throw invalid_argument("The soft max is currently only supported on the outmost layer");

				if (perceptronConfig->ConnectionType() != MatunaFullConnection)
					throw invalid_argument("We only support full connection on the perceptron layer at the moment");

				unique_ptr<ForwardBackPropLayer> layer(
					new PerceptronLayer<T>(context, inputDataDescriptions,
					backPropActivation, perceptronConfig));

				this->InterlockAndAddLayer(perceptronConfig, move(layer));
		}

		template<class T>
		void OCLConvNetFactoryVisitor<T>::Visit(
			const ConvolutionLayerConfig* const convolutionConfig) {

				//if (backPropActivation == MatunaSoftMaxActivation)
				//	throw invalid_argument("The soft max is currently only supported on the outmost layer");

				//if (convolutionConfig->ConnectionType() != MatunaFullConnection)
				//	throw invalid_argument("We only support full connection on the convolution layer at the moment");

				//unique_ptr<ForwardBackPropLayer> layer(
				//	new ConvolutionLayer<T>(context, inputDataDescriptions,
				//	backPropActivation, convolutionConfig));

				//this->InterlockAndAddLayer(convolutionConfig, move(layer));
		}
		template<class T>
		void OCLConvNetFactoryVisitor<T>::Visit(
			const StandardOutputLayerConfig* const outputConfig) {
				for (auto& inputData : inputDataDescriptions)
					if (backPropActivation == MatunaSoftMaxActivation && inputData.Units == 1)
						throw invalid_argument("You cannot use Softmax together with only one unit. Use sigmoid instead!");

				unique_ptr<OutputLayer> layer(
					new StandardOutputLayer<T>(context, inputDataDescriptions,
					backPropActivation, outputConfig));

				this->InterlockAndAddLayer(outputConfig, move(layer));
		}

		template class OCLConvNetFactoryVisitor < cl_float > ;
		template class OCLConvNetFactoryVisitor < cl_double > ;

	} /* namespace MachineLearning */
} /* namespace Matuna */
