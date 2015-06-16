/*
* StandardOutputLayer.h
*
*  Created on: May 5, 2015
*      Author: Mikael
*/

#ifndef MATUNA_OCLConvNet_STANDARDOUTPUTLAYER_H_
#define MATUNA_OCLConvNet_STANDARDOUTPUTLAYER_H_

#include "Matuna.ConvNet/OutputLayer.h"
#include "Matuna.ConvNet/StandardOutputLayerConfig.h"
#include "Matuna.OCLHelper/OCLContext.h"
#include "Matuna.OCLHelper/OCLDevice.h"
#include "Matuna.OCLHelper/OCLMemory.h"
#include "LayerKernel.h"
#include <memory>
#include <unordered_map>

using namespace std;
using namespace Matuna::Helper;

namespace Matuna
{
	namespace MachineLearning
	{

		template<class T>
		class StandardOutputLayer final: public OutputLayer
		{
		private:
			shared_ptr<OCLContext> context;
			unordered_map<OCLDevice*, LayerKernel<T>*> imageOutputKernels;
			unordered_map<OCLDevice*, LayerKernel<T>*> imageErrorKernels;

			StandardOutputLayerConfig config;
			LayerDataDescription inputDescription;

		public:
			StandardOutputLayer(shared_ptr<OCLContext> context,
				const vector<LayerDataDescription>& inputLayerDescriptions,
				MatunaActivationFunction backPropActivation,
				const StandardOutputLayerConfig* outputLayerConfig);
			~StandardOutputLayer();

			virtual void InterlockFinalized() override;

			void EnqueueBackPropagation(OCLDevice* device, int queueIndex,
				OCLMemory* previousInput, OCLMemory* target,
				OCLMemory* deltaOutput, bool blocking = true);

			T CalculateError(OCLDevice* device,
				int queueIndex, OCLMemory* previousInput, OCLMemory* target);

		private:
			void InitializeImageProgram(unordered_map<OCLDevice*, unique_ptr<OCLProgram>>& programs);
			void SetErrorFunctionDefine(LayerKernel<T>* kernel, string path, bool binary);
		};

	}
	/* namespace MachineLearning */
} /* namespace Matuna */

#endif /* OCLConvNet_STANDARDOUTPUTLAYER_H_ */
