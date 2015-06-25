/*
* StandardOutputLayer.h
*
*  Created on: May 5, 2015
*      Author: Mikael
*/

#ifndef MATUNA_MATUNA_OCLCONVNET_STANDARDOUTPUTLAYER_H_
#define MATUNA_MATUNA_OCLCONVNET_STANDARDOUTPUTLAYER_H_

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
			void InitializePrograms();
			void InitializeMemoryDescriptions(const vector<LayerDataDescription>& inputLayerDescriptions, const StandardOutputLayerConfig* config);

			void InitializeErrorKernel(OCLDevice* device, OCLProgram* program);
			void InitializeOutputKernel(OCLDevice* device, OCLProgram* program);
		};

	}
	/* namespace MachineLearning */
} /* namespace Matuna */

#endif /* MATUNA_MATUNA_OCLCONVNET_STANDARDOUTPUTLAYER_H_ */
