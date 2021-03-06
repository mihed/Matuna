/*
* VanillaSamplingLayer.h
*
*  Created on: Jun 21, 2015
*      Author: Mikael
*/

#ifndef MATUNA_MATUNA_OCLCONVNET_VANILLASAMPLINGLAYER_H_
#define MATUNA_MATUNA_OCLCONVNET_VANILLASAMPLINGLAYER_H_

#include "OCLForwardBackPropLayer.h"
#include "Matuna.ConvNet/VanillaSamplingLayerConfig.h"
#include "Matuna.OCLHelper/OCLContext.h"
#include "LayerKernel.h"

#include <unordered_map>
#include <vector>
#include <memory>

using namespace std;
using namespace Matuna::Helper;

namespace Matuna
{
	namespace MachineLearning
	{

		template<class T>
		class VanillaSamplingLayer: public OCLForwardBackPropLayer<T>
		{

		private:
			VanillaSamplingLayerConfig config;

			unordered_map<OCLDevice*, LayerKernel<T>*> deviceAndVanillaSamplingKernels;
			unordered_map<OCLDevice*, LayerKernel<T>*> deviceAndVanillaUpSamplingKernels;

		public:
			VanillaSamplingLayer(shared_ptr<OCLContext> context,
				const vector<LayerDataDescription>& inputLayerDescriptions,
				MatunaActivationFunction backPropActivation,
				const VanillaSamplingLayerConfig* config);
			~VanillaSamplingLayer();

			VanillaSamplingLayerConfig GetConfig() const;

			virtual void InterlockFinalized() override;

			virtual void EnqueueForwardPropagation(OCLDevice* device, int queueIndex,
				OCLMemory* previousInput, OCLMemory* output, bool blocking =
				true) override;

			virtual void EnqueueBackPropagation(OCLDevice* device, int queueIndex,
				OCLMemory* previousInput, OCLMemory* delta,
				OCLMemory* deltaOutput, bool blocking = true) override;

			virtual vector<OCLMemory*> GetParameters() override;

			virtual void GetParameters(T* parameters, OCLDevice* device,
				int queueIndex, bool blocking = true) override;

			virtual void SetParameters(T* parameters, OCLDevice* device,
				int queueIndex, bool blocking = true) override;

			virtual void EnqueueCalculateGradient(OCLDevice* device, int queueIndex,
				OCLMemory* previousInput, OCLMemory* delta, vector<OCLMemory*> gradient, bool blocking = true) override;

			virtual vector<size_t> GetMultipleParameterCount() override;

			virtual size_t GetParameterCount() override;

		private:
			void InitializeMemoryDescriptions(const vector<LayerDataDescription>& inputLayerDescriptions, const VanillaSamplingLayerConfig* config);
			void InitializePrograms();
			void InitializeVanillaSamplingKernel(OCLDevice* device, OCLProgram* program);
			void InitializeVanillaUpSamplingKernel(OCLDevice* device, OCLProgram* program);
		};

	} /* namespace MachineLearning */
} /* namespace Matuna */

#endif /* MATUNA_MATUNA_OCLCONVNET_VANILLASAMPLINGLAYER_H_ */
