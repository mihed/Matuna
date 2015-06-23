/*
* MaxPoolingLayer.h
*
*  Created on: Jun 23, 2015
*      Author: Mikael
*/

#ifndef MATUNA_MATUNA_OCLCONVNET_MAXPOOLINGLAYER_H_
#define MATUNA_MATUNA_OCLCONVNET_MAXPOOLINGLAYER_H_

#include "OCLForwardBackPropLayer.h"
#include "Matuna.ConvNet/MaxPoolingLayerConfig.h"
#include "Matuna.OCLHelper/OCLContext.h"
#include "LayerKernel.h"

#include <unordered_map>
#include <vector>
#include <memory>

namespace Matuna
{
	namespace MachineLearning
	{

		template <class T>
		class MaxPoolingLayer: public OCLForwardBackPropLayer<T>
		{

		private:
			MaxPoolingLayerConfig config;

		public:
			MaxPoolingLayer(shared_ptr<OCLContext> context,
				const vector<LayerDataDescription>& inputLayerDescriptions,
				MatunaActivationFunction backPropActivation,
				const MaxPoolingLayerConfig* config);
			~MaxPoolingLayer();

			MaxPoolingLayerConfig GetConfig() const;

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
		};

	} /* namespace MachineLearning */
} /* namespace Matuna */

#endif /* MATUNA_MATUNA_OCLCONVNET_MAXPOOLINGLAYER_H_ */
