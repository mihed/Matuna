/*
* PerceptronLayer.h
*
*  Created on: May 5, 2015
*      Author: Mikael
*/

#ifndef MATUNA_MATUNA_OCLCONVNET_PERCEPTRONLAYER_H_
#define MATUNA_MATUNA_OCLCONVNET_PERCEPTRONLAYER_H_

#include "LayerKernel.h"
#include "Matuna.ConvNet/PerceptronLayerConfig.h"
#include "Matuna.Math/Matrix.h"
#include "Matuna.OCLHelper/OCLContext.h"

#include "OCLForwardBackPropLayer.h"

#include <unordered_map>
#include <vector>
#include <memory>

using namespace std;
using namespace Matuna::Helper;
using namespace Matuna::Math;

namespace Matuna
{
	namespace MachineLearning
	{

		template<class T>
		class PerceptronLayer: public OCLForwardBackPropLayer<T>
		{

		private:
			unordered_map<OCLDevice*, LayerKernel<T>*> deviceAndImageForwardKernels;
			unordered_map<OCLDevice*, LayerKernel<T>*> deviceAndImageBackKernels;
			unordered_map<OCLDevice*, LayerKernel<T>*> deviceAndImageGradientKernels;

			unique_ptr<OCLMemory> scalarCache;
			unordered_map<OCLDevice*, LayerKernel<T>*> deviceAndDivideByScalarKernels;
			unordered_map<OCLDevice*, LayerKernel<T>*> deviceAndSimpleSumKernels;

			unique_ptr<OCLMemory> weights;
			unique_ptr<OCLMemory> biases;
			PerceptronLayerConfig config;
			LayerDataDescription inputDescription;

		public:
			PerceptronLayer(shared_ptr<OCLContext> context,
				const vector<LayerDataDescription>& inputLayerDescriptions,
				MatunaActivationFunction backPropActivation,
				const PerceptronLayerConfig* config);
			virtual ~PerceptronLayer();

			virtual void InterlockFinalized() override;

			virtual void EnqueueForwardPropagation(OCLDevice* device, int queueIndex,
				OCLMemory* previousInput, OCLMemory* output, bool blocking =
				true) override;

			virtual void EnqueueBackPropagation(OCLDevice* device, int queueIndex,
				OCLMemory* previousInput, OCLMemory* delta,
				OCLMemory* deltaOutput, bool blocking = true) override;

			virtual vector<OCLMemory*> GetParameters() override;

			virtual void SetParameters(T* parameters, OCLDevice* device,
				int queueIndex, bool blocking = true) override;

			virtual void GetParameters(T* parameters, OCLDevice* device,
				int queueIndex, bool blocking = true) override;

			virtual size_t GetParameterCount() override;

			virtual void EnqueueCalculateGradient(OCLDevice* device, int queueIndex,
				OCLMemory* previousInput, OCLMemory* delta, vector<OCLMemory*> gradient, bool blocking = true) override;

			virtual vector<size_t> GetMultipleParameterCount() override;

			PerceptronLayerConfig GetConfig() const;

			Matrix<T> GetWeights();
			Matrix<T> GetBias();

		private:
			void InitializeMemoryDescriptions(const vector<LayerDataDescription>& inputLayerDescriptions, const PerceptronLayerConfig* config);
			void InitializeParameters();
			void InitializePrograms();

			void InitializeGradientPerceptronKernel(OCLDevice* device, OCLProgram* program);
			void InitializeBackPropPerceptronKernel(OCLDevice* device, OCLProgram* program);
			void InitializeForwardPerceptronKernel(OCLDevice* device, OCLProgram* program);
		};

	}
	/* namespace MachineLearning */
} /* namespace Matuna */

#endif /* MATUNA_MATUNA_OCLCONVNET_PERCEPTRONLAYER_H_ */
