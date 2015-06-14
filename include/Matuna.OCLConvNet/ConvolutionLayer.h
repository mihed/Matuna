/*
* ConvolutionLayer.h
*
*  Created on: May 5, 2015
*      Author: Mikael
*/

#ifndef MATUNA_OCLCONVNET_CONVOLUTIONLAYER_H_
#define MATUNA_OCLCONVNET_CONVOLUTIONLAYER_H_

#include "OCLForwardBackPropLayer.h"
#include "Matuna.ConvNet/ConvolutionLayerConfig.h"
#include "Matuna.Math/Matrix.h"
#include "Matuna.OCLHelper/OCLContext.h"
#include "LayerKernel.h"

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
		class ConvolutionLayer: public OCLForwardBackPropLayer<T>
		{
		private:

			unordered_map<OCLDevice*, LayerKernel<T>*> deviceAndConvolutionKernels;
			unordered_map<OCLDevice*, LayerKernel<T>*> deviceAndSumKernels;
			unordered_map<OCLDevice*, LayerKernel<T>*> deviceAndBackConvolutionKernels;
			unordered_map<OCLDevice*, LayerKernel<T>*> deviceAndMultiplyKernels;
			unordered_map<OCLDevice*, LayerKernel<T>*> deviceAndZeroKernels;
			unordered_map<OCLDevice*, LayerKernel<T>*> deviceAndSumUnitKernels;
			unordered_map<OCLDevice*, LayerKernel<T>*> deviceAndMultiplyWithOffsetKernels;

			//HACK: this must be changed when removed the deprecated functions 
			unordered_map<OCLDevice*,  LayerKernel<T>*> deviceAndSumUnitKernels2;

			ConvolutionLayerConfig convolutionConfig;
			unique_ptr<OCLMemory> filters;
			unique_ptr<OCLMemory> biases;
			unique_ptr<OCLMemory> summaryCache;

		public:
			ConvolutionLayer(shared_ptr<OCLContext> context,
				const vector<LayerDataDescription>& inputLayerDescriptions,
				MatunaActivationFunction backPropActivation,
				const ConvolutionLayerConfig* config);
			virtual ~ConvolutionLayer();

			ConvolutionLayerConfig GetConfig() const;
			vector<Matrix<T>> GetFilters() const;
			vector<T> GetBiases() const;

			virtual void InterlockFinalized() override;

			virtual void EnqueueForwardPropagation(OCLDevice* device, int queueIndex,
				OCLMemory* previousInput, OCLMemory* output, bool blocking =
				true) override;

			virtual void EnqueueBackPropagation(OCLDevice* device, int queueIndex,
				OCLMemory* previousInput, OCLMemory* delta,
				OCLMemory* deltaOutput, bool blocking = true) override;

			virtual void EnqueueCalculateGradient(OCLDevice* device, int queueIndex,
				OCLMemory* previousInput, OCLMemory* delta, OCLMemory* gradient, bool blocking = true) override;

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
			void InitializeParameters();
			void InitializeProgram(unordered_map<OCLDevice*, unique_ptr<OCLProgram>>& programs);
		};

	} /* namespace MachineLearning */
} /* namespace Matuna */

#endif /* MATUNA_OCLCONVNET_CONVOLUTIONLAYER_H_ */
