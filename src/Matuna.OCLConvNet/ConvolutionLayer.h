/*
* ConvolutionLayer.h
*
*  Created on: May 5, 2015
*      Author: Mikael
*/

#ifndef MATUNA_OCLConvNet_CONVOLUTIONLAYER_H_
#define MATUNA_OCLConvNet_CONVOLUTIONLAYER_H_

#include "OCLForwardBackPropLayer.h"
#include "Matuna.ConvNet/ConvolutionLayerConfig.h"
#include "Matuna.Math/Matrix.h"
#include "Matuna.OCLHelper/OCLContext.h"

#include "ConvolutionKernel.h"
#include "BackConvolutionKernel.h"
#include "MultiplyAllUnitsKernel.h"
#include "ZeroBorderKenel.h"
#include "SumAllUnitsKernel.h"
#include "SumUnitKernel.h"
#include "MultiplyWithOffsetKernel.h"

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

			unordered_map<OCLDevice*, unique_ptr<ConvolutionKernel<T>>> deviceAndConvolutionKernels;
			unordered_map<OCLDevice*, unique_ptr<SumAllUnitsKernel<T>>> deviceAndSumKernels;
			unordered_map<OCLDevice*, unique_ptr<BackConvolutionKernel<T>>> deviceAndBackConvolutionKernels;
			unordered_map<OCLDevice*, unique_ptr<MultiplyAllUnitsKernel<T>>> deviceAndMultiplyKernels;
			unordered_map<OCLDevice*, unique_ptr<ZeroBorderKenel<T>>> deviceAndZeroKernels;
			unordered_map<OCLDevice*, unique_ptr<SumUnitKernel<T>>> deviceAndSumUnitKernels;
			unordered_map<OCLDevice*, unique_ptr<MultiplyWithOffsetKernel<T>>> deviceAndMultiplyWithOffsetKernels;

			//HACK: this must be changed when removed the deprecated functions 
			unordered_map<OCLDevice*, unique_ptr<SumUnitKernel<T>>> deviceAndSumUnitKernels2;

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
			void InitializeConvolutionKernel();
			void InitializeSumAllKernel();
			void InitializeBackConvolutionKernel();
			void InitializeMultiplyKernel();
			void InitializeZeroKernel();
			void InitializeSumUnitKernel();
			void InitializeMultiplyWithOffsetKernel();
		};

	} /* namespace MachineLearning */
} /* namespace Matuna */

#endif /* MATUNA_OCLConvNet_CONVOLUTIONLAYER_H_ */
