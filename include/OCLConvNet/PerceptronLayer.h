/*
* PerceptronLayer.h
*
*  Created on: May 5, 2015
*      Author: Mikael
*/

#ifndef MATUNA_OCLConvNet_PERCEPTRONLAYER_H_
#define MATUNA_OCLConvNet_PERCEPTRONLAYER_H_

#include "OCLForwardBackPropLayer.h"
#include "BackPerceptronKernel.h"
#include "ForwardPerceptronKernel.h"
#include "GradientPerceptronKernel.h"
#include "SimpleSumKernel.h"
#include "ImageForwardPerceptronKernel.h"
#include "ImageBackPerceptronKernel.h"
#include "ImageGradientPerceptronKernel.h"
#include "DivideByScalarKernel.h"
#include "ConvNet/PerceptronLayerConfig.h"
#include "Math/Matrix.h"
#include "OCLHelper/OCLContext.h"

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
			unordered_map<OCLDevice*, unique_ptr<ForwardPerceptronKernel<T>>> deviceAndForwardKernels;
			unordered_map<OCLDevice*, unique_ptr<ImageForwardPerceptronKernel<T>>> deviceAndImageForwardKernels;
			unordered_map<OCLDevice*, unique_ptr<BackPerceptronKernel<T>>> deviceAndBackKernels;
			unordered_map<OCLDevice*, unique_ptr<ImageBackPerceptronKernel<T>>> deviceAndImageBackKernels;
			unordered_map<OCLDevice*, unique_ptr<GradientPerceptronKernel<T>>> deviceAndGradientKernels;
			unordered_map<OCLDevice*, unique_ptr<ImageGradientPerceptronKernel<T>>> deviceAndImageGradientKernels;

			unique_ptr<OCLMemory> scalarCache;
			unordered_map<OCLDevice*, unique_ptr<DivideByScalarKernel<T>>> deviceAndDivideByScalarKernels;
			unordered_map<OCLDevice*, unique_ptr<SimpleSumKernel<T>>> deviceAndSimpleSumKernels;

			unique_ptr<OCLMemory> weights;
			unique_ptr<OCLMemory> biases;
			PerceptronLayerConfig config;
			LayerDataDescription inputDescription;

			bool useImage;

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

			virtual void EnqueueCalculateGradient(OCLDevice* device, int queueIndex,
				OCLMemory* previousInput, OCLMemory* delta, OCLMemory* gradient, bool blocking = true) override;

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

			//TODO: Add some read / write parameters. Now it's all random

		private:
			void InitializeNormalForwardPerceptron();
			void InitializeImageForwardPerceptron();
			void InitializeNormalBackPerceptron();
			void InitializeImageBackPerceptron();
			void InitializeNormalGradientKernel();
			void InitializeImageGradientKernel();
			void InitializeParameters();
		};

	}
	/* namespace MachineLearning */
} /* namespace Matuna */

#endif /* MATUNA_OCLConvNet_PERCEPTRONLAYER_H_ */
