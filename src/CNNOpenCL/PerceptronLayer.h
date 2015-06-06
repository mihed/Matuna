/*
* PerceptronLayer.h
*
*  Created on: May 5, 2015
*      Author: Mikael
*/

#ifndef MATUNA_CNNOPENCL_PERCEPTRONLAYER_H_
#define MATUNA_CNNOPENCL_PERCEPTRONLAYER_H_

#include "OpenCLForwardBackPropLayer.h"
#include "BackPerceptronKernel.h"
#include "ForwardPerceptronKernel.h"
#include "GradientPerceptronKernel.h"
#include "SimpleSumKernel.h"
#include "ImageForwardPerceptronKernel.h"
#include "ImageBackPerceptronKernel.h"
#include "ImageGradientPerceptronKernel.h"
#include "DivideByScalarKernel.h"
#include "CNN/PerceptronLayerConfig.h"
#include "Math/Matrix.h"
#include "OpenCLHelper/OpenCLContext.h"

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
		class PerceptronLayer: public OpenCLForwardBackPropLayer<T>
		{

		private:
			unordered_map<OpenCLDevice*, unique_ptr<ForwardPerceptronKernel<T>>> deviceAndForwardKernels;
			unordered_map<OpenCLDevice*, unique_ptr<ImageForwardPerceptronKernel<T>>> deviceAndImageForwardKernels;
			unordered_map<OpenCLDevice*, unique_ptr<BackPerceptronKernel<T>>> deviceAndBackKernels;
			unordered_map<OpenCLDevice*, unique_ptr<ImageBackPerceptronKernel<T>>> deviceAndImageBackKernels;
			unordered_map<OpenCLDevice*, unique_ptr<GradientPerceptronKernel<T>>> deviceAndGradientKernels;
			unordered_map<OpenCLDevice*, unique_ptr<ImageGradientPerceptronKernel<T>>> deviceAndImageGradientKernels;

			unique_ptr<OpenCLMemory> scalarCache;
			unordered_map<OpenCLDevice*, unique_ptr<DivideByScalarKernel<T>>> deviceAndDivideByScalarKernels;
			unordered_map<OpenCLDevice*, unique_ptr<SimpleSumKernel<T>>> deviceAndSimpleSumKernels;

			unique_ptr<OpenCLMemory> weights;
			unique_ptr<OpenCLMemory> biases;
			PerceptronLayerConfig config;
			LayerDataDescription inputDescription;

			bool useImage;

		public:
			PerceptronLayer(shared_ptr<OpenCLContext> context,
				const vector<LayerDataDescription>& inputLayerDescriptions,
				MatunaActivationFunction backPropActivation,
				const PerceptronLayerConfig* config);
			virtual ~PerceptronLayer();

			virtual void InterlockFinalized() override;

			virtual void EnqueueForwardPropagation(OpenCLDevice* device, int queueIndex,
				OpenCLMemory* previousInput, OpenCLMemory* output, bool blocking =
				true) override;

			virtual void EnqueueBackPropagation(OpenCLDevice* device, int queueIndex,
				OpenCLMemory* previousInput, OpenCLMemory* delta,
				OpenCLMemory* deltaOutput, bool blocking = true) override;

			virtual void EnqueueCalculateGradient(OpenCLDevice* device, int queueIndex,
				OpenCLMemory* previousInput, OpenCLMemory* delta, OpenCLMemory* gradient, bool blocking = true) override;

			virtual vector<OpenCLMemory*> GetParameters() override;

			virtual void SetParameters(T* parameters, OpenCLDevice* device,
				int queueIndex, bool blocking = true) override;

			virtual void GetParameters(T* parameters, OpenCLDevice* device,
				int queueIndex, bool blocking = true) override;

			virtual size_t GetParameterCount() override;

			virtual void EnqueueCalculateGradient(OpenCLDevice* device, int queueIndex,
				OpenCLMemory* previousInput, OpenCLMemory* delta, vector<OpenCLMemory*> gradient, bool blocking = true) override;

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

#endif /* MATUNA_CNNOPENCL_PERCEPTRONLAYER_H_ */
