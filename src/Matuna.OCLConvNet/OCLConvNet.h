/*
* OCLConvNet.h
*
*  Created on: May 5, 2015
*      Author: Mikael
*/

#ifndef MATUNA_MATUNA_OCLCONVNET_OCLCONVNET_H_
#define MATUNA_MATUNA_OCLCONVNET_OCLCONVNET_H_

#include "Matuna.ConvNet/ConvNetConfig.h"
#include "Matuna.ConvNet/ConvNet.h"
#include "Matuna.ConvNet/TrainableConvNet.h"
#include "Matuna.ConvNet/ConvNetTrainer.h"
#include "Matuna.ConvNet/IAlgorithmConfig.h"

#include "Matuna.OCLHelper/OCLDeviceInfo.h"
#include "Matuna.OCLHelper/OCLContext.h"
#include "Matuna.OCLHelper/OCLDevice.h"
#include "Matuna.OCLHelper/OCLMemory.h"

#include "OCLForwardBackPropLayer.h"
#include "StandardOutputLayer.h"

#include <memory>
#include <vector>

using namespace std;
using namespace Matuna::Helper;

namespace Matuna
{
	namespace MachineLearning
	{

		template<class T>
		class OCLConvNet final: public TrainableConvNet<T>
		{

		private:
			vector<shared_ptr<OCLContext>> contexts;
			vector<unique_ptr<OCLForwardBackPropLayer<T>>> layers;
			unique_ptr<StandardOutputLayer<T>> outputLayer;

			bool lowMemoryUsage;

		public:
			OCLConvNet(const vector<OCLDeviceInfo>& devices, unique_ptr<ConvNetConfig> config);
			virtual ~OCLConvNet();

			//TODO: the context index is bloody ugly and should be changed later. This is just a remainder that it needs to be fixed
			unique_ptr<OCLMemory> CreateInputMemory(T* input, int formatIndex, int contextIndex) const;

			//TODO: the context index is bloody ugly and should be changed later. This is just a remainder that it needs to be fixed
			unique_ptr<OCLMemory> CreateTargetMemory(T* target, int formatIndex, int contextIndex) const;


			unique_ptr<T[]> FeedForwardAligned(OCLMemory* input, int formatIndex);
			virtual unique_ptr<T[]> FeedForwardAligned(T* input, int formatIndex) override;


			//TODO: This function is not even useful at the moment since we need to send back and forth from the device. FIX PLEASE
			virtual T CalculateErrorFromForwardAligned(T* propagatedValue, int formatIndex, T* target) override;

			T CalculateErrorAligned(OCLMemory* input, int formatIndex, OCLMemory* target);
			virtual T CalculateErrorAligned(T* input, int formatIndex, T* target) override;

			unique_ptr<T[]> BackPropAligned(OCLMemory* input, int formatIndex, OCLMemory* target);
			virtual unique_ptr<T[]> BackPropAligned(T* input, int formatIndex, T* target) override;

			unique_ptr<T[]> CalculateGradientAligned(OCLMemory* input, int formatIndex, OCLMemory* target);
			virtual unique_ptr<T[]> CalculateGradientAligned(T* input, int formatIndex, T* target) override;

			virtual unique_ptr<T[]> GetParameters() override;

			virtual void SetParameters(T* parameters) override;

			virtual size_t GetParameterCount() override;

			virtual void TrainNetwork(unique_ptr<ConvNetTrainer<T>> trainer, unique_ptr<IAlgorithmConfig> algorithm) override;

			vector<OCLForwardBackPropLayer<T>*> GetLayers() const;
			StandardOutputLayer<T>* GetOutputLayer() const;
			vector<OCLContext*> GetOCLContexts() const;

		private:
			void InitializeContexts(const vector<OCLDeviceInfo>& devices);

			unique_ptr<T[]> FeedForwardLowMemory(OCLMemory* input, int formatIndex);
			unique_ptr<OCLMemory> FeedForwardLowMemoryOCLOutput(OCLMemory* input, int formatIndex);
			unique_ptr<T[]> FeedForwardHighMemory(OCLMemory* input, int formatIndex);
			void FeedForwardHighMemory(OCLMemory* input, int formatIndex, vector<unique_ptr<OCLMemory>>& outputMemories);

			unique_ptr<T[]> BackPropLowMemory(OCLMemory* input, int formatIndex, OCLMemory* target);
			unique_ptr<T[]> BackPropHighMemory(OCLMemory* input, int formatIndex, OCLMemory* target);

			unique_ptr<T[]> CalculateGradientLowMemory(OCLMemory* input, int formatIndex, OCLMemory* target);
			unique_ptr<T[]> CalculateGradientHighMemory(OCLMemory* input, int formatIndex, OCLMemory* target);

			T CalculateError(OCLMemory* lastOutput, int formatIndex, OCLMemory* target);
		};

	} /* namespace MachineLearning */
} /* namespace Matuna */

#endif /* MATUNA_MATUNA_OCLCONVNET_OCLCONVNET_H_ */
