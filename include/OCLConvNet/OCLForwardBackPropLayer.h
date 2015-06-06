/*
* OCLForwardBackPropLayer.h
*
*  Created on: May 5, 2015
*      Author: Mikael
*/

#ifndef MATUNA_OCLConvNet_OCLFORWARDBACKPROPLAYER_H_
#define MATUNA_OCLConvNet_OCLFORWARDBACKPROPLAYER_H_

#include "ConvNet/ForwardBackPropLayer.h"
#include "OCLHelper/OCLContext.h"
#include "OCLHelper/OCLMemory.h"
#include <memory>
#include <tuple>
#include <vector>

using namespace Matuna::Helper;
using namespace std;

namespace Matuna
{
	namespace MachineLearning
	{

		template<class T>
		class OCLForwardBackPropLayer: public ForwardBackPropLayer
		{
		protected:
			shared_ptr<OCLContext> context;

		public:
			OCLForwardBackPropLayer(shared_ptr<OCLContext> context,
				const vector<LayerDataDescription>& inputLayerDescriptions,
				MatunaActivationFunction backPropActivation,
				const ForwardBackPropLayerConfig* config);
			virtual ~OCLForwardBackPropLayer();

			virtual void EnqueueForwardPropagation(OCLDevice* device, int queueIndex,
				OCLMemory* previousInput, OCLMemory* output, bool blocking =
				true) = 0;

			virtual void EnqueueBackPropagation(OCLDevice* device, int queueIndex,
				OCLMemory* previousInput, OCLMemory* delta,
				OCLMemory* deltaOutput, bool blocking = true) = 0;

			//Deprecated!
			virtual void EnqueueCalculateGradient(OCLDevice* device, int queueIndex,
				OCLMemory* previousInput, OCLMemory* delta, OCLMemory* gradient, bool blocking = true) = 0;

			virtual void EnqueueCalculateGradient(OCLDevice* device, int queueIndex,
				OCLMemory* previousInput, OCLMemory* delta, vector<OCLMemory*> gradient, bool blocking = true) = 0;

			virtual vector<size_t> GetMultipleParameterCount() = 0;

			virtual vector<OCLMemory*> GetParameters() = 0;

			virtual void GetParameters(T* parameters, OCLDevice* device,
				int queueIndex, bool blocking = true) = 0;

			virtual void SetParameters(T* parameters, OCLDevice* device,
				int queueIndex, bool blocking = true) = 0;

			virtual size_t GetParameterCount() = 0;
		};

	} /* namespace MachineLearning */
} /* namespace Matuna */

#endif /* MATUNA_OCLConvNet_OCLFORWARDBACKPROPLAYER_H_ */
