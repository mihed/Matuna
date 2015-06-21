/*
* VanillaSamplingLayer.cpp
*
*  Created on: Jun 21, 2015
*      Author: Mikael
*/

#include "VanillaSamplingLayer.h"
#include "CheckPrecision.h"

namespace Matuna
{
	namespace MachineLearning
	{

		template<class T>
		VanillaSamplingLayer<T>::VanillaSamplingLayer(shared_ptr<OCLContext> context,
			const vector<LayerDataDescription>& inputLayerDescriptions,
			MatunaActivationFunction backPropActivation,
			const VanillaSamplingLayerConfig* config) :
		OCLForwardBackPropLayer<T>(context, inputLayerDescriptions,
			backPropActivation, config), config(*config)
		{
			if (inputLayerDescriptions.size() == 0)
				throw invalid_argument(
				"There's no input data descriptions for the convolution layer.");

			if (inputLayerDescriptions.size() != 1)
				throw runtime_error("Not implemented exception");

			//Make sure the type we want to execute is supported on the device.
			static_assert(is_same<cl_double, T>::value || is_same<cl_float, T>::value, "The type is not yet supported");
			for (auto device : context->GetDevices()) 
			{
				auto deviceInfo = device->DeviceInfo();
				CheckPrecision<is_same<cl_double, T>::value>::Check(deviceInfo);
			}
		}

		template<class T>
		VanillaSamplingLayer<T>::~VanillaSamplingLayer()
		{

		}

		template<class T>
		VanillaSamplingLayerConfig VanillaSamplingLayer<T>::GetConfig() const
		{
			return config;
		}

		template<class T>
		void VanillaSamplingLayer<T>::InterlockFinalized()
		{

		}

		template<class T>
		void VanillaSamplingLayer<T>::EnqueueForwardPropagation(OCLDevice* device, int queueIndex,
			OCLMemory* previousInput, OCLMemory* output, bool blocking)
		{

		}

		template<class T>
		void VanillaSamplingLayer<T>::EnqueueBackPropagation(OCLDevice* device, int queueIndex,
			OCLMemory* previousInput, OCLMemory* delta,
			OCLMemory* deltaOutput, bool blocking)
		{

		}

		template<class T>
		vector<OCLMemory*> VanillaSamplingLayer<T>::GetParameters()
		{
			return vector<OCLMemory*>();
		}

		template<class T>
		void VanillaSamplingLayer<T>::GetParameters(T*, OCLDevice*,
			int, bool)
		{

		}

		template<class T>
		void VanillaSamplingLayer<T>::SetParameters(T*, OCLDevice*,
			int, bool)
		{

		}

		template<class T>
		void VanillaSamplingLayer<T>::EnqueueCalculateGradient(OCLDevice*, int,
			OCLMemory*, OCLMemory*, vector<OCLMemory*>, bool)
		{

		}

		template <class T>
		vector<size_t> VanillaSamplingLayer<T>::GetMultipleParameterCount()
		{
			return vector<size_t>();
		}

		template<class T>
		size_t VanillaSamplingLayer<T>::GetParameterCount()
		{
			return 0;
		}

		template class VanillaSamplingLayer<cl_float>;
		template class VanillaSamplingLayer<cl_double>;

	} /* namespace MachineLearning */
} /* namespace Matuna */
