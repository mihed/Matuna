/*
* MaxPoolingLayer.cpp
*
*  Created on: Jun 23, 2015
*      Author: Mikael
*/

#include "MaxPoolingLayer.h"

#include "CheckPrecision.h"

#include "Matuna.Helper/Path.h"
#include "Matuna.Helper/FileHelper.h"
#include "Matuna.Helper/Converter.h"

namespace Matuna
{
	namespace MachineLearning
	{

		template<class T>
		MaxPoolingLayer<T>::MaxPoolingLayer(shared_ptr<OCLContext> context,
			const vector<LayerDataDescription>& inputLayerDescriptions,
			MatunaActivationFunction backPropActivation,
			const MaxPoolingLayerConfig* config)
			:
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

			InitializeMemoryDescriptions(inputLayerDescriptions, config);
		}

		template<class T>
		MaxPoolingLayer<T>::~MaxPoolingLayer()
		{

		}

		template<class T>
		void MaxPoolingLayer<T>::InitializeMemoryDescriptions(const vector<LayerDataDescription>& inputLayerDescriptions, const MaxPoolingLayerConfig* config)
		{

		}

		template<class T>
		MaxPoolingLayerConfig MaxPoolingLayer<T>::GetConfig() const
		{
			return config;
		}

		template<class T>
		void MaxPoolingLayer<T>::InterlockFinalized()
		{

		}

		template<class T>
		void MaxPoolingLayer<T>::InitializePrograms()
		{

		}

		template<class T>
		void MaxPoolingLayer<T>::EnqueueForwardPropagation(OCLDevice* device, int queueIndex,
			OCLMemory* previousInput, OCLMemory* output, bool blocking)
		{

		}

		template<class T>
		void MaxPoolingLayer<T>::EnqueueBackPropagation(OCLDevice* device, int queueIndex,
			OCLMemory* previousInput, OCLMemory* delta,
			OCLMemory* deltaOutput, bool blocking) 
		{

		}

		template<class T>
		vector<OCLMemory*> MaxPoolingLayer<T>::GetParameters()
		{
			return vector<OCLMemory*>();
		}

		template<class T>
		void MaxPoolingLayer<T>::GetParameters(T* parameters, OCLDevice* device,
			int queueIndex, bool blocking)
		{

		}

		template<class T>
		void MaxPoolingLayer<T>::SetParameters(T* parameters, OCLDevice* device,
			int queueIndex, bool blocking)
		{

		}

		template<class T>
		void MaxPoolingLayer<T>::EnqueueCalculateGradient(OCLDevice* device, int queueIndex,
			OCLMemory* previousInput, OCLMemory* delta, vector<OCLMemory*> gradient, bool blocking)
		{

		}

		template<class T>
		vector<size_t> MaxPoolingLayer<T>::GetMultipleParameterCount()
		{
			return vector<size_t>();
		}

		template<class T>
		size_t MaxPoolingLayer<T>::GetParameterCount()
		{
			return 0;
		}

		template class MaxPoolingLayer<cl_float>;
		template class MaxPoolingLayer<cl_double>;

	} /* namespace MachineLearning */
} /* namespace Matuna */
