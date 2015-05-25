/*
 * ConvolutionLayer.cpp
 *
 *  Created on: May 5, 2015
 *      Author: Mikael
 */

#include "ConvolutionLayer.h"
#include <stdexcept>
#include <type_traits>
#include <random>

namespace ATML {
	namespace MachineLearning {

		template<class T>
		ConvolutionLayer<T>::ConvolutionLayer(shared_ptr<OpenCLContext> context,
			const vector<LayerDataDescription>& inputLayerDescriptions,
			ATMLActivationFunction backPropActivation,
			const ConvolutionLayerConfig* config) :
			OpenCLForwardBackPropLayer<T>(context, inputLayerDescriptions,
			backPropActivation, config), convolutionConfig(*config)
		{

			if (inputLayerDescriptions.size() == 0)
				throw invalid_argument(
				"There's no input data descriptions for the convolution layer.");

			if (inputLayerDescriptions.size() != 1)
				throw runtime_error("Not implemented exception");

			if (config->ConnectionType() != ATMLFullConnection)
				throw runtime_error("Not implemented exception");

			//FIXME: Since we are not using any optimization at the moment, such as local memory.
			//We will not require any padding on the input and output proposals
			for (auto& layerDescription : inputLayerDescriptions)
			{
				LayerMemoryDescription inForwardMemProp;
				inForwardMemProp.Height = layerDescription.Height;
				inForwardMemProp.Width = layerDescription.Width;
				inForwardMemProp.Units = layerDescription.Units;
				inForwardMemProp.HeightOffset = 0;
				inForwardMemProp.UnitOffset = 0;
				inForwardMemProp.WidthOffset = 0;

				this->inForwardPropMemoryProposals.push_back(inForwardMemProp);
				this->outBackPropMemoryProposals.push_back(inForwardMemProp);

				LayerDataDescription outForwardDataDesc;
				outForwardDataDesc.Height = layerDescription.Height - config->FilterHeight() + 1;
				outForwardDataDesc.Width = layerDescription.Width - config->FilterWidth() + 1;
				outForwardDataDesc.Units = config->FilterCount();
				this->outForwardPropDataDescriptions.push_back(outForwardDataDesc);

				LayerMemoryDescription outForwardMemProp;
				outForwardMemProp.Height = layerDescription.Height - config->FilterHeight() + 1;
				outForwardMemProp.Width = layerDescription.Width - config->FilterWidth() + 1;
				outForwardMemProp.Units = config->FilterCount();
				outForwardMemProp.HeightOffset = 0;
				outForwardMemProp.UnitOffset = 0;
				outForwardMemProp.WidthOffset = 0;

				this->outForwardPropMemoryProposals.push_back(outForwardMemProp);


				//Since we will add padding to the input, we will require that we have a border of size filterdimension - 1
				LayerMemoryDescription inBackMemProp;
				inBackMemProp.Height = layerDescription.Height + 2 * (config->FilterHeight() - 1);
				inBackMemProp.Width = layerDescription.Width + 2 * (config->FilterWidth() - 1);
				inBackMemProp.Units = config->FilterCount();
				inBackMemProp.HeightOffset = config->FilterHeight() - 1;
				inBackMemProp.WidthOffset = config->FilterWidth() - 1;
				inBackMemProp.UnitOffset = 0;

				this->inBackPropMemoryProposals.push_back(inBackMemProp);
			}

			this->inBackPropDataDescriptions = this->outForwardPropDataDescriptions;

		}

		template<class T>
		ConvolutionLayer<T>::~ConvolutionLayer()
		{

		}

		template<class T>
		ConvolutionLayerConfig ConvolutionLayer<T>::GetConfig() const
		{
			return convolutionConfig;
		}

		template<class T>
		void ConvolutionLayer<T>::InitializeParameters()
		{

			//TODO: There are optimization to be made here if the network is read-only. (i.e. non trainable)
			auto filterElementCount = convolutionConfig.FilterCount() * convolutionConfig.FilterWidth() * convolutionConfig.FilterHeight();
			auto biasElementCount = convolutionConfig.FilterCount();
			filters = move(this->context->CreateMemory(CL_MEM_READ_WRITE, sizeof(T) * filterElementCount));

			biases = move(this->context->CreateMemory(CL_MEM_READ_WRITE, sizeof(T) * biasElementCount));

			random_device tempDevice;
			mt19937 mt(tempDevice());

			//TODO: The initial weight values could be something to tweak
			uniform_real_distribution<T> uniformDistribution(-0.1, 0.1);

			vector<T> initialWeightValues;
			initialWeightValues.resize(filterElementCount);
			for (int i = 0; i < filterElementCount; i++)
				initialWeightValues[i] = uniformDistribution(mt);

			vector<T> initialBiasValues;
			initialBiasValues.resize(biasElementCount);
			for (int i = 0; i < biasElementCount; i++)
				initialBiasValues[i] = uniformDistribution(mt);

			//Since this is initialization, we don't really care about which device and device queue we are using
			OpenCLDevice* device = this->context->GetDevices()[0];
			device->WriteMemory(filters.get(), sizeof(T) * initialWeightValues.size(),
				initialWeightValues.data(), 0, false);
			device->WriteMemory(biases.get(), sizeof(T) * initialBiasValues.size(),
				initialBiasValues.data(), 0, false);
			device->WaitForDeviceQueue(0);
		}

		template<class T>
		void ConvolutionLayer<T>::InterlockFinalized()
		{

		}

		template<class T>
		void ConvolutionLayer<T>::EnqueueForwardPropagation(OpenCLDevice* device,
			int queueIndex, OpenCLMemory* previousInput, OpenCLMemory* output,
			bool blocking)
		{

		}

		template<class T>
		void ConvolutionLayer<T>::EnqueueBackPropagation(OpenCLDevice* device,
			int queueIndex, OpenCLMemory* previousInput, OpenCLMemory* delta,
			OpenCLMemory* deltaOutput, bool blocking)
		{

		}

		template<class T>
		void ConvolutionLayer<T>::EnqueueCalculateGradient(OpenCLDevice* device,
			int queueIndex, OpenCLMemory* previousInput, OpenCLMemory* delta,
			OpenCLMemory* gradient, bool blocking)
		{

		}

		template<class T>
		vector<tuple<OpenCLMemory*, int>> ConvolutionLayer<T>::GetParameters()
		{
			vector<tuple<OpenCLMemory*, int> > result;

			auto filterTuple = make_tuple(filters.get(), convolutionConfig.FilterCount() *
				convolutionConfig.FilterWidth() * convolutionConfig.FilterHeight());
			auto biasTuple = make_tuple(biases.get(), convolutionConfig.FilterCount());
			result.push_back(filterTuple);
			result.push_back(biasTuple);

			return result;
		}

		template<class T>
		void ConvolutionLayer<T>::GetParameters(T* parameters, OpenCLDevice* device,
			int queueIndex, bool blocking)
		{
			device->ReadMemory(filters.get(), filters->ByteSize(), parameters,
				queueIndex, blocking);

			auto biasPosition = parameters + convolutionConfig.FilterCount() * convolutionConfig.FilterWidth() *
				convolutionConfig.FilterHeight();

			device->ReadMemory(biases.get(), biases->ByteSize(), biasPosition,
				queueIndex, blocking);
		}

		template<class T>
		void ConvolutionLayer<T>::SetParameters(T* parameters, OpenCLDevice* device,
			int queueIndex, bool blocking)
		{
			device->WriteMemory(filters.get(), filters->ByteSize(), parameters,
				queueIndex, blocking);

			auto biasPosition = parameters + convolutionConfig.FilterCount() * convolutionConfig.FilterWidth() *
				convolutionConfig.FilterHeight();

			device->WriteMemory(biases.get(), biases->ByteSize(), biasPosition,
				queueIndex, blocking);
		}

		template<class T>
		size_t ConvolutionLayer<T>::GetParameterCount()
		{
			return convolutionConfig.FilterCount() *
				convolutionConfig.FilterWidth() *
				convolutionConfig.FilterHeight() + convolutionConfig.FilterCount();
		}


		template class ConvolutionLayer < cl_float > ;
		template class ConvolutionLayer < cl_double > ;

	} /* namespace MachineLearning */
} /* namespace ATML */
