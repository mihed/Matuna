/*
 * CNNOpenCL.cpp
 *
 *  Created on: May 5, 2015
 *      Author: Mikael
 */

#include "CNNOpenCL.h"
#include "CNNOpenCLFactoryVisitor.h"
#include "OpenCLHelper/OpenCLHelper.h"

#include <CL/cl.h>
#include <stdexcept>
#include <tuple>
#include <unordered_map>

namespace ATML
{
	namespace MachineLearning
	{
		template class CNNOpenCL < cl_float > ;
		template class CNNOpenCL < cl_double > ;

		template<class T>
		CNNOpenCL<T>::CNNOpenCL(const vector<OpenCLDeviceInfo>& devices,
			unique_ptr<CNNConfig> config) :
			TrainableCNN<T>(*config)
		{

			InitializeContexts(devices);

			//TODO: At the moment we only support one context
			CNNOpenCLFactoryVisitor<T> factory(contexts[0], this);
			config->Accept(&factory);
			auto createdLayers = factory.GetLayers();
			auto createdOutputLayer = factory.GetOutputLayer();

			//Transfer ownership to this class
			for (auto& layer : createdLayers)
			{
				auto temp = dynamic_cast<OpenCLForwardBackPropLayer<T>*>(layer.get());
				if (!temp)
					throw runtime_error("The factory does not yield correct layers");
				layers.push_back(unique_ptr<OpenCLForwardBackPropLayer<T>>(temp));
				layer.release();
			}

			auto temp2 = dynamic_cast<StandardOutputLayer<T>*>(createdOutputLayer.get());
			outputLayer = unique_ptr<StandardOutputLayer<T>>(temp2);
			if (!temp2)
				throw runtime_error(
				"The factory does not yield correct layers. Every CNN must have an output layer");
			createdOutputLayer.release();
		}

		template<class T>
		CNNOpenCL<T>::~CNNOpenCL()
		{

		}

		template<class T>
		void CNNOpenCL<T>::InitializeContexts(
			const vector<OpenCLDeviceInfo>& deviceInfos)
		{

			unordered_map < cl_platform_id,
				tuple<OpenCLPlatformInfo, vector<OpenCLDeviceInfo>> > platformsAndDevices;
			for (auto& deviceInfo : deviceInfos)
			{
				if (!deviceInfo.DeviceAvailable())
					throw invalid_argument("The device is not available");

				if (!deviceInfo.CompilerAvailable())
					throw invalid_argument("The device does not have an available compiler");

				if (is_same<cl_double, T>::value)
				{
					if (deviceInfo.PreferredDoubleVectorWidth() == 0)
						throw invalid_argument(
						"The template argument is not supported on the chosen devices");
				}
				else if (is_same<cl_float, T>::value)
				{
					if (deviceInfo.PreferredFloatVectorWidth() == 0)
						throw invalid_argument(
						"The template argument is not supported on the chosen devices");
				}
				else
					throw runtime_error(
					"The template argument does not match the supported arguments");

				auto platformInfo = deviceInfo.PlatformInfo();
				auto platformID = platformInfo.PlatformID();
				if (platformsAndDevices.find(platformID) == platformsAndDevices.end())
				{
					vector<OpenCLDeviceInfo> infos;
					infos.push_back(deviceInfo);
					platformsAndDevices.insert(make_pair(platformID, make_tuple(platformInfo, infos)));
				}
				else
				{
					auto keyValuePair = platformsAndDevices.find(platformID);
					auto& platformAndDevice = keyValuePair->second;
					auto& infos = get<1>(platformAndDevice);
					infos.push_back(deviceInfo);
				}
			}

			//This network uses two command queues per device
			OpenCLDeviceConfig config;
			config.AddCommandQueue();
			config.AddCommandQueue();

			vector<tuple<OpenCLPlatformInfo, vector<tuple<OpenCLDeviceConfig, OpenCLDeviceInfo>>>> contextsConfigurations;
			for (auto& platformAndDevices : platformsAndDevices)
			{
				vector<tuple<OpenCLDeviceConfig, OpenCLDeviceInfo>> contextConfig;
				auto& infoDeviceTuple = platformAndDevices.second;
				for (auto& deviceInfo : get<1>(infoDeviceTuple))
					contextConfig.push_back(make_tuple(config, deviceInfo));

				contextsConfigurations.push_back(make_tuple(get<0>(infoDeviceTuple), contextConfig));
			}

			for (auto& contextConfig : contextsConfigurations)
				contexts.push_back(shared_ptr<OpenCLContext>(move(OpenCLHelper::GetContext(get<0>(contextConfig), get<1>(contextConfig)))));
		}

		template<class T>
		vector<OpenCLContext*> CNNOpenCL<T>::GetOpenCLContexts() const
		{
			vector<OpenCLContext*> result;
			for (auto& context : contexts)
				result.push_back(context.get());

			return result;
		}

		template<class T>
		unique_ptr<T[]> CNNOpenCL<T>::FeedForwardAligned(T* input, int formatIndex)
		{
			//TODO: At the moment we only support one context and one device 
			auto devices = contexts[0]->GetDevices();
			auto device = devices[0];
			LayerMemoryDescription inputMemoryDescription =
				this->InputForwardMemoryDescriptions()[formatIndex];

			unique_ptr<OpenCLMemory> inputMemory = contexts[0]->CreateMemory(CL_MEM_READ_ONLY, sizeof(T) * inputMemoryDescription.TotalMemory());
			device->WriteMemory(inputMemory.get(), inputMemory->ByteSize(), input, 0, true);

			//We need some synchronization in order not to use blocking calls. 
			//We could probably yield some better performance in that case.
			for (auto& layer : layers)
			{
				inputMemoryDescription = layer->OutForwardPropMemoryDescriptions()[formatIndex];
				unique_ptr<OpenCLMemory> outputMemory = contexts[0]->CreateMemory(CL_MEM_READ_WRITE, sizeof(T) * inputMemoryDescription.TotalMemory());
				layer->EnqueueForwardPropagation(device, 0, inputMemory.get(), outputMemory.get(), true);
				inputMemory.reset();
				inputMemory = move(outputMemory);
			}

			unique_ptr<T[]> output(new T[inputMemoryDescription.TotalMemory()]);
			device->ReadMemory(inputMemory.get(), inputMemory->ByteSize(), output.get(), 0, true);

			return move(output);
		}

		template<class T>
		unique_ptr<T[]> CNNOpenCL<T>::BackPropAligned(T* input,
			int formatIndex, T* target)
		{
			//TODO: At the moment we only support one context and one device 
			auto devices = contexts[0]->GetDevices();
			auto device = devices[0];


			vector<unique_ptr<OpenCLMemory>> inputMemories;
			//First the forward propagation phase, we need to save all the memory.
			LayerMemoryDescription inputMemoryDescription =
				this->InputForwardMemoryDescriptions()[formatIndex];

			unique_ptr<OpenCLMemory> inputMemory = contexts[0]->CreateMemory(CL_MEM_READ_ONLY, sizeof(T) * inputMemoryDescription.TotalMemory());
			device->WriteMemory(inputMemory.get(), inputMemory->ByteSize(), input, 0, false);

			inputMemories.push_back(move(inputMemory));
			for (auto& layer : layers)
			{
				inputMemoryDescription = layer->OutForwardPropMemoryDescriptions()[formatIndex];
				unique_ptr<OpenCLMemory> outputMemory = contexts[0]->CreateMemory(
					CL_MEM_READ_WRITE, sizeof(T) * inputMemoryDescription.TotalMemory());
				inputMemories.push_back(move(outputMemory));
			}

			int count = layers.size();

			for (int i = 1; i < count; i++)
				layers[i - 1]->EnqueueForwardPropagation(device, 0, inputMemories[i - 1].get(), inputMemories[i].get(), false);

			device->WaitForDeviceQueue(0);

			LayerMemoryDescription inBackPropMemoryDescription = this->OutputForwardMemoryDescriptions()[formatIndex];

			unique_ptr<OpenCLMemory> targetMemory = contexts[0]->CreateMemory(CL_MEM_READ_ONLY, sizeof(T) * inBackPropMemoryDescription.TotalMemory());
			device->WriteMemory(targetMemory.get(), targetMemory->ByteSize(), target, 0, true);

			unique_ptr<OpenCLMemory> backPropOutputMemory = contexts[0]->CreateMemory(CL_MEM_READ_ONLY, sizeof(T) * inBackPropMemoryDescription.TotalMemory());;
			outputLayer->EnqueueBackPropagation(device, 0, inputMemories[inputMemories.size() - 1].get(), targetMemory.get(), backPropOutputMemory.get(), true);
			targetMemory.reset();

			for (int i = count - 1; i >= 0; i--)
			{
				inBackPropMemoryDescription = layers[i]->OutBackPropMemoryDescriptions()[formatIndex];
				unique_ptr<OpenCLMemory> outputMemory = contexts[0]->CreateMemory(
					CL_MEM_READ_WRITE, sizeof(T) * inBackPropMemoryDescription.TotalMemory());
				layers[i]->EnqueueBackPropagation(device, 0, inputMemories[i].get(), backPropOutputMemory.get(), outputMemory.get(), true);
				backPropOutputMemory.reset();
				backPropOutputMemory = move(outputMemory);
			}

			device->WaitForDeviceQueue(0);

			LayerMemoryDescription outBackPropMemoryDescription = this->OutputBackMemoryDescriptions()[formatIndex];

			unique_ptr<T[]> output(new T[outBackPropMemoryDescription.TotalMemory()]);
			device->ReadMemory(backPropOutputMemory.get(), backPropOutputMemory->ByteSize(), output.get(), 0, true);

			return move(output);
		}

		template<class T>
		T CNNOpenCL<T>::CalculateErrorAligned(T* propagatedValue, int formatIndex,
			T* target)
		{
			return T();
		}

		template<class T>
		unique_ptr<T[]> CNNOpenCL<T>::CalculateGradientAligned(T* input,
			int formatIndex, T* target)
		{
			return nullptr;
		}

		template<class T>
		unique_ptr<T[]> CNNOpenCL<T>::GetParameters()
		{
			//TODO: At the moment we only support one context and one device 
			auto devices = contexts[0]->GetDevices();
			auto device = devices[0];
			unique_ptr<T[]> parameters(new T[GetParameterCount()]);
			auto pointerPosition = parameters.get();
			for (auto& layer : layers)
			{
				layer->GetParameters(pointerPosition, device, 0, false);
				pointerPosition += layer->GetParameterCount();
			}

			device->WaitForDeviceQueue(0);

			return move(parameters);
		}

		template<class T>
		void CNNOpenCL<T>::SetParameters(T* parameters)
		{
			//TODO: At the moment we only support one context and one device 
			auto devices = contexts[0]->GetDevices();
			auto device = devices[0];
			auto pointerPosition = parameters;
			for (auto& layer : layers)
			{
				layer->SetParameters(pointerPosition, device, 0, false);
				pointerPosition += layer->GetParameterCount();
			}

			device->WaitForDeviceQueue(0);
		}

		template<class T>
		size_t CNNOpenCL<T>::GetParameterCount()
		{
			size_t result = 0;
			for (auto& layer : layers)
				result += layer->GetParameterCount();

			return result;
		}

		template<class T>
		void CNNOpenCL<T>::TrainNetwork(unique_ptr<CNNTrainer<T>> trainer,
			unique_ptr<IAlgorithmConfig> algorithm)
		{

		}

		template<class T>
		vector<OpenCLForwardBackPropLayer<T>*> CNNOpenCL<T>::GetLayers() const
		{
			vector<OpenCLForwardBackPropLayer<T>*> result;
			for (auto& layer : layers)
				result.push_back(layer.get());

			return result;
		}

		template<class T>
		StandardOutputLayer<T>* CNNOpenCL<T>::GetOutputLayer() const
		{
			return outputLayer.get();
		}

	} /* namespace MachineLearning */
} /* namespace ATML */
