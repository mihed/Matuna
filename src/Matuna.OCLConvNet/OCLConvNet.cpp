/*
* OCLConvNet.cpp
*
*  Created on: May 5, 2015
*      Author: Mikael
*/

#include "OCLConvNet.h"
#include "OCLConvNetFactoryVisitor.h"
#include "CheckPrecision.h"
#include "Matuna.OCLHelper/OCLHelper.h"
#include "Matuna.ConvNet/GradientDescentConfig.h"
#include "Matuna.Helper/FileHelper.h"
#include "Matuna.Helper/Path.h"
#include "Matuna.Helper/Converter.h"

#include "LayerKernel.h"

#include <functional>

namespace Matuna 
{
	namespace MachineLearning 
	{
		template<class T>
		OCLConvNet<T>::OCLConvNet(const vector<OCLDeviceInfo>& devices,
			unique_ptr<ConvNetConfig> config) :
		TrainableConvNet<T>(*config) 
		{

			InitializeContexts(devices);

			//TODO: At the moment we only support one context
			OCLConvNetFactoryVisitor<T> factory(contexts[0], this);
			config->Accept(&factory);
			auto createdLayers = factory.GetLayers();
			auto createdOutputLayer = factory.GetOutputLayer();

			lowMemoryUsage = config->HasLowMemoryUsage();

			//Transfer ownership to this class
			for (auto& layer : createdLayers) 
			{
				auto temp = dynamic_cast<OCLForwardBackPropLayer<T>*>(layer.get());
				if (!temp)
					throw runtime_error("The factory does not yield correct layers");
				layers.push_back(unique_ptr<OCLForwardBackPropLayer<T>>(temp));
				layer.release();
			}

			auto temp2 = dynamic_cast<StandardOutputLayer<T>*>(createdOutputLayer.get());
			outputLayer = unique_ptr<StandardOutputLayer<T>>(temp2);
			if (!temp2)
				throw runtime_error(
				"The factory does not yield correct layers. Every ConvNet must have an output layer");
			createdOutputLayer.release();
		}

		template<class T>
		OCLConvNet<T>::~OCLConvNet() 
		{
			layers.clear();
			outputLayer.reset();
			contexts.clear();
		}


		//TODO: the context index is bloody ugly and should be changed later
		template<class T>
		unique_ptr<OCLMemory> OCLConvNet<T>::CreateInputMemory(T* input, int formatIndex, int contextIndex) const
		{
			//TODO: At the moment we only support one context and one device

			LayerMemoryDescription inputMemoryDescription =
				this->InputForwardMemoryDescriptions()[formatIndex];
			auto devices = contexts[contextIndex]->GetDevices();
			auto device = devices[contextIndex];
			unique_ptr<OCLMemory> inputMemory = contexts[contextIndex]->CreateMemory(
				CL_MEM_READ_ONLY, sizeof(T) * inputMemoryDescription.TotalMemory());
			device->WriteMemory(inputMemory.get(), inputMemory->ByteSize(), input, 0, true);

			return move(inputMemory);
		}

		//TODO: the context index is bloody ugly and should be changed later
		template<class T>
		unique_ptr<OCLMemory> OCLConvNet<T>::CreateTargetMemory(T* target, int formatIndex, int contextIndex) const
		{
			//TODO: At the moment we only support one context and one device

			LayerMemoryDescription inBackPropMemoryDescription =
				this->OutputForwardMemoryDescriptions()[formatIndex];
			auto devices = contexts[contextIndex]->GetDevices();
			auto device = devices[contextIndex];
			unique_ptr<OCLMemory> targetMemory = contexts[contextIndex]->CreateMemory(CL_MEM_READ_ONLY,
				sizeof(T) * inBackPropMemoryDescription.TotalMemory());
			device->WriteMemory(targetMemory.get(), targetMemory->ByteSize(), target, 0, true);

			return move(targetMemory);
		}

		template<class T>
		void OCLConvNet<T>::InitializeContexts(
			const vector<OCLDeviceInfo>& deviceInfos) 
		{
			static_assert(is_same<cl_double, T>::value || is_same<cl_float, T>::value, "The type is not supported");

			unordered_map<cl_platform_id, tuple<OCLPlatformInfo, vector<OCLDeviceInfo>> > platformsAndDevices;

			for (auto& deviceInfo : deviceInfos) 
			{
				if (!deviceInfo.DeviceAvailable())
					throw invalid_argument("The device is not available");

				if (!deviceInfo.CompilerAvailable())
					throw invalid_argument(
					"The device does not have an available compiler");

				CheckPrecision<is_same<cl_double, T>::value>::Check(deviceInfo);

				auto platformInfo = deviceInfo.PlatformInfo();
				auto platformID = platformInfo.PlatformID();
				if (platformsAndDevices.find(platformID) == platformsAndDevices.end()) 
				{
					vector<OCLDeviceInfo> infos;
					infos.push_back(deviceInfo);
					platformsAndDevices.insert(
						make_pair(platformID, make_tuple(platformInfo, infos)));
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
			OCLDeviceConfig config;
			config.AddCommandQueue();
			config.AddCommandQueue();

			vector<tuple<OCLPlatformInfo, vector<tuple<OCLDeviceConfig, OCLDeviceInfo>>>> contextsConfigurations;
			for (auto& platformAndDevices : platformsAndDevices) 
			{
				vector<tuple<OCLDeviceConfig, OCLDeviceInfo>> contextConfig;
				auto& infoDeviceTuple = platformAndDevices.second;
				for (auto& deviceInfo : get<1>(infoDeviceTuple))
					contextConfig.push_back(make_tuple(config, deviceInfo));

				contextsConfigurations.push_back(
					make_tuple(get<0>(infoDeviceTuple), contextConfig));
			}

			for (auto& contextConfig : contextsConfigurations)
				contexts.push_back(
				shared_ptr<OCLContext>(
				move(
				OCLHelper::GetContext(get<0>(contextConfig),
				get<1>(contextConfig)))));
		}

		template<class T>
		vector<OCLContext*> OCLConvNet<T>::GetOCLContexts() const 
		{
			vector<OCLContext*> result;
			for (auto& context : contexts)
				result.push_back(context.get());

			return result;
		}

		template<class T>
		unique_ptr<OCLMemory> OCLConvNet<T>::FeedForwardLowMemoryOCLOutput(OCLMemory* input, int formatIndex)
		{
			auto devices = contexts[0]->GetDevices();
			auto device = devices[0];
			LayerMemoryDescription inputMemoryDescription = this->InputForwardMemoryDescriptions()[formatIndex];

			unique_ptr<OCLMemory> outputMemory;
			if (layers.size() != 0)
			{
				auto& layer = layers[0];
				inputMemoryDescription = layer->OutForwardPropMemoryDescriptions()[formatIndex];
				outputMemory = contexts[0]->CreateMemory(CL_MEM_READ_WRITE, sizeof(T) * inputMemoryDescription.TotalMemory());
				layer->EnqueueForwardPropagation(device, 0, input, outputMemory.get(), true);
			}

			//Any ideas how this could be done without blocking calls?
			unique_ptr<OCLMemory> inputMemory = move(outputMemory);
			for (size_t i = 1; i < layers.size(); i++) 
			{
				auto& layer = layers[i];
				inputMemoryDescription = layer->OutForwardPropMemoryDescriptions()[formatIndex];
				outputMemory = contexts[0]->CreateMemory(CL_MEM_READ_WRITE, sizeof(T) * inputMemoryDescription.TotalMemory());
				layer->EnqueueForwardPropagation(device, 0, inputMemory.get(), outputMemory.get(), true);

				//Input to previous enqueue call can be cleaned here safely and still have a full device queue

				inputMemory.reset();
				inputMemory = move(outputMemory);
			}

			return move(inputMemory);
		}

		template<class T>
		unique_ptr<T[]> OCLConvNet<T>::FeedForwardLowMemory(OCLMemory* input, int formatIndex)
		{
			//TODO: We should probably think a bit more how to handle the devices
			auto devices = contexts[0]->GetDevices();
			auto device = devices[0];
			LayerMemoryDescription outputMemoryDescription = this->OutputForwardMemoryDescriptions()[formatIndex];
			auto outputMemory = FeedForwardLowMemoryOCLOutput(input, formatIndex);
			unique_ptr<T[]> output(new T[outputMemoryDescription.TotalMemory()]);
			device->ReadMemory(outputMemory.get(), outputMemory->ByteSize(), output.get(), 0, true);
			return move(output);
		}


		template<class T>
		void OCLConvNet<T>::FeedForwardHighMemory(OCLMemory* input, int formatIndex, vector<unique_ptr<OCLMemory>>& outputMemories)
		{
			//TODO: We should probably think a bit more how to handle the devices
			auto devices = contexts[0]->GetDevices();
			auto device = devices[0];
			LayerMemoryDescription inputMemoryDescription = this->InputForwardMemoryDescriptions()[formatIndex];
			for (auto& layer : layers) 
			{
				inputMemoryDescription = layer->OutForwardPropMemoryDescriptions()[formatIndex];
				unique_ptr<OCLMemory> outputMemory = contexts[0]->CreateMemory(CL_MEM_READ_WRITE, sizeof(T) * inputMemoryDescription.TotalMemory());
				layer->EnqueueForwardPropagation(device, 0, input, outputMemory.get(), false);
				input = outputMemory.get();
				outputMemories.push_back(move(outputMemory));
			}
		}

		template<class T>
		unique_ptr<T[]> OCLConvNet<T>::FeedForwardHighMemory(OCLMemory* input, int formatIndex)
		{
			//TODO: We should probably think a bit more how to handle the devices
			auto devices = contexts[0]->GetDevices();
			auto device = devices[0];
			vector<unique_ptr<OCLMemory>> outputMemories;
			FeedForwardHighMemory(input, formatIndex, outputMemories);
			auto& outputMemory = outputMemories[outputMemories.size() - 1];
			LayerMemoryDescription outputMemoryDescription = this->OutputForwardMemoryDescriptions()[formatIndex];
			unique_ptr<T[]> output(new T[outputMemoryDescription.TotalMemory()]);
			device->ReadMemory(outputMemory.get(), outputMemory->ByteSize(), output.get(), 0, true);
			return move(output);
		}

		template<class T>
		unique_ptr<T[]> OCLConvNet<T>::FeedForwardAligned(OCLMemory* input, int formatIndex)
		{
			if (lowMemoryUsage)
				return FeedForwardLowMemory(input, formatIndex);
			else
				return FeedForwardHighMemory(input, formatIndex);
		}

		template<class T>
		unique_ptr<T[]> OCLConvNet<T>::FeedForwardAligned(T* input, int formatIndex)
		{
			unique_ptr<OCLMemory> inputMemory = CreateInputMemory(input, formatIndex, 0);
			if (lowMemoryUsage)
				return FeedForwardLowMemory(inputMemory.get(), formatIndex);
			else
				return FeedForwardHighMemory(inputMemory.get(), formatIndex);
		}

		template<class T>
		T OCLConvNet<T>::CalculateError(OCLMemory* lastOutput, int formatIndex, OCLMemory* target)
		{
			//TODO: Need a better way of handling the context and devices
			auto devices = contexts[0]->GetDevices();
			auto device = devices[0];
			LayerMemoryDescription inBackPropMemoryDescription = this->OutputForwardMemoryDescriptions()[formatIndex];
			return this->outputLayer->CalculateError(device, 0, lastOutput, target);
		}

		template<class T>
		T OCLConvNet<T>::CalculateErrorAligned(OCLMemory* input, int formatIndex, OCLMemory* target)
		{
			if(lowMemoryUsage)
			{
				auto outputMemory = FeedForwardLowMemoryOCLOutput(input, formatIndex);
				return CalculateError(outputMemory.get(), formatIndex, target);
			}
			else
			{
				vector<unique_ptr<OCLMemory>> outputMemories;
				FeedForwardHighMemory(input, formatIndex, outputMemories);
				return CalculateError(outputMemories[outputMemories.size() - 1].get(), formatIndex, target);
			}
		}

		template<class T>
		T OCLConvNet<T>::CalculateErrorAligned(T* input, int formatIndex, T* target) 
		{
			auto inputMemory = CreateInputMemory(input, formatIndex, 0);
			auto targetMemory = CreateTargetMemory(target, formatIndex, 0);
			if(lowMemoryUsage)
			{
				auto outputMemory = FeedForwardLowMemoryOCLOutput(inputMemory.get(), formatIndex);
				return CalculateError(outputMemory.get(), formatIndex, targetMemory.get());
			}
			else
			{
				vector<unique_ptr<OCLMemory>> outputMemories;
				FeedForwardHighMemory(inputMemory.get(), formatIndex, outputMemories);
				return CalculateError(outputMemories[outputMemories.size() - 1].get(), formatIndex, targetMemory.get());
			}
		}


		//TODO: Fix this implementation
		template<class T>
		T OCLConvNet<T>::CalculateErrorFromForwardAligned(T* propagatedValue,
			int formatIndex, T* target)
		{
			//TODO: At the moment we only support one context and one device
			auto devices = contexts[0]->GetDevices();
			auto device = devices[0];

			LayerMemoryDescription outputDescription =
				this->OutputForwardMemoryDescriptions()[formatIndex];
			unique_ptr<OCLMemory> targetMemory = contexts[0]->CreateMemory(
				CL_MEM_READ_ONLY, sizeof(T) * outputDescription.TotalMemory());
			device->WriteMemory(targetMemory.get(), targetMemory->ByteSize(), target, 0,
				true);
			unique_ptr<OCLMemory> propagatedMemory = contexts[0]->CreateMemory(
				CL_MEM_READ_ONLY, sizeof(T) * outputDescription.TotalMemory());
			device->WriteMemory(propagatedMemory.get(), propagatedMemory->ByteSize(),
				propagatedValue, 0, true);

			return outputLayer->CalculateError(device, 0, propagatedMemory.get(),
				targetMemory.get());
		}

		template<class T>
		unique_ptr<T[]> OCLConvNet<T>::BackPropLowMemory(OCLMemory* input, int formatIndex, OCLMemory* target)
		{
			//TODO: At the moment we only support one context and one device
			auto devices = contexts[0]->GetDevices();
			auto device = devices[0];
			vector<unique_ptr<OCLMemory>> inputMemoriesHolder;
			FeedForwardHighMemory(input, formatIndex, inputMemoriesHolder);
			vector<OCLMemory*> inputMemories;
			inputMemories.push_back(input);
			for (auto& holder : inputMemoriesHolder)
				inputMemories.push_back(holder.get());

			LayerMemoryDescription inBackPropMemoryDescription = outputLayer->OutBackPropMemoryDescriptions()[formatIndex];
			unique_ptr<OCLMemory> backPropOutputMemory = contexts[0]->CreateMemory(CL_MEM_READ_ONLY, sizeof(T) * inBackPropMemoryDescription.TotalMemory());

			//TODO: Clear the input memories that we don't use
			outputLayer->EnqueueBackPropagation(device, 0, inputMemories[inputMemories.size() - 1], target, backPropOutputMemory.get(), true);
			inputMemoriesHolder[inputMemoriesHolder.size() - 1].reset();

			for (int i = static_cast<int>(layers.size()) - 1; i >= 1; i--) 
			{
				inBackPropMemoryDescription = layers[i]->OutBackPropMemoryDescriptions()[formatIndex];
				unique_ptr<OCLMemory> outputMemory = contexts[0]->CreateMemory(CL_MEM_READ_WRITE, sizeof(T) * inBackPropMemoryDescription.TotalMemory());
				layers[i]->EnqueueBackPropagation(device, 0, inputMemories[i], backPropOutputMemory.get(), outputMemory.get(), true);
				inputMemoriesHolder[i - 1].reset();
				backPropOutputMemory.reset();
				backPropOutputMemory = move(outputMemory);
			}

			LayerMemoryDescription outBackPropMemoryDescription = this->OutputBackMemoryDescriptions()[formatIndex];

			unique_ptr<T[]> output(new T[outBackPropMemoryDescription.TotalMemory()]);
			device->ReadMemory(backPropOutputMemory.get(), backPropOutputMemory->ByteSize(), output.get(), 0, true);

			return move(output);
		}

		template<class T>
		unique_ptr<T[]> OCLConvNet<T>::BackPropHighMemory(OCLMemory* input, int formatIndex, OCLMemory* target)
		{
			//TODO: At the moment we only support one context and one device
			auto devices = contexts[0]->GetDevices();
			auto device = devices[0];
			vector<unique_ptr<OCLMemory>> inputMemoriesHolder;
			FeedForwardHighMemory(input, formatIndex, inputMemoriesHolder);
			vector<OCLMemory*> inputMemories;
			inputMemories.push_back(input);
			for (auto& holder : inputMemoriesHolder)
				inputMemories.push_back(holder.get());

			vector<unique_ptr<OCLMemory>> outputMemoryHolder;
			OCLMemory* outputMemory1;
			OCLMemory* outputMemory2;

			LayerMemoryDescription inBackPropMemoryDescription = outputLayer->OutBackPropMemoryDescriptions()[formatIndex];
			unique_ptr<OCLMemory> backPropOutputMemory = contexts[0]->CreateMemory(CL_MEM_READ_ONLY, sizeof(T) * inBackPropMemoryDescription.TotalMemory());

			outputMemory1 =	backPropOutputMemory.get();
			outputMemoryHolder.push_back(move(backPropOutputMemory));

			//TODO: Clear the input memories that we don't use
			outputLayer->EnqueueBackPropagation(device, 0, inputMemories[inputMemories.size() - 1], target, outputMemory1, false);

			for (int i = static_cast<int>(layers.size()) - 1; i >= 1; i--) 
			{
				inBackPropMemoryDescription = layers[i]->OutBackPropMemoryDescriptions()[formatIndex];
				unique_ptr<OCLMemory> outputMemory = contexts[0]->CreateMemory(CL_MEM_READ_WRITE, sizeof(T) * inBackPropMemoryDescription.TotalMemory());
				outputMemory2 = outputMemory.get();
				outputMemoryHolder.push_back(move(outputMemory));
				layers[i]->EnqueueBackPropagation(device, 0, inputMemories[i], outputMemory1, outputMemory2, false);
				outputMemory1 = outputMemory2;
			}

			LayerMemoryDescription outBackPropMemoryDescription = this->OutputBackMemoryDescriptions()[formatIndex];

			unique_ptr<T[]> output(new T[outBackPropMemoryDescription.TotalMemory()]);
			device->ReadMemory(outputMemory1, outputMemory1->ByteSize(), output.get(), 0, true);

			return move(output);
		}

		template<class T>
		unique_ptr<T[]> OCLConvNet<T>::BackPropAligned(OCLMemory* input, int formatIndex, OCLMemory* target)
		{
			if (lowMemoryUsage)
				return move(BackPropLowMemory(input, formatIndex, target));
			else
				return move(BackPropHighMemory(input, formatIndex, target));
		}

		template<class T>
		unique_ptr<T[]> OCLConvNet<T>::BackPropAligned(T* input, int formatIndex, T* target) 
		{

			auto targetMemory = CreateTargetMemory(target, formatIndex, 0);
			auto inputMemory = CreateInputMemory(input, formatIndex, 0);

			if (lowMemoryUsage)
				return move(BackPropLowMemory(inputMemory.get(), formatIndex, targetMemory.get()));
			else
				return move(BackPropHighMemory(inputMemory.get(), formatIndex, targetMemory.get()));
		}

		template<class T>
		unique_ptr<T[]> OCLConvNet<T>::CalculateGradientHighMemory(OCLMemory* input, int formatIndex, OCLMemory* target)
		{
			auto devices = contexts[0]->GetDevices();
			auto device = devices[0];
			vector<unique_ptr<OCLMemory>> inputMemoriesHolder;
			FeedForwardHighMemory(input, formatIndex, inputMemoriesHolder);
			vector<OCLMemory*> inputMemories;
			inputMemories.push_back(input);
			for (auto& holder : inputMemoriesHolder)
				inputMemories.push_back(holder.get());

			LayerMemoryDescription inBackPropMemoryDescription = outputLayer->OutBackPropMemoryDescriptions()[formatIndex];

			vector<unique_ptr<OCLMemory>> backPropMemoryHolder;
			unique_ptr<OCLMemory> backPropOutputMemory = contexts[0]->CreateMemory(CL_MEM_READ_ONLY, sizeof(T) * inBackPropMemoryDescription.TotalMemory());
			OCLMemory* backPropMemory = backPropOutputMemory.get();
			backPropMemoryHolder.push_back(move(backPropOutputMemory));
			outputLayer->EnqueueBackPropagation(device, 0, inputMemories[inputMemories.size() - 1], target, backPropMemory, false);

			vector<vector<unique_ptr<OCLMemory>>> gradientMemoryHolders;
			vector<vector<OCLMemory*>> gradientMemories;
			vector<vector<size_t>> allParameterCounts;

			if (layers.size() != 0)
			{
				auto parameterCounts = layers[layers.size() - 1]->GetMultipleParameterCount();
				allParameterCounts.push_back(parameterCounts);
				gradientMemoryHolders.push_back(vector<unique_ptr<OCLMemory>>());
				gradientMemories.push_back(vector<OCLMemory*>());
				auto& gradientMemoryVector = gradientMemories[gradientMemories.size() -1];
				auto& gradientMemoryHolder = gradientMemoryHolders[gradientMemoryHolders.size() -1];
				for (auto parameterCount : parameterCounts)
				{
					unique_ptr<OCLMemory> gradientMemory = contexts[0]->CreateMemory(CL_MEM_WRITE_ONLY, sizeof(T) * parameterCount);
					gradientMemoryVector.push_back(gradientMemory.get());
					gradientMemoryHolder.push_back(move(gradientMemory));
				}

				layers[layers.size() - 1]->EnqueueCalculateGradient(device, 0, inputMemories[inputMemories.size() - 2], backPropMemory, gradientMemoryVector, false);
			}

			for (int i = static_cast<int>(layers.size()) - 1; i >= 1; i--) 
			{
				inBackPropMemoryDescription = layers[i]->OutBackPropMemoryDescriptions()[formatIndex];
				unique_ptr<OCLMemory> outputMemory = contexts[0]->CreateMemory(CL_MEM_READ_WRITE, sizeof(T) * inBackPropMemoryDescription.TotalMemory());
				layers[i]->EnqueueBackPropagation(device, 0, inputMemories[i], backPropMemory, outputMemory.get(), false);
				backPropMemory = outputMemory.get();
				backPropMemoryHolder.push_back(move(outputMemory));


				auto parameterCounts = layers[i - 1]->GetMultipleParameterCount();
				allParameterCounts.push_back(parameterCounts);
				gradientMemoryHolders.push_back(vector<unique_ptr<OCLMemory>>());
				gradientMemories.push_back(vector<OCLMemory*>());
				auto& gradientMemoryVector = gradientMemories[gradientMemories.size() -1];
				auto& gradientMemoryHolder = gradientMemoryHolders[gradientMemoryHolders.size() -1];
				for (auto parameterCount : parameterCounts)
				{
					unique_ptr<OCLMemory> gradientMemory = contexts[0]->CreateMemory(CL_MEM_WRITE_ONLY, sizeof(T) * parameterCount);
					gradientMemoryVector.push_back(gradientMemory.get());
					gradientMemoryHolder.push_back(move(gradientMemory));
				}

				layers[i - 1]->EnqueueCalculateGradient(device, 0, inputMemories[i - 1], backPropMemory, gradientMemoryVector, false);
			}

			unique_ptr<T[]> gradient(new T[GetParameterCount()]);
			size_t gradientMemoryPosition = 0;


			if (gradientMemories.size() != gradientMemoryHolders.size())
				throw runtime_error("Invalid implementation");

			if (gradientMemories.size() != allParameterCounts.size())
				throw runtime_error("Invalid implementation");

			for(size_t i = 0; i < gradientMemories.size(); i++)
			{
				if (gradientMemories[i].size() != allParameterCounts[i].size())
					throw runtime_error("Invalid implementation");

				auto& gradientMemoryVector = gradientMemories[i];
				auto& parameterCount = allParameterCounts[i];
				auto rawGradient = gradient.get();
				rawGradient += gradientMemoryPosition;
				for (size_t k = 0; k < parameterCount.size(); k++)
				{
					device->ReadMemory(gradientMemoryVector[k], gradientMemoryVector[k]->ByteSize(), rawGradient, 0, false);
					gradientMemoryPosition += parameterCount[k];
					rawGradient = gradient.get() + gradientMemoryPosition;
				}
			}

			device->WaitForDeviceQueue(0);

			return move(gradient);
		}

		template<class T>
		unique_ptr<T[]> OCLConvNet<T>::CalculateGradientLowMemory(OCLMemory* input, int formatIndex, OCLMemory* target)
		{
			auto devices = contexts[0]->GetDevices();
			auto device = devices[0];
			vector<unique_ptr<OCLMemory>> inputMemoriesHolder;
			FeedForwardHighMemory(input, formatIndex, inputMemoriesHolder);
			vector<OCLMemory*> inputMemories;
			inputMemories.push_back(input);
			for (auto& holder : inputMemoriesHolder)
				inputMemories.push_back(holder.get());

			LayerMemoryDescription inBackPropMemoryDescription = outputLayer->OutBackPropMemoryDescriptions()[formatIndex];
			unique_ptr<OCLMemory> backPropOutputMemory = contexts[0]->CreateMemory(CL_MEM_READ_ONLY, sizeof(T) * inBackPropMemoryDescription.TotalMemory());

			outputLayer->EnqueueBackPropagation(device, 0, inputMemories[inputMemories.size() - 1], target, backPropOutputMemory.get(), false);
			auto count = layers.size();

			unique_ptr<T[]> gradient(new T[GetParameterCount()]);
			size_t gradientMemoryPosition = 0;
			if (layers.size() != 0) 
			{
				//Allocate memory for the gradient
				auto parametersCount = layers[count - 1]->GetMultipleParameterCount();
				vector<unique_ptr<OCLMemory>> gradientMemories;
				vector<OCLMemory*> gradientsPointers;
				for (auto parameterCount : parametersCount)
				{
					unique_ptr<OCLMemory> gradientMemory = contexts[0]->CreateMemory(CL_MEM_WRITE_ONLY, sizeof(T) * parameterCount);
					gradientsPointers.push_back(gradientMemory.get());
					gradientMemories.push_back(move(gradientMemory));
				}
				layers[count - 1]->EnqueueCalculateGradient(device, 0, inputMemories[inputMemories.size() - 2], backPropOutputMemory.get(), gradientsPointers, false);

				//Write gradient memory to the host buffer
				auto rawGradient = gradient.get();
				rawGradient += gradientMemoryPosition;
				for (size_t k = 0; k < parametersCount.size(); k++)
				{
					device->ReadMemory(gradientsPointers[k], gradientsPointers[k]->ByteSize(), rawGradient, 0, false);
					gradientMemoryPosition += parametersCount[k];
					rawGradient = gradient.get() + gradientMemoryPosition;
				}
			}

			device->WaitForDeviceQueue(0);
			inputMemoriesHolder[inputMemoriesHolder.size() - 1].reset();

			for (int i = static_cast<int>(count) - 1; i >= 1; i--) 
			{
				inBackPropMemoryDescription = layers[i]->OutBackPropMemoryDescriptions()[formatIndex];

				unique_ptr<OCLMemory> outputMemory = contexts[0]->CreateMemory(CL_MEM_READ_WRITE, sizeof(T) * inBackPropMemoryDescription.TotalMemory());
				layers[i]->EnqueueBackPropagation(device, 0, inputMemories[i], backPropOutputMemory.get(), outputMemory.get(), true);
				inputMemoriesHolder[i - 1].reset();
				backPropOutputMemory.reset();
				backPropOutputMemory = move(outputMemory);

				//Allocate memory for the gradient
				auto parametersCount = layers[i- 1]->GetMultipleParameterCount();
				vector<unique_ptr<OCLMemory>> gradientMemories;
				vector<OCLMemory*> gradientsPointers;
				for (auto parameterCount : parametersCount)
				{
					unique_ptr<OCLMemory> gradientMemory = contexts[0]->CreateMemory(CL_MEM_WRITE_ONLY, sizeof(T) * parameterCount);
					gradientsPointers.push_back(gradientMemory.get());
					gradientMemories.push_back(move(gradientMemory));
				}

				layers[i - 1]->EnqueueCalculateGradient(device, 0, inputMemories[i - 1], backPropOutputMemory.get(), gradientsPointers, false);

				//Write gradient memory to the host buffer
				auto rawGradient = gradient.get();
				rawGradient += gradientMemoryPosition;
				for (size_t k = 0; k < parametersCount.size(); k++)
				{
					device->ReadMemory(gradientsPointers[k], gradientsPointers[k]->ByteSize(), rawGradient, 0, false);
					gradientMemoryPosition += parametersCount[k];
					rawGradient = gradient.get() + gradientMemoryPosition;
				}

				device->WaitForDeviceQueue(0);
			}

			return move(gradient);
		}

		template<class T>
		unique_ptr<T[]> OCLConvNet<T>::CalculateGradientAligned(OCLMemory* input, int formatIndex, OCLMemory* target)
		{
			if (lowMemoryUsage)
				return move(CalculateGradientLowMemory(input, formatIndex, target));
			else
				return move(CalculateGradientHighMemory(input, formatIndex, target));
		}

		template<class T>
		unique_ptr<T[]> OCLConvNet<T>::CalculateGradientAligned(T* input, int formatIndex, T* target) 
		{
			auto targetMemory = CreateTargetMemory(target, formatIndex, 0);
			auto inputMemory = CreateInputMemory(input, formatIndex, 0);
			if (lowMemoryUsage)
				return move(CalculateGradientLowMemory(inputMemory.get(), formatIndex, targetMemory.get()));
			else
				return move(CalculateGradientHighMemory(inputMemory.get(), formatIndex, targetMemory.get()));
		}

		template<class T>
		unique_ptr<T[]> OCLConvNet<T>::GetParameters() 
		{
			//TODO: At the moment we only support one context and one device
			auto devices = contexts[0]->GetDevices();
			auto device = devices[0];
			unique_ptr<T[]> parameters(new T[GetParameterCount()]);
			auto pointerPosition = parameters.get();

			//The parameters are always given in reverse order since it's easier for the gradient calculation
			for (int i = static_cast<int>(layers.size()) - 1; i >= 0; i--) {
				layers[i]->GetParameters(pointerPosition, device, 0, false);
				pointerPosition += layers[i]->GetParameterCount();
			}

			device->WaitForDeviceQueue(0);

			return move(parameters);
		}

		template<class T>
		void OCLConvNet<T>::SetParameters(T* parameters)
		{
			//TODO: At the moment we only support one context and one device
			auto devices = contexts[0]->GetDevices();
			auto device = devices[0];
			auto pointerPosition = parameters;

			//The parameters are always given in reverse order since it's easier for the gradient calculation
			for (int i = static_cast<int>(layers.size()) - 1; i >= 0; i--) {
				layers[i]->SetParameters(pointerPosition, device, 0, false);
				pointerPosition += layers[i]->GetParameterCount();
			}

			device->WaitForDeviceQueue(0);
		}

		template<class T>
		size_t OCLConvNet<T>::GetParameterCount() 
		{
			size_t result = 0;
			for (auto& layer : layers)
				result += layer->GetParameterCount();

			return result;
		}

		template<class T>
		void OCLConvNet<T>::TrainNetwork2(unique_ptr<ConvNetTrainer<T>> trainer,
			unique_ptr<IAlgorithmConfig> algorithm) 
		{
			auto gdConfig = dynamic_cast<GradientDescentConfig<T>*>(algorithm.get());
			if (gdConfig)
			{
				algorithm.release();
				if (lowMemoryUsage)
					TrainNetworkGDLowMemory(unique_ptr<GradientDescentConfig<T>>(gdConfig), move(trainer));
				else
					TrainNetworkGDHighMemory(unique_ptr<GradientDescentConfig<T>>(gdConfig), move(trainer));
			}
			else
				throw invalid_argument("Algorithm is not supported");

		}

		template<class T>
		void OCLConvNet<T>::ReadInputDataAsync(ConvNetTrainer<T>* trainer)
		{
			//TODO: At the moment we only support one context and one device
			auto devices = contexts[0]->GetDevices();
			auto device = devices[0];

			//TODO: Only support format 0 at the moment
			LayerMemoryDescription inputMemoryDescription = this->InputForwardMemoryDescriptions()[0];
			LayerMemoryDescription inBackPropMemoryDescription = this->OutputForwardMemoryDescriptions()[0];

			while(trainingIsRunning)
			{
				int dataID = trainer->DataIDRequest();
				int referencesForID = inputDataBufferQueue->ReferenceCount(dataID);

				T* input;
				T* target;
				int formatIndex;

				//If we don't have a reference, we are 100% sure that we need to read the data
				if (referencesForID == 0)
				{
					trainer->MapInputAndTarget(dataID, input, target, formatIndex);
					if (formatIndex != 0)
						throw runtime_error("Other format indices than 0 is not supported at the moment");

					//printf("Writing data to device memory, ID: %i \n", dataID);

					//Here we will read the memory using the second device queue to maximize the througput of the device
					unique_ptr<OCLMemory> inputMemory = contexts[0]->CreateMemory(CL_MEM_READ_ONLY, inputMemoryDescription.TotalMemory() * sizeof(T));
					unique_ptr<OCLMemory> targetMemory = contexts[0]->CreateMemory(CL_MEM_READ_ONLY, inBackPropMemoryDescription.TotalMemory() * sizeof(T));
					device->WriteMemory(inputMemory.get(), inputMemory->ByteSize(), input, 1, false);
					device->WriteMemory(targetMemory.get(), targetMemory->ByteSize(), target, 1, false);
					device->WaitForDeviceQueue(1);
					trainer->UnmapInputAndTarget(dataID, input, target, formatIndex);
					inputDataBufferQueue->Push(dataID, formatIndex, move(inputMemory), move(targetMemory));
				}
				else
					inputDataBufferQueue->Push(dataID);
			}
		}

		template<class T>
		void OCLConvNet<T>::InnerLoopGDHighMemory(
			size_t layerCount,
			int sample,
			LayerKernel<T>* vectorKernel,
			OCLDevice* device,
			const vector<OCLMemory*>& inputMemoriesInput,
			const vector<OCLMemory*>& backPropMemories,
			const vector<vector<OCLMemory*>>& gradientsPointers, 
			const vector<vector<OCLMemory*>>& accumulatedGradientsPointers)
		{
			//Observe that the input data must live as long as we need it!
			try
			{
				auto inputData = inputDataBufferQueue->LockAndAcquire();
				int formatIndex = inputData.FormatIndex();

				if (formatIndex != 0)
					throw invalid_argument("We don't support other formats than 0 at the moment");

				vector<OCLMemory*> inputMemories;
				inputMemories.push_back(inputData.GetInput());
				for(auto memoryObject : inputMemoriesInput)
					inputMemories.push_back(memoryObject);

				//Forward prop
				for (size_t i = 0; i < layerCount; i++)
					layers[i]->EnqueueForwardPropagation(device, 0, inputMemories[i], inputMemories[i + 1], false);

				//Back prop
				outputLayer->EnqueueBackPropagation(device, 0, inputMemories[inputMemories.size() - 1], inputData.GetTarget(), backPropMemories[backPropMemories.size() -1], false);

				if (layerCount != 0) 
					layers[layerCount - 1]->EnqueueCalculateGradient(device, 0, inputMemories[inputMemories.size() - 2],
					backPropMemories[backPropMemories.size() -1], gradientsPointers[layerCount - 1], false);

				for (int i = static_cast<int>(layerCount) - 1; i >= 1; i--) 
				{
					layers[i]->EnqueueBackPropagation(device, 0, inputMemories[i], backPropMemories[i], backPropMemories[i - 1], false);
					layers[i - 1]->EnqueueCalculateGradient(device, 0, inputMemories[i - 1], backPropMemories[i - 1], gradientsPointers[i - 1], false);
				}

			}
			catch(...)
			{
				inputDataBufferQueue->UnlockAcquire();
				throw;
			}

			inputDataBufferQueue->UnlockAcquire();

			//If this is the first sample, add the memory to the accumulator and continue
			if (sample == 0) 
			{
				for (size_t i = 0; i < layerCount; i++) 
				{
					for (size_t j = 0; j < gradientsPointers[i].size(); j++) 
					{
						if (gradientsPointers[i][j]->ByteSize() == accumulatedGradientsPointers[i][j]->ByteSize())
							device->CopyCLMemory(gradientsPointers[i][j], accumulatedGradientsPointers[i][j], 0, 0, gradientsPointers[i][j]->ByteSize(), 0, false);
						else
							throw runtime_error( "The memory does not match for the accumulated gradient.");
					}
				}
			} 
			else 
			{
				for (size_t i = 0; i < layerCount; i++) 
				{
					vector<size_t> parameterCount = layers[i]->GetMultipleParameterCount();
					for (size_t j = 0; j < gradientsPointers[i].size(); j++) 
					{
						vectorKernel->SetMemoryArg(accumulatedGradientsPointers[i][j], 1);
						vectorKernel->SetMemoryArg(gradientsPointers[i][j], 0);
						vectorKernel->ClearGlobalSizes();
						vectorKernel->AddGlobalSize(parameterCount[j]);
						device->ExecuteKernel(vectorKernel, 0, false);
					}
				}
			}
		}

		template<class T>
		void OCLConvNet<T>::InnerLoopGDLowMemory(size_t layerCount, int sample,
			LayerKernel<T>* vectorKernel,
			const vector<vector<OCLMemory*>>& gradientsPointers, 
			OCLDevice* device,
			const vector<vector<OCLMemory*>>& accumulatedGradientsPointers)
		{
			//Observe that the input data must live as long as we need it!
			vector<unique_ptr<OCLMemory>> inputMemoriesHolder;
			vector<OCLMemory*> inputMemories;
			int formatIndex;
			unique_ptr<OCLMemory> backPropOutputMemory;
			try
			{
				auto inputData = inputDataBufferQueue->LockAndAcquire();
				formatIndex = inputData.FormatIndex();
				FeedForwardHighMemory(inputData.GetInput(), inputData.FormatIndex(), inputMemoriesHolder);
				inputMemories.push_back(inputData.GetInput());
				for (auto& holder : inputMemoriesHolder)
					inputMemories.push_back(holder.get());

				LayerMemoryDescription outBackPropMemoryDescription = outputLayer->OutBackPropMemoryDescriptions()[formatIndex];
				backPropOutputMemory = contexts[0]->CreateMemory(CL_MEM_READ_WRITE, sizeof(T) * outBackPropMemoryDescription.TotalMemory());
				outputLayer->EnqueueBackPropagation(device, 0, inputMemories[inputMemories.size() - 1], inputData.GetTarget(), backPropOutputMemory.get(), false);

				if (layerCount != 0) 
					layers[layerCount - 1]->EnqueueCalculateGradient(device, 0, inputMemories[inputMemories.size() - 2],
					backPropOutputMemory.get(), gradientsPointers[layerCount - 1], false);

				device->WaitForDeviceQueue(0);
				inputMemoriesHolder[inputMemoriesHolder.size() - 1].reset();

				for (int i = static_cast<int>(layerCount) - 1; i >= 1; i--) 
				{
					LayerMemoryDescription inBackPropMemoryDescription = layers[i]->OutBackPropMemoryDescriptions()[formatIndex];
					unique_ptr<OCLMemory> outputMemory = contexts[0]->CreateMemory(CL_MEM_READ_WRITE, sizeof(T) * inBackPropMemoryDescription.TotalMemory());
					layers[i]->EnqueueBackPropagation(device, 0, inputMemories[i], backPropOutputMemory.get(), outputMemory.get(), true);
					backPropOutputMemory.reset();
					backPropOutputMemory = move(outputMemory);
					inputMemoriesHolder[i - 1].reset();
					layers[i - 1]->EnqueueCalculateGradient(device, 0, inputMemories[i - 1], backPropOutputMemory.get(), gradientsPointers[i - 1], false);
				}

			}
			catch(...)
			{
				inputDataBufferQueue->UnlockAcquire();
				throw;
			}

			inputDataBufferQueue->UnlockAcquire();

			//If this is the first sample, add the memory to the accumulator and continue
			if (sample == 0) 
			{
				for (size_t i = 0; i < layerCount; i++) 
				{
					for (size_t j = 0; j < gradientsPointers[i].size(); j++) 
					{
						if (gradientsPointers[i][j]->ByteSize() == accumulatedGradientsPointers[i][j]->ByteSize())
							device->CopyCLMemory(gradientsPointers[i][j], accumulatedGradientsPointers[i][j], 0, 0, gradientsPointers[i][j]->ByteSize(), 0, false);
						else
							throw runtime_error( "The memory does not match for the accumulated gradient.");
					}
				}
			} 
			else 
			{
				for (size_t i = 0; i < layerCount; i++) 
				{
					vector<size_t> parameterCount = layers[i]->GetMultipleParameterCount();
					for (size_t j = 0; j < gradientsPointers[i].size(); j++) 
					{
						vectorKernel->SetMemoryArg(accumulatedGradientsPointers[i][j], 1);
						vectorKernel->SetMemoryArg(gradientsPointers[i][j], 0);
						vectorKernel->ClearGlobalSizes();
						vectorKernel->AddGlobalSize(parameterCount[j]);
						device->ExecuteKernel(vectorKernel, 0, false);
					}
				}
			}

			//We don't really need this one since the gradient memory will not be collected
			device->WaitForDeviceQueue(0);
		}

		template<class T>
		void OCLConvNet<T>::UpdateGradient(size_t layerCount, const vector<vector<OCLMemory*>>& accumulatedGradientsPointers,
			LayerKernel<T>* scalarKernel, OCLDevice* device, bool blocking)
		{
			for (size_t i = 0; i < layerCount; i++) 
			{
				vector<OCLMemory*> parametersToUpdate = layers[i]->GetParameters();
				vector<size_t> parameterCount = layers[i]->GetMultipleParameterCount();

				if (parametersToUpdate.size() != accumulatedGradientsPointers[i].size())
					throw runtime_error("The accumulated gradients do not match the layer parameters");

				for (size_t j = 0; j < accumulatedGradientsPointers[i].size(); j++) 
				{
					if (parametersToUpdate[j]->ByteSize() != accumulatedGradientsPointers[i][j]->ByteSize())
						throw runtime_error("The gradient and the parameter memories does not match");

					scalarKernel->SetMemoryArg(accumulatedGradientsPointers[i][j], 0);
					scalarKernel->SetMemoryArg(parametersToUpdate[j], 1);
					scalarKernel->ClearGlobalSizes();
					scalarKernel->AddGlobalSize(parameterCount[j]);
					device->ExecuteKernel(scalarKernel, 0, false);
				}
			}

			if (blocking)
				device->WaitForDeviceQueue(0);
		}

		template<class T>
		void OCLConvNet<T>::TrainNetworkGDHighMemory(unique_ptr<GradientDescentConfig<T>> gdConfig, unique_ptr<ConvNetTrainer<T>> trainer)
		{
			//Pre-requisists
			function<T(int)> stepSizeCallback = gdConfig->GetStepSizeCallback();
			int batchSize = gdConfig->GetBatchSize();
			int samplesPerEpoch = gdConfig->GetSamplesPerEpoch();
			int epochs = gdConfig->GetEpochs();
			int dataBufferSize = trainer->GetBufferSize();
			bool enableErrorReporting = trainer->GetEnableError();

			//The pointer to the program that we'll detach in the end
			//TODO: At the moment we only support one context and one device
			auto devices = contexts[0]->GetDevices();
			auto device = devices[0];
			vector<OCLDevice*> oneDeviceVector;
			oneDeviceVector.push_back(device);

			unique_ptr<OCLProgram> gradientProgram(new OCLProgram());
			gradientProgram->AddIncludePath(OCLProgram::DefaultSourceLocation);
			gradientProgram->AddProgramPath(Path::Combine(OCLProgram::DefaultSourceLocation, "AccumulateVectorProgram.cl"));
			gradientProgram->SetName("GradientProgram" + Converter::ConvertToString(gradientProgram->InstanceCount()));
			auto programPointer = gradientProgram.get();
			unique_ptr<LayerKernel<T>> vectorKernelHolder(new LayerKernel<T>());
			auto vectorKernel = vectorKernelHolder.get();
			vectorKernel->SetKernelName("AccumulateVectorKernel");
			unique_ptr<LayerKernel<T>> scalarKernelHolder(new LayerKernel<T>());
			auto scalarKernel = scalarKernelHolder.get();
			scalarKernel->SetKernelName("AccumulateVectorWithScalarKernel");

			auto deviceInfo = device->DeviceInfo();
			if (is_same<cl_double, T>::value) 
				gradientProgram->AddDefine("DOUBLE_PRECISION");

			T currentStepSize = -stepSizeCallback(0);

			gradientProgram->AttachKernel(move(vectorKernelHolder));
			gradientProgram->AttachKernel(move(scalarKernelHolder));

			contexts[0]->AttachProgram(move(gradientProgram), oneDeviceVector);

			scalarKernel->SetRealArg(currentStepSize, 2);

			int batchIterations = samplesPerEpoch / batchSize;
			int remainder = samplesPerEpoch % batchSize;

			//Allocate all intermediate memory
			vector<unique_ptr<OCLMemory>> inputMemoriesHolder;
			vector<OCLMemory*> inputMemories;

			vector<unique_ptr<OCLMemory>> outputMemoriesHolder;
			vector<OCLMemory*> outputMemories;

			vector<vector<unique_ptr<OCLMemory>>> gradients;
			vector<vector<OCLMemory*>> gradientsPointers;
			vector<vector<unique_ptr<OCLMemory>>> accumulatedGradients;
			vector<vector<OCLMemory*>> accumulatedGradientsPointers;
			size_t layerCount = layers.size();
			for (size_t i = 0; i < layerCount; i++) 
			{
				//TODO: We only support one format at the moment
				LayerMemoryDescription inputMemoryDescription = layers[i]->OutForwardPropMemoryDescriptions()[0];
				unique_ptr<OCLMemory> inputMemory = contexts[0]->CreateMemory(CL_MEM_READ_WRITE, sizeof(T) * inputMemoryDescription.TotalMemory());
				inputMemories.push_back(inputMemory.get());
				inputMemoriesHolder.push_back(move(inputMemory));

				if (i != 0)
				{
					//TODO: We only support one format at the moment
					LayerMemoryDescription inBackPropMemoryDescription = layers[i]->OutBackPropMemoryDescriptions()[0];
					unique_ptr<OCLMemory> outputMemory = contexts[0]->CreateMemory(CL_MEM_READ_WRITE, sizeof(T) * inBackPropMemoryDescription.TotalMemory());
					outputMemories.push_back(outputMemory.get());
					outputMemoriesHolder.push_back(move(outputMemory));
				}

				gradients.push_back(vector<unique_ptr<OCLMemory>>());
				accumulatedGradients.push_back(vector<unique_ptr<OCLMemory>>());
				gradientsPointers.push_back(vector<OCLMemory*>());
				accumulatedGradientsPointers.push_back(vector<OCLMemory*>());

				vector<size_t> parameterCount = layers[i]->GetMultipleParameterCount();
				for (size_t k = 0; k < parameterCount.size(); k++) 
				{
					unique_ptr<OCLMemory> tempGradientMemory1 = contexts[0]->CreateMemory(CL_MEM_READ_WRITE, sizeof(T) * parameterCount[k]);
					unique_ptr<OCLMemory> tempGradientMemory2 = contexts[0]->CreateMemory(CL_MEM_READ_WRITE, sizeof(T) * parameterCount[k]);
					gradientsPointers[i].push_back(tempGradientMemory1.get());
					gradients[i].push_back(move(tempGradientMemory1));
					accumulatedGradientsPointers[i].push_back(tempGradientMemory2.get());
					accumulatedGradients[i].push_back(move(tempGradientMemory2));
				}
			}

			//TODO: We only support one format at the moment
			LayerMemoryDescription inBackPropMemoryDescription = outputLayer->OutBackPropMemoryDescriptions()[0];
			unique_ptr<OCLMemory> backPropOutputMemory = contexts[0]->CreateMemory(CL_MEM_READ_ONLY, sizeof(T) * inBackPropMemoryDescription.TotalMemory());

			outputMemories.push_back(backPropOutputMemory.get());
			outputMemoriesHolder.push_back(move(backPropOutputMemory));

			bool stopped = false;

			//These calls will read data into a circular buffer in parallel with the computations.
			//The buffer size is determined by the trainer. This makes it possible to tweak the memory consumption
			//For smaller datasets, we can read the entire data into RAM and in this way avoid memory copies.
			trainingIsRunning = true;
			inputDataBufferQueue = unique_ptr<InputDataBufferQueue>(new InputDataBufferQueue(dataBufferSize));
			thread inputReader(&OCLConvNet<T>::ReadInputDataAsync, this, trainer.get());

			try
			{
				for (int epoch = 0; epoch < epochs; epoch++) 
				{
					trainer->EpochStarted();
					T stepSize = -stepSizeCallback(0);
					for (int batch = 0; batch < batchIterations; batch++) 
					{
						trainer->BatchStarted();
						for (int sample = 0; sample < batchSize; sample++) 
						{
							if (trainer->Stopping()) 
							{
								stopped = true;
								break;
							}
							InnerLoopGDHighMemory(layerCount, sample, vectorKernel,
								device,inputMemories, outputMemories, gradientsPointers,
								accumulatedGradientsPointers);
						}
						if(stopped)
							break;
						if (stepSize != currentStepSize) 
						{
							scalarKernel->SetRealArg(stepSize, 2);
							currentStepSize = stepSize;
						}
						UpdateGradient(layerCount, accumulatedGradientsPointers, scalarKernel, device, false);
						//TODO: Enable error reporting
						trainer->BatchFinished(-1);
					}
					if(stopped)
						break;
					if (remainder != 0)
					{
						trainer->BatchStarted();
						for (int sample = 0; sample < remainder; sample++) 
						{
							if (trainer->Stopping()) 
							{
								stopped = true;
								break;
							}
							InnerLoopGDHighMemory(layerCount, sample, vectorKernel,
								device,inputMemories, outputMemories, gradientsPointers,
								accumulatedGradientsPointers);
						}
						if (stopped)
							break;
						if (stepSize != currentStepSize) 
						{
							scalarKernel->SetRealArg(stepSize, 2);
							currentStepSize = stepSize;
						}
						UpdateGradient(layerCount, accumulatedGradientsPointers, scalarKernel, device, false);
						//TODO: Enable error reporting
						trainer->BatchFinished(-1);
					}

					trainer->EpochFinished();
				}
			}
			catch(...)
			{
				trainingIsRunning = false;
				inputDataBufferQueue->MoveReader(); //Observe that we can never get stuck in this thread as long as the other thread is pumping data.
				inputReader.join();
				inputDataBufferQueue.reset();
				device->WaitForDeviceQueue(0);
				device->WaitForDeviceQueue(1);
				contexts[0]->DetachProgram(programPointer);
				throw;
			}

			trainingIsRunning = false;
			inputDataBufferQueue->MoveReader(); //Observe that we can never get stuck in this thread as long as the other thread is pumping data.
			inputReader.join();
			inputDataBufferQueue.reset();
			device->WaitForDeviceQueue(0);
			device->WaitForDeviceQueue(1);
			contexts[0]->DetachProgram(programPointer);
		}

		template<class T>
		void OCLConvNet<T>::TrainNetworkGDLowMemory(unique_ptr<GradientDescentConfig<T>> gdConfig, unique_ptr<ConvNetTrainer<T>> trainer)
		{
			//Pre-requisists
			function<T(int)> stepSizeCallback = gdConfig->GetStepSizeCallback();
			int batchSize = gdConfig->GetBatchSize();
			int samplesPerEpoch = gdConfig->GetSamplesPerEpoch();
			int epochs = gdConfig->GetEpochs();
			int dataBufferSize = trainer->GetBufferSize();
			bool enableErrorReporting = trainer->GetEnableError();

			//The pointer to the program that we'll detach in the end
			//TODO: At the moment we only support one context and one device
			auto devices = contexts[0]->GetDevices();
			auto device = devices[0];
			vector<OCLDevice*> oneDeviceVector;
			oneDeviceVector.push_back(device);

			unique_ptr<OCLProgram> gradientProgram(new OCLProgram());
			gradientProgram->AddIncludePath(OCLProgram::DefaultSourceLocation);
			gradientProgram->AddProgramPath(Path::Combine(OCLProgram::DefaultSourceLocation, "AccumulateVectorProgram.cl"));
			gradientProgram->SetName("GradientProgram" + Converter::ConvertToString(gradientProgram->InstanceCount()));
			auto programPointer = gradientProgram.get();
			unique_ptr<LayerKernel<T>> vectorKernelHolder(new LayerKernel<T>());
			auto vectorKernel = vectorKernelHolder.get();
			vectorKernel->SetKernelName("AccumulateVectorKernel");
			unique_ptr<LayerKernel<T>> scalarKernelHolder(new LayerKernel<T>());
			auto scalarKernel = scalarKernelHolder.get();
			scalarKernel->SetKernelName("AccumulateVectorWithScalarKernel");

			auto deviceInfo = device->DeviceInfo();
			if (is_same<cl_double, T>::value) 
				gradientProgram->AddDefine("DOUBLE_PRECISION");

			T currentStepSize = -stepSizeCallback(0);

			gradientProgram->AttachKernel(move(vectorKernelHolder));
			gradientProgram->AttachKernel(move(scalarKernelHolder));

			contexts[0]->AttachProgram(move(gradientProgram), oneDeviceVector);

			scalarKernel->SetRealArg(currentStepSize, 2);

			int batchIterations = samplesPerEpoch / batchSize;
			int remainder = samplesPerEpoch % batchSize;

			//Allocate memory for the gradients
			vector<vector<unique_ptr<OCLMemory>>> gradients;
			vector<vector<OCLMemory*>> gradientsPointers;
			vector<vector<unique_ptr<OCLMemory>>> accumulatedGradients;
			vector<vector<OCLMemory*>> accumulatedGradientsPointers;
			size_t layerCount = layers.size();
			for (size_t i = 0; i < layerCount; i++) 
			{
				gradients.push_back(vector<unique_ptr<OCLMemory>>());
				accumulatedGradients.push_back(vector<unique_ptr<OCLMemory>>());
				gradientsPointers.push_back(vector<OCLMemory*>());
				accumulatedGradientsPointers.push_back(vector<OCLMemory*>());

				vector<size_t> parameterCount = layers[i]->GetMultipleParameterCount();
				for (size_t k = 0; k < parameterCount.size(); k++) 
				{
					unique_ptr<OCLMemory> tempGradientMemory1 = contexts[0]->CreateMemory(CL_MEM_READ_WRITE, sizeof(T) * parameterCount[k]);
					unique_ptr<OCLMemory> tempGradientMemory2 = contexts[0]->CreateMemory(CL_MEM_READ_WRITE, sizeof(T) * parameterCount[k]);
					gradientsPointers[i].push_back(tempGradientMemory1.get());
					gradients[i].push_back(move(tempGradientMemory1));
					accumulatedGradientsPointers[i].push_back(tempGradientMemory2.get());
					accumulatedGradients[i].push_back(move(tempGradientMemory2));
				}
			}

			bool stopped = false;

			//These calls will read data into a circular buffer in parallel with the computations.
			//The buffer size is determined by the trainer. This makes it possible to tweak the memory consumption
			//For smaller datasets, we can read the entire data into RAM and in this way avoid memory copies.
			trainingIsRunning = true;
			inputDataBufferQueue = unique_ptr<InputDataBufferQueue>(new InputDataBufferQueue(dataBufferSize));
			thread inputReader(&OCLConvNet<T>::ReadInputDataAsync, this, trainer.get());

			try
			{
				for (int epoch = 0; epoch < epochs; epoch++) 
				{
					trainer->EpochStarted();
					T stepSize = -stepSizeCallback(0);
					for (int batch = 0; batch < batchIterations; batch++) 
					{
						trainer->BatchStarted();
						for (int sample = 0; sample < batchSize; sample++) 
						{
							if (trainer->Stopping()) 
							{
								stopped = true;
								break;
							}
							InnerLoopGDLowMemory(layerCount, sample, vectorKernel, gradientsPointers, device, accumulatedGradientsPointers);
						}
						if(stopped)
							break;
						if (stepSize != currentStepSize) 
						{
							scalarKernel->SetRealArg(stepSize, 2);
							currentStepSize = stepSize;
						}
						UpdateGradient(layerCount, accumulatedGradientsPointers, scalarKernel, device, true);
						//TODO: Enable error reporting
						trainer->BatchFinished(-1);
					}
					if(stopped)
						break;
					if (remainder != 0)
					{
						trainer->BatchStarted();
						for (int sample = 0; sample < remainder; sample++) 
						{
							if (trainer->Stopping()) 
							{
								stopped = true;
								break;
							}
							InnerLoopGDLowMemory(layerCount, sample, vectorKernel, gradientsPointers, device, accumulatedGradientsPointers);
						}
						if (stopped)
							break;
						if (stepSize != currentStepSize) 
						{
							scalarKernel->SetRealArg(stepSize, 2);
							currentStepSize = stepSize;
						}
						UpdateGradient(layerCount, accumulatedGradientsPointers, scalarKernel, device, true);
						//TODO: Enable error reporting
						trainer->BatchFinished(-1);
					}

					trainer->EpochFinished();
				}
			}
			catch(...)
			{
				trainingIsRunning = false;
				inputDataBufferQueue->MoveReader(); //Observe that we can never get stuck in this thread as long as the other thread is pumping data.
				inputReader.join();
				inputDataBufferQueue.reset();
				device->WaitForDeviceQueue(0);
				device->WaitForDeviceQueue(1);
				contexts[0]->DetachProgram(programPointer);
				throw;
			}

			trainingIsRunning = false;
			inputDataBufferQueue->MoveReader(); //Observe that we can never get stuck in this thread as long as the other thread is pumping data.
			inputReader.join();
			inputDataBufferQueue.reset();
			device->WaitForDeviceQueue(0);
			device->WaitForDeviceQueue(1);
			contexts[0]->DetachProgram(programPointer);
		}

		template<class T>
		void OCLConvNet<T>::TrainNetwork(unique_ptr<ConvNetTrainer<T>> trainer,
			unique_ptr<IAlgorithmConfig> algorithm) 
		{
			auto gdConfig = dynamic_cast<GradientDescentConfig<T>*>(algorithm.get());
			if (!gdConfig)
				throw invalid_argument(
				"Gradient descent is the only algorithm supported at the moment");

			function<T(int)> stepSizeCallback = gdConfig->GetStepSizeCallback();
			int batchSize = gdConfig->GetBatchSize();
			int samplesPerEpoch = gdConfig->GetSamplesPerEpoch();
			int epochs = gdConfig->GetEpochs();

			//TODO: At the moment we only support one context and one device
			auto devices = contexts[0]->GetDevices();
			auto device = devices[0];
			vector<OCLDevice*> oneDeviceVector;
			oneDeviceVector.push_back(device);

			unique_ptr<OCLProgram> gradientProgram(new OCLProgram());
			gradientProgram->AddIncludePath(OCLProgram::DefaultSourceLocation);
			gradientProgram->AddProgramPath(Path::Combine(OCLProgram::DefaultSourceLocation, "AccumulateVectorProgram.cl"));
			gradientProgram->SetName("GradientProgram" + Converter::ConvertToString(gradientProgram->InstanceCount()));
			auto programPointer = gradientProgram.get();
			unique_ptr<LayerKernel<T>> vectorKernelHolder(new LayerKernel<T>());
			auto vectorKernel = vectorKernelHolder.get();
			vectorKernel->SetKernelName("AccumulateVectorKernel");
			unique_ptr<LayerKernel<T>> scalarKernelHolder(new LayerKernel<T>());
			auto scalarKernel = scalarKernelHolder.get();
			scalarKernel->SetKernelName("AccumulateVectorWithScalarKernel");

			auto deviceInfo = device->DeviceInfo();
			if (is_same<cl_double, T>::value) 
				gradientProgram->AddDefine("DOUBLE_PRECISION");

			T currentStepSize = -stepSizeCallback(0);

			gradientProgram->AttachKernel(move(vectorKernelHolder));
			gradientProgram->AttachKernel(move(scalarKernelHolder));

			contexts[0]->AttachProgram(move(gradientProgram), oneDeviceVector);

			scalarKernel->SetRealArg(currentStepSize, 2);

			vector<unique_ptr<OCLMemory>> inputMemories;

			//TODO: Only support one format index at the moment
			//First the forward propagation phase, we need to save all the memory.
			LayerMemoryDescription inputMemoryDescription =
				this->InputForwardMemoryDescriptions()[0];

			LayerMemoryDescription inBackPropMemoryDescription =
				this->OutputForwardMemoryDescriptions()[0];
			LayerMemoryDescription outBackPropMemoryDescription =
				outputLayer->OutBackPropMemoryDescriptions()[0];

			unique_ptr<OCLMemory> inputMemory = contexts[0]->CreateMemory(
				CL_MEM_READ_ONLY, sizeof(T) * inputMemoryDescription.TotalMemory());

			unique_ptr<OCLMemory> targetMemory = contexts[0]->CreateMemory(
				CL_MEM_READ_ONLY,
				sizeof(T) * inBackPropMemoryDescription.TotalMemory());

			inputMemories.push_back(move(inputMemory));
			for (auto& layer : layers) {
				inputMemoryDescription = layer->OutForwardPropMemoryDescriptions()[0];
				unique_ptr<OCLMemory> outputMemory = contexts[0]->CreateMemory(
					CL_MEM_READ_WRITE,
					sizeof(T) * inputMemoryDescription.TotalMemory());
				inputMemories.push_back(move(outputMemory));
			}

			size_t layerCount = layers.size();
			//Allocate memory for the gradients
			vector<vector<unique_ptr<OCLMemory>>> gradients;
			vector<vector<OCLMemory*>> gradientsPointers;
			vector<vector<unique_ptr<OCLMemory>>> accumulatedGradients;
			vector<vector<OCLMemory*>> accumulatedGradientsPointers;

			for (size_t i = 0; i < layerCount; i++) 
			{
				gradients.push_back(vector<unique_ptr<OCLMemory>>());
				accumulatedGradients.push_back(vector<unique_ptr<OCLMemory>>());
				gradientsPointers.push_back(vector<OCLMemory*>());
				accumulatedGradientsPointers.push_back(vector<OCLMemory*>());

				vector<size_t> parameterCount = layers[i]->GetMultipleParameterCount();
				for (size_t k = 0; k < parameterCount.size(); k++) {
					unique_ptr<OCLMemory> tempGradientMemory1 =
						contexts[0]->CreateMemory(CL_MEM_READ_WRITE,
						sizeof(T) * parameterCount[k]);
					unique_ptr<OCLMemory> tempGradientMemory2 =
						contexts[0]->CreateMemory(CL_MEM_READ_WRITE,
						sizeof(T) * parameterCount[k]);

					gradientsPointers[i].push_back(tempGradientMemory1.get());
					gradients[i].push_back(move(tempGradientMemory1));
					accumulatedGradientsPointers[i].push_back(
						tempGradientMemory2.get());
					accumulatedGradients[i].push_back(move(tempGradientMemory2));
				}
			}

			int batchIterations = samplesPerEpoch / batchSize;
			//TODO:
			//int remainder = samplesPerEpoch % batchSize;
			bool stopped = false;

			for (int epoch = 0; epoch < epochs; epoch++) 
			{
				trainer->EpochStarted();
				T stepSize = -stepSizeCallback(0);
				for (int batch = 0; batch < batchIterations; batch++) 
				{
					trainer->BatchStarted();
					for (int sample = 0; sample < batchSize; sample++) 
					{
						if (trainer->Stopping()) 
						{
							stopped = true;
							break;
						}

						T* input;
						T* target;
						int formatIndex;

						int dataID = trainer->DataIDRequest();

						//Assuming aligned memory
						trainer->MapInputAndTarget(dataID, input, target, formatIndex);

						if (formatIndex != 0)
							throw runtime_error(
							"Other format indices than 0 is not supported at the moment");

						device->WriteMemory(inputMemories[0].get(),
							inputMemories[0]->ByteSize(), input, 0, false);

						device->WriteMemory(targetMemory.get(),
							targetMemory->ByteSize(), target, 0, false);

						trainer->UnmapInputAndTarget(dataID, input, target, formatIndex);

						for (size_t i = 0; i < layerCount; i++)
							layers[i]->EnqueueForwardPropagation(device, 0,
							inputMemories[i].get(), inputMemories[i + 1].get(),
							false);

						device->WaitForDeviceQueue(0);

						unique_ptr<OCLMemory> backPropOutputMemory =
							contexts[0]->CreateMemory(CL_MEM_READ_WRITE, sizeof(T) * outBackPropMemoryDescription.TotalMemory());

						outputLayer->EnqueueBackPropagation(device, 0,
							inputMemories[inputMemories.size() - 1].get(),
							targetMemory.get(), backPropOutputMemory.get(), true);

						if (layerCount != 0) 
						{
							layers[layerCount - 1]->EnqueueCalculateGradient(device, 0,
								inputMemories[inputMemories.size() - 2].get(),
								backPropOutputMemory.get(),
								gradientsPointers[layerCount - 1], true);
						}

						for (int i = static_cast<int>(layerCount) - 1; i >= 1; i--) 
						{
							inBackPropMemoryDescription =
								layers[i]->OutBackPropMemoryDescriptions()[formatIndex];

							unique_ptr<OCLMemory> outputMemory =
								contexts[0]->CreateMemory(CL_MEM_READ_WRITE,
								sizeof(T)
								* inBackPropMemoryDescription.TotalMemory());
							layers[i]->EnqueueBackPropagation(device, 0,
								inputMemories[i].get(), backPropOutputMemory.get(),
								outputMemory.get(), true);
							backPropOutputMemory.reset();
							backPropOutputMemory = move(outputMemory);

							layers[i - 1]->EnqueueCalculateGradient(device, 0,
								inputMemories[i - 1].get(),
								backPropOutputMemory.get(),
								gradientsPointers[i - 1], true);
						}

						device->WaitForDeviceQueue(0);

						//If this is the first sample, add the memory to the accumulator and continue
						if (sample == 0) 
						{
							for (size_t i = 0; i < layerCount; i++) 
							{
								for (size_t j = 0; j < gradientsPointers[i].size(); j++) 
								{
									if (gradientsPointers[i][j]->ByteSize()
										== accumulatedGradientsPointers[i][j]->ByteSize())
										device->CopyCLMemory(gradientsPointers[i][j],
										accumulatedGradientsPointers[i][j], 0,
										0, gradientsPointers[i][j]->ByteSize(),
										0, false);
									else
										throw runtime_error(
										"The memory does not match for the accumulated gradient.");
								}
							}

							device->WaitForDeviceQueue(0);
						} 
						else 
						{
							for (size_t i = 0; i < layerCount; i++) 
							{
								vector<size_t> parameterCount =
									layers[i]->GetMultipleParameterCount();
								for (size_t j = 0; j < gradientsPointers[i].size(); j++) {
									vectorKernel->SetMemoryArg(
										accumulatedGradientsPointers[i][j], 1);
									vectorKernel->SetMemoryArg(gradientsPointers[i][j], 0);
									vectorKernel->ClearGlobalSizes();
									vectorKernel->AddGlobalSize(parameterCount[j]);
									device->ExecuteKernel(vectorKernel, 0, false);
								}
							}

							device->WaitForDeviceQueue(0);
						}
					}

					if (stopped)
						break;

					//Fetch the parameters from the layers, accumulate with scalar and continue
					if (stepSize != currentStepSize) 
					{
						scalarKernel->SetRealArg(stepSize, 2);
						currentStepSize = stepSize;
					}

					for (size_t i = 0; i < layerCount; i++) 
					{
						vector<OCLMemory*> parametersToUpdate =
							layers[i]->GetParameters();
						vector<size_t> parameterCount =
							layers[i]->GetMultipleParameterCount();

						if (parametersToUpdate.size()
							!= accumulatedGradientsPointers[i].size())
							throw runtime_error(
							"The accumulated gradients do not match the layer parameters");

						for (size_t j = 0; j < accumulatedGradientsPointers[i].size(); j++) 
						{
							if (parametersToUpdate[j]->ByteSize()
								!= accumulatedGradientsPointers[i][j]->ByteSize())
								throw runtime_error(
								"The gradient and the parameter memories does not match");

							scalarKernel->SetMemoryArg(
								accumulatedGradientsPointers[i][j], 0);
							scalarKernel->SetMemoryArg(
								parametersToUpdate[j], 1);
							scalarKernel->ClearGlobalSizes();
							scalarKernel->AddGlobalSize(parameterCount[j]);
							device->ExecuteKernel(scalarKernel, 0, false);
						}
					}

					device->WaitForDeviceQueue(0);
					trainer->BatchFinished(-1);
				}

				trainer->EpochFinished();

				//TODO:
				//for (int k = 0; k < remainder; k++)
				//	throw runtime_error(
				//	"Non dividable batch size is not implemented yet");

				if (stopped)
					break;
			}
			contexts[0]->DetachProgram(programPointer);
		}

		template<class T>
		vector<OCLForwardBackPropLayer<T>*> OCLConvNet<T>::GetLayers() const 
		{
			vector<OCLForwardBackPropLayer<T>*> result;
			for (auto& layer : layers)
				result.push_back(layer.get());

			return result;
		}

		template<class T>
		StandardOutputLayer<T>* OCLConvNet<T>::GetOutputLayer() const 
		{
			return outputLayer.get();
		}

		template class OCLConvNet<cl_float> ;
		template class OCLConvNet<cl_double> ;

	} /* namespace MachineLearning */
} /* namespace Matuna */
