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

#include <stdexcept>
#include <tuple>
#include <unordered_map>
#include <type_traits>

namespace Matuna {
	namespace MachineLearning {

		template<class T>
		OCLConvNet<T>::OCLConvNet(const vector<OCLDeviceInfo>& devices,
			unique_ptr<ConvNetConfig> config) :
		TrainableConvNet<T>(*config) {

			InitializeContexts(devices);

			//TODO: At the moment we only support one context
			OCLConvNetFactoryVisitor<T> factory(contexts[0], this);
			config->Accept(&factory);
			auto createdLayers = factory.GetLayers();
			auto createdOutputLayer = factory.GetOutputLayer();

			//Transfer ownership to this class
			for (auto& layer : createdLayers) {
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

		template<class T>
		void OCLConvNet<T>::InitializeContexts(
			const vector<OCLDeviceInfo>& deviceInfos) {


				static_assert(is_same<cl_double, T>::value || is_same<cl_float, T>::value, "The type is not supported");

				unordered_map<cl_platform_id, tuple<OCLPlatformInfo, vector<OCLDeviceInfo>> > platformsAndDevices;
				for (auto& deviceInfo : deviceInfos) {
					if (!deviceInfo.DeviceAvailable())
						throw invalid_argument("The device is not available");

					if (!deviceInfo.CompilerAvailable())
						throw invalid_argument(
						"The device does not have an available compiler");

					CheckPrecision<is_same<cl_double, T>::value>::Check(deviceInfo);

					auto platformInfo = deviceInfo.PlatformInfo();
					auto platformID = platformInfo.PlatformID();
					if (platformsAndDevices.find(platformID) == platformsAndDevices.end()) {
						vector<OCLDeviceInfo> infos;
						infos.push_back(deviceInfo);
						platformsAndDevices.insert(
							make_pair(platformID, make_tuple(platformInfo, infos)));
					} else {
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

				vector<
					tuple<OCLPlatformInfo,
					vector<tuple<OCLDeviceConfig, OCLDeviceInfo>>> > contextsConfigurations;
				for (auto& platformAndDevices : platformsAndDevices) {
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
		vector<OCLContext*> OCLConvNet<T>::GetOCLContexts() const {
			vector<OCLContext*> result;
			for (auto& context : contexts)
				result.push_back(context.get());

			return result;
		}

		template<class T>
		unique_ptr<T[]> OCLConvNet<T>::FeedForwardAligned(T* input, int formatIndex) {
			//TODO: At the moment we only support one context and one device
			auto devices = contexts[0]->GetDevices();
			auto device = devices[0];
			LayerMemoryDescription inputMemoryDescription =
				this->InputForwardMemoryDescriptions()[formatIndex];

			unique_ptr<OCLMemory> inputMemory = contexts[0]->CreateMemory(
				CL_MEM_READ_ONLY, sizeof(T) * inputMemoryDescription.TotalMemory());
			device->WriteMemory(inputMemory.get(), inputMemory->ByteSize(), input, 0,
				true);

			//We need some synchronization in order not to use blocking calls.
			//We could probably yield some better performance in that case.
			for (auto& layer : layers) {
				inputMemoryDescription =
					layer->OutForwardPropMemoryDescriptions()[formatIndex];
				unique_ptr<OCLMemory> outputMemory = contexts[0]->CreateMemory(
					CL_MEM_READ_WRITE,
					sizeof(T) * inputMemoryDescription.TotalMemory());
				layer->EnqueueForwardPropagation(device, 0, inputMemory.get(),
					outputMemory.get(), true);
				inputMemory.reset();
				inputMemory = move(outputMemory);
			}

			unique_ptr<T[]> output(new T[inputMemoryDescription.TotalMemory()]);
			device->ReadMemory(inputMemory.get(), inputMemory->ByteSize(), output.get(),
				0, true);

			return move(output);
		}

		template<class T>
		T OCLConvNet<T>::CalculateErrorAligned(T* input, int formatIndex, T* target) {
			//TODO: At the moment we only support one context and one device
			auto devices = contexts[0]->GetDevices();
			auto device = devices[0];
			LayerMemoryDescription inputMemoryDescription =
				this->InputForwardMemoryDescriptions()[formatIndex];

			unique_ptr<OCLMemory> inputMemory = contexts[0]->CreateMemory(
				CL_MEM_READ_ONLY, sizeof(T) * inputMemoryDescription.TotalMemory());
			device->WriteMemory(inputMemory.get(), inputMemory->ByteSize(), input, 0,
				true);

			//We need some synchronization in order not to use blocking calls.
			//We could probably yield some better performance in that case.
			for (auto& layer : layers) {
				inputMemoryDescription =
					layer->OutForwardPropMemoryDescriptions()[formatIndex];
				unique_ptr<OCLMemory> outputMemory = contexts[0]->CreateMemory(
					CL_MEM_READ_WRITE,
					sizeof(T) * inputMemoryDescription.TotalMemory());
				layer->EnqueueForwardPropagation(device, 0, inputMemory.get(),
					outputMemory.get(), true);
				inputMemory.reset();
				inputMemory = move(outputMemory);
			}

			LayerMemoryDescription inBackPropMemoryDescription =
				this->OutputForwardMemoryDescriptions()[formatIndex];
			unique_ptr<OCLMemory> targetMemory = contexts[0]->CreateMemory(
				CL_MEM_READ_ONLY,
				sizeof(T) * inBackPropMemoryDescription.TotalMemory());
			device->WriteMemory(targetMemory.get(), targetMemory->ByteSize(), target, 0,
				true);

			return this->outputLayer->CalculateError(device, 0, inputMemory.get(),
				targetMemory.get());
		}

		template<class T>
		unique_ptr<T[]> OCLConvNet<T>::BackPropAligned(T* input, int formatIndex,
			T* target) {
				//TODO: At the moment we only support one context and one device
				auto devices = contexts[0]->GetDevices();
				auto device = devices[0];

				vector<unique_ptr<OCLMemory>> inputMemories;
				//First the forward propagation phase, we need to save all the memory.
				LayerMemoryDescription inputMemoryDescription =
					this->InputForwardMemoryDescriptions()[formatIndex];

				unique_ptr<OCLMemory> inputMemory = contexts[0]->CreateMemory(
					CL_MEM_READ_ONLY, sizeof(T) * inputMemoryDescription.TotalMemory());
				device->WriteMemory(inputMemory.get(), inputMemory->ByteSize(), input, 0,
					false);

				inputMemories.push_back(move(inputMemory));
				for (auto& layer : layers) {
					inputMemoryDescription =
						layer->OutForwardPropMemoryDescriptions()[formatIndex];
					unique_ptr<OCLMemory> outputMemory = contexts[0]->CreateMemory(
						CL_MEM_READ_WRITE,
						sizeof(T) * inputMemoryDescription.TotalMemory());
					inputMemories.push_back(move(outputMemory));
				}

				auto count = layers.size();

				for (size_t i = 0; i < count; i++)
					layers[i]->EnqueueForwardPropagation(device, 0, inputMemories[i].get(),
					inputMemories[i + 1].get(), false);

				device->WaitForDeviceQueue(0);

				LayerMemoryDescription inBackPropMemoryDescription =
					this->OutputForwardMemoryDescriptions()[formatIndex];

				unique_ptr<OCLMemory> targetMemory = contexts[0]->CreateMemory(
					CL_MEM_READ_ONLY,
					sizeof(T) * inBackPropMemoryDescription.TotalMemory());
				device->WriteMemory(targetMemory.get(), targetMemory->ByteSize(), target, 0,
					true);

				inBackPropMemoryDescription =
					outputLayer->OutBackPropMemoryDescriptions()[formatIndex];
				unique_ptr<OCLMemory> backPropOutputMemory = contexts[0]->CreateMemory(
					CL_MEM_READ_ONLY,
					sizeof(T) * inBackPropMemoryDescription.TotalMemory());

				outputLayer->EnqueueBackPropagation(device, 0,
					inputMemories[inputMemories.size() - 1].get(), targetMemory.get(),
					backPropOutputMemory.get(), true);
				targetMemory.reset();

				for (int i = static_cast<int>(count) - 1; i >= 1; i--) {
					inBackPropMemoryDescription =
						layers[i]->OutBackPropMemoryDescriptions()[formatIndex];
					unique_ptr<OCLMemory> outputMemory = contexts[0]->CreateMemory(
						CL_MEM_READ_WRITE,
						sizeof(T) * inBackPropMemoryDescription.TotalMemory());
					layers[i]->EnqueueBackPropagation(device, 0, inputMemories[i].get(),
						backPropOutputMemory.get(), outputMemory.get(), true);
					backPropOutputMemory.reset();
					backPropOutputMemory = move(outputMemory);
				}

				device->WaitForDeviceQueue(0);

				LayerMemoryDescription outBackPropMemoryDescription =
					this->OutputBackMemoryDescriptions()[formatIndex];

				unique_ptr<T[]> output(new T[outBackPropMemoryDescription.TotalMemory()]);
				device->ReadMemory(backPropOutputMemory.get(),
					backPropOutputMemory->ByteSize(), output.get(), 0, true);

				return move(output);
		}

		template<class T>
		T OCLConvNet<T>::CalculateErrorFromForwardAligned(T* propagatedValue,
			int formatIndex, T* target) {
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
		unique_ptr<T[]> OCLConvNet<T>::CalculateGradientAligned(T* input,
			int formatIndex, T* target) {
				//TODO: At the moment we only support one context and one device
				auto devices = contexts[0]->GetDevices();
				auto device = devices[0];

				vector<unique_ptr<OCLMemory>> inputMemories;
				//First the forward propagation phase, we need to save all the memory.
				LayerMemoryDescription inputMemoryDescription =
					this->InputForwardMemoryDescriptions()[formatIndex];

				unique_ptr<OCLMemory> inputMemory = contexts[0]->CreateMemory(
					CL_MEM_READ_ONLY, sizeof(T) * inputMemoryDescription.TotalMemory());
				device->WriteMemory(inputMemory.get(), inputMemory->ByteSize(), input, 0,
					false);

				inputMemories.push_back(move(inputMemory));
				for (auto& layer : layers) {
					inputMemoryDescription =
						layer->OutForwardPropMemoryDescriptions()[formatIndex];
					unique_ptr<OCLMemory> outputMemory = contexts[0]->CreateMemory(
						CL_MEM_READ_WRITE,
						sizeof(T) * inputMemoryDescription.TotalMemory());
					inputMemories.push_back(move(outputMemory));
				}

				auto count = layers.size();

				for (size_t i = 0; i < count; i++)
					layers[i]->EnqueueForwardPropagation(device, 0, inputMemories[i].get(),
					inputMemories[i + 1].get(), false);

				device->WaitForDeviceQueue(0);

				//Allocate the necessary memory for the gradient
				unique_ptr<T[]> gradient(new T[GetParameterCount()]);

				LayerMemoryDescription inBackPropMemoryDescription =
					this->OutputForwardMemoryDescriptions()[formatIndex];

				unique_ptr<OCLMemory> targetMemory = contexts[0]->CreateMemory(
					CL_MEM_READ_ONLY,
					sizeof(T) * inBackPropMemoryDescription.TotalMemory());
				device->WriteMemory(targetMemory.get(), targetMemory->ByteSize(), target, 0,
					true);

				inBackPropMemoryDescription =
					outputLayer->OutBackPropMemoryDescriptions()[formatIndex];
				unique_ptr<OCLMemory> backPropOutputMemory = contexts[0]->CreateMemory(
					CL_MEM_READ_ONLY,
					sizeof(T) * inBackPropMemoryDescription.TotalMemory());

				outputLayer->EnqueueBackPropagation(device, 0,
					inputMemories[inputMemories.size() - 1].get(), targetMemory.get(),
					backPropOutputMemory.get(), true);
				targetMemory.reset();

				size_t gradientMemoryPosition = 0;
				if (layers.size() != 0) 
				{
					//Allocate memory for the gradient
					auto parametersCount = layers[count - 1]->GetMultipleParameterCount();

					vector<unique_ptr<OCLMemory>> gradientMemories;
					vector<OCLMemory*> gradientsPointers;
					for (auto parameterCount : parametersCount)
					{
						unique_ptr<OCLMemory> gradientMemory = contexts[0]->CreateMemory(
							CL_MEM_WRITE_ONLY, sizeof(T) * parameterCount);
						gradientsPointers.push_back(gradientMemory.get());
						gradientMemories.push_back(move(gradientMemory));
					}

					layers[count - 1]->EnqueueCalculateGradient(device, 0,
						inputMemories[inputMemories.size() - 2].get(),
						backPropOutputMemory.get(), gradientsPointers, true);

					//Write gradient memory to the host buffer
					auto rawGradient = gradient.get();
					rawGradient += gradientMemoryPosition;
					for (size_t k = 0; k < parametersCount.size(); k++)
					{
						device->ReadMemory(gradientsPointers[k], gradientsPointers[k]->ByteSize(),
							rawGradient, 0, true);
						gradientMemoryPosition += parametersCount[k];
						rawGradient = gradient.get() + gradientMemoryPosition;
					}
				}

				for (int i = static_cast<int>(count) - 1; i >= 1; i--) 
				{
					inBackPropMemoryDescription =
						layers[i]->OutBackPropMemoryDescriptions()[formatIndex];

					unique_ptr<OCLMemory> outputMemory = contexts[0]->CreateMemory(
						CL_MEM_READ_WRITE,
						sizeof(T) * inBackPropMemoryDescription.TotalMemory());
					layers[i]->EnqueueBackPropagation(device, 0, inputMemories[i].get(),
						backPropOutputMemory.get(), outputMemory.get(), true);
					backPropOutputMemory.reset();
					backPropOutputMemory = move(outputMemory);

					//Allocate memory for the gradient
					auto parametersCount = layers[i- 1]->GetMultipleParameterCount();
					vector<unique_ptr<OCLMemory>> gradientMemories;
					vector<OCLMemory*> gradientsPointers;
					for (auto parameterCount : parametersCount)
					{
						unique_ptr<OCLMemory> gradientMemory = contexts[0]->CreateMemory(
							CL_MEM_WRITE_ONLY, sizeof(T) * parameterCount);
						gradientsPointers.push_back(gradientMemory.get());
						gradientMemories.push_back(move(gradientMemory));
					}

					layers[i - 1]->EnqueueCalculateGradient(device, 0,
						inputMemories[i - 1].get(), backPropOutputMemory.get(),
						gradientsPointers, true);

					//Write gradient memory to the host buffer
					auto rawGradient = gradient.get();
					rawGradient += gradientMemoryPosition;
					for (size_t k = 0; k < parametersCount.size(); k++)
					{
						device->ReadMemory(gradientsPointers[k], gradientsPointers[k]->ByteSize(),
							rawGradient, 0, true);
						gradientMemoryPosition += parametersCount[k];
						rawGradient = gradient.get() + gradientMemoryPosition;
					}
				}

				device->WaitForDeviceQueue(0);

				return move(gradient);
		}

		template<class T>
		unique_ptr<T[]> OCLConvNet<T>::GetParameters() {
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
		void OCLConvNet<T>::SetParameters(T* parameters) {
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
		size_t OCLConvNet<T>::GetParameterCount() {
			size_t result = 0;
			for (auto& layer : layers)
				result += layer->GetParameterCount();

			return result;
		}

		template<class T>
		void OCLConvNet<T>::TrainNetwork(unique_ptr<ConvNetTrainer<T>> trainer,
			unique_ptr<IAlgorithmConfig> algorithm) {
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

				for (size_t i = 0; i < layerCount; i++) {
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

				for (int epoch = 0; epoch < epochs; epoch++) {
					trainer->EpochStarted();
					T stepSize = -stepSizeCallback(0);
					for (int batch = 0; batch < batchIterations; batch++) {
						trainer->BatchStarted();
						for (int sample = 0; sample < batchSize; sample++) {
							if (trainer->Stopping()) {
								stopped = true;
								break;
							}

							T* input;
							T* target;
							int formatIndex;

							//Assuming aligned memory
							trainer->MapInputAndTarget(input, target, formatIndex);

							if (formatIndex != 0)
								throw runtime_error(
								"Other format indices than 0 is not supported at the moment");

							device->WriteMemory(inputMemories[0].get(),
								inputMemories[0]->ByteSize(), input, 0, false);

							device->WriteMemory(targetMemory.get(),
								targetMemory->ByteSize(), target, 0, false);

							trainer->UnmapInputAndTarget(input, target, formatIndex);

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

							if (layerCount != 0) {
								layers[layerCount - 1]->EnqueueCalculateGradient(device, 0,
									inputMemories[inputMemories.size() - 2].get(),
									backPropOutputMemory.get(),
									gradientsPointers[layerCount - 1], true);
							}

							for (int i = static_cast<int>(layerCount) - 1; i >= 1; i--) {
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
							if (sample == 0) {
								for (size_t i = 0; i < layerCount; i++) {
									for (size_t j = 0; j < gradientsPointers[i].size(); j++) {
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
							} else {
								for (size_t i = 0; i < layerCount; i++) {
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
						if (stepSize != currentStepSize) {
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
		vector<OCLForwardBackPropLayer<T>*> OCLConvNet<T>::GetLayers() const {
			vector<OCLForwardBackPropLayer<T>*> result;
			for (auto& layer : layers)
				result.push_back(layer.get());

			return result;
		}

		template<class T>
		StandardOutputLayer<T>* OCLConvNet<T>::GetOutputLayer() const {
			return outputLayer.get();
		}

		template class OCLConvNet<cl_float> ;
		template class OCLConvNet<cl_double> ;

	} /* namespace MachineLearning */
} /* namespace Matuna */
