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
#include "Matuna.ConvNet/GradientDescentConfig.h"

#include "Matuna.OCLHelper/OCLDeviceInfo.h"
#include "Matuna.OCLHelper/OCLContext.h"
#include "Matuna.OCLHelper/OCLDevice.h"
#include "Matuna.OCLHelper/OCLMemory.h"

#include "OCLForwardBackPropLayer.h"
#include "StandardOutputLayer.h"

#include <memory>
#include <vector>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <stdexcept>
#include <tuple>
#include <unordered_map>
#include <type_traits>
#include <queue>

using namespace std;
using namespace Matuna::Helper;

namespace Matuna
{
	namespace MachineLearning
	{
		//The purpose of this class is simply to wrap a memory object that can be destroyed.
		//This is completely transparent to the user.
		//The only thing to be careful about is to make sure this object always lives long enough for the pointer to be useful.
		class InputDataWrapper
		{
		private:
			unique_ptr<OCLMemory> inputHandledMemory;
			OCLMemory* inputMemoryPointer;
			unique_ptr<OCLMemory> targetHandledMemory;
			OCLMemory* targetMemoryPointer;
			int formatIndex;

		public:
			InputDataWrapper(unique_ptr<OCLMemory> inputHandledMemory, unique_ptr<OCLMemory> targetHandledMemory, int formatIndex)
			{
				this->formatIndex = formatIndex;
				this->inputHandledMemory = move(inputHandledMemory);
				this->inputMemoryPointer = inputHandledMemory.get();
				this->targetHandledMemory = move(targetHandledMemory);
				this->targetMemoryPointer = targetHandledMemory.get();
			};

			InputDataWrapper(OCLMemory* inputMemoryPointer, OCLMemory* targetMemoryPointer, int formatIndex)
			{
				this->formatIndex = formatIndex;
				this->inputMemoryPointer = inputMemoryPointer;
				this->targetMemoryPointer = targetMemoryPointer;
			};

			InputDataWrapper(InputDataWrapper&& original)
			{
				this->formatIndex = original.formatIndex;
				this->inputHandledMemory = move(original.inputHandledMemory);
				this->inputMemoryPointer = original.inputMemoryPointer;
				this->targetHandledMemory = move(original.targetHandledMemory);
				this->targetMemoryPointer = original.targetMemoryPointer;
			}

			OCLMemory* GetInput() const { return inputMemoryPointer; };

			OCLMemory* GetTarget() const { return targetMemoryPointer; };

			int FormatIndex() const { return formatIndex;};
		};

		//The purpose of this class it to allow thread-safe insertions into the input data buffer
		class InputDataBufferQueue
		{
		private:
			queue<int> dataIDQueue;
			//First id, second tuple: reference, formatindex, input, target
			unordered_map<int, tuple<int, int, unique_ptr<OCLMemory>, unique_ptr<OCLMemory>>> idAndInputData;
			size_t maxBufferSize;

			condition_variable notFull;
			condition_variable notEmpty;

			mutex mut;

		public:
			InputDataBufferQueue(size_t maxBufferSize)
			{
				if (maxBufferSize == 0)
					throw invalid_argument("The buffer size has to be at least of size 1");

				this->maxBufferSize = maxBufferSize;
			};

			InputDataWrapper Pop()
			{
				unique_lock<mutex> lock(mut);

				//Wait for not empty event
				notEmpty.wait(lock, [this](){ return dataIDQueue.size() != 0;});

				if (dataIDQueue.size() == 0)
					throw runtime_error("You cannot pop from an empty queue");

				if (dataIDQueue.size() > maxBufferSize)
					throw runtime_error("The internal queue is exceeding the buffer size");

				if (idAndInputData.size() > dataIDQueue.size())
					throw runtime_error("The resource holder contains more elements than the amount of ids in the queue");

				int dataID = dataIDQueue.front();
				dataIDQueue.pop();

				if (idAndInputData.find(dataID) == idAndInputData.end())
					throw runtime_error("We cannot have ID in the queue that is not located in the hash map");

				auto& referenceDataTuple = idAndInputData[dataID];

				if (get<0>(referenceDataTuple) == 1)
				{
					InputDataWrapper result(move(get<2>(referenceDataTuple)), move(get<3>(referenceDataTuple)), get<1>(referenceDataTuple));
					idAndInputData.erase(dataID);
					notFull.notify_one();
					return move(result);
				}
				else if(get<0>(referenceDataTuple) > 1)
				{
					InputDataWrapper result(get<2>(referenceDataTuple).get(), get<3>(referenceDataTuple).get(), get<1>(referenceDataTuple));
					get<0>(referenceDataTuple) = get<0>(referenceDataTuple) - 1;
					notFull.notify_one();
					return move(result);
				}
				else
					throw runtime_error("The memory references inside the the id hash map is invalid");
			};

			//This functions pushes an ID to the queue. This is to happen if we already have a reference on the data
			//If the return value is false, this means that it has failed. This can occur if Pop() was called previously, erasing the data
			bool Push(int dataID)
			{
				unique_lock<mutex> lock(mut);

				//Wait until the buffer is not full
				notFull.wait(lock, [this](){ return dataIDQueue.size() != maxBufferSize;});

				if (dataIDQueue.size() > maxBufferSize)
					throw runtime_error("The internal queue is exceeding the buffer size");

				if (idAndInputData.size() > dataIDQueue.size())
					throw runtime_error("The resource holder contains more elements than the amount of ids in the queue");


				if (idAndInputData.find(dataID) == idAndInputData.end())
					return false;

				dataIDQueue.push(dataID);
				auto& referenceDataTuple = idAndInputData[dataID];
				get<0>(referenceDataTuple) = get<0>(referenceDataTuple) + 1;

				notEmpty.notify_one();

				return true;
			}

			//This function should only be called if we don't have a reference of the dataID inside the buffer
			//If we have it, an exception will be thrown
			void Push(int dataID, int formatIndex, unique_ptr<OCLMemory> inputMemory, unique_ptr<OCLMemory> targetMemory)
			{
				unique_lock<mutex> lock(mut);

				//Wait until the buffer is not full
				notFull.wait(lock, [this](){ return dataIDQueue.size() != maxBufferSize;});

				if (dataIDQueue.size() > maxBufferSize)
					throw runtime_error("The internal queue is exceeding the buffer size");

				if (idAndInputData.size() > dataIDQueue.size())
					throw runtime_error("The resource holder contains more elements than the amount of ids in the queue");

				if (idAndInputData.find(dataID) != idAndInputData.end())
					throw runtime_error("The data has already been added to the buffer");

				idAndInputData.insert(make_pair(dataID, make_tuple(1, formatIndex, move(inputMemory), move(targetMemory))));
				dataIDQueue.push(dataID);

				notEmpty.notify_one();
			}

			void Clear()
			{
				unique_lock<mutex> lock(mut);
				idAndInputData.clear();
				queue<int> empty;
				swap(dataIDQueue, empty);
				notFull.notify_one();
			};

			//If this function returns 0, add the actual memory. Else increase the reference count of the dataID
			int ReferenceCount(int dataID)
			{
				unique_lock<mutex> lock(mut);
				if (idAndInputData.find(dataID) == idAndInputData.end())
					return 0;
				else
					return get<0>(idAndInputData[dataID]);
			};
		};

		template<class T>
		class OCLConvNet final: public TrainableConvNet<T>
		{

		private:
			vector<shared_ptr<OCLContext>> contexts;
			vector<unique_ptr<OCLForwardBackPropLayer<T>>> layers;
			unique_ptr<StandardOutputLayer<T>> outputLayer;

			unique_ptr<InputDataBufferQueue> inputDataBufferQueue;
			bool trainingIsRunning;

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

			void TrainNetwork2(unique_ptr<ConvNetTrainer<T>> trainer, unique_ptr<IAlgorithmConfig> algorithm);
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

			void TrainNetworkGDLowMemory(unique_ptr<GradientDescentConfig<T>> gdConfig, unique_ptr<ConvNetTrainer<T>> trainer);
			void ReadInputDataAsync(ConvNetTrainer<T>* trainer);
		};

	} /* namespace MachineLearning */
} /* namespace Matuna */

#endif /* MATUNA_MATUNA_OCLCONVNET_OCLCONVNET_H_ */
