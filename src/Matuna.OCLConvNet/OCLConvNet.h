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
			OCLMemory* inputMemoryPointer;
			OCLMemory* targetMemoryPointer;
			int formatIndex;

		public:
			InputDataWrapper(OCLMemory* inputMemoryPointer, OCLMemory* targetMemoryPointer, int formatIndex)
			{
				this->formatIndex = formatIndex;
				this->inputMemoryPointer = inputMemoryPointer;
				this->targetMemoryPointer = targetMemoryPointer;
			};

			OCLMemory* GetInput() const { return inputMemoryPointer; };

			OCLMemory* GetTarget() const { return targetMemoryPointer; };

			int FormatIndex() const { return formatIndex;};
		};

		//The purpose of this class it to allow thread-safe insertions into the input data buffer
		class InputDataBufferQueue
		{
		private:

			condition_variable notFull;
			condition_variable notEmpty;

			unordered_map<int, unique_ptr<OCLMemory>> idAndInputMemory;
			unordered_map<int, unique_ptr<OCLMemory>> idAndTargetMemory;
			unordered_map<int, int> idAndFormats;
			unordered_map<int, int> idAndReferences;

			vector<int> dataIDBuffer;
			bool acquiredCalled;
			int bufferSize;
			int readPosition;
			int writePosition;
			int count;
			mutex mute;

		public:
			InputDataBufferQueue(int maxBufferSize)
			{
				if (maxBufferSize == 0)
					throw invalid_argument("The buffer size has to be at least of size 1");

				this->bufferSize = maxBufferSize;
				readPosition = 0;
				writePosition = 0;
				count = 0;
				acquiredCalled = false;
			};

			InputDataWrapper LockAndAcquire()
			{
				acquiredCalled = true;
				unique_lock<mutex> lock(mute);

				//Wait for not empty event
				notEmpty.wait(lock, [this](){ return count != 0;});

				int dataID = dataIDBuffer[readPosition];
				if (idAndReferences.find(dataID) == idAndReferences.end())
					throw runtime_error("We must have a ID in the buffer we are acquiring from");

				//printf("Lock acquire, id: %i, read position: %i \n", dataID, readPosition);

				InputDataWrapper result(idAndInputMemory[dataID].get(), idAndTargetMemory[dataID].get(), idAndFormats[dataID]);

				return result;
			}

			void UnlockAcquire()
			{
				unique_lock<mutex> lock(mute);

				if (!acquiredCalled)
					throw runtime_error("You cannot unlock and acquire if you have not acquired");

				//printf("Unlock acquire, read position: %i \n", readPosition);

				acquiredCalled = false;
				readPosition = (readPosition + 1) % bufferSize;
				count--;
				lock.unlock();
				notFull.notify_one();
			}

			void Push(int dataID)
			{
				unique_lock<mutex> lock(mute);

				//Wait until the buffer is not full
				notFull.wait(lock, [this](){ return count != bufferSize;});

				//printf("Pushing id: %i, write position: %i \n", dataID, writePosition);

				if (idAndReferences.find(dataID) == idAndReferences.end())
					throw invalid_argument("We must have a reference count to the memory");

				bool increaseReferenceCount = true;
				//If the buffer is being filled for the first time, there's nothing to overwrite
				if (static_cast<int>(dataIDBuffer.size()) <= writePosition)
					dataIDBuffer.push_back(dataID);
				else
				{
					auto overwriteID = dataIDBuffer[writePosition];
					//If the reference count is 1, we may delete the OCL memory
					auto idAndReference = idAndReferences.find(overwriteID);

					if (idAndReference == idAndReferences.end())
						throw runtime_error("We must have a reference count of the IDs in the data buffer");

					if(idAndReference->second < 1)
						throw runtime_error("We cannot have undeleted references that have 0 reference count");

					//One important case is if the id we are overwriting is the same as the one we are pushing.
					//If this is the case, the memory holders and references should remain unchanged since
					//we are effectively removing and adding the same data
					if (overwriteID != dataID)
					{
						if (idAndReference->second == 1)
						{
							idAndReferences.erase(overwriteID);
							idAndFormats.erase(overwriteID);
							idAndInputMemory.erase(overwriteID);
							idAndTargetMemory.erase(overwriteID);
							//printf("Pushing id: %i, write position: %i, erased id: %i \n", dataID, writePosition, overwriteID);
						}
						else
							idAndReferences[overwriteID]--;
					}
					else
						increaseReferenceCount = false;

					dataIDBuffer[writePosition] = dataID;
				}

				//If we are overwriting the same data, we want the reference count to remain unchanged
				if (increaseReferenceCount)
					idAndReferences[dataID]++;

				writePosition = (writePosition + 1) % bufferSize;
				count++;

				// Manual unlocking is done before notifying, to avoid waking up
				// the waiting thread only to block again
				lock.unlock();
				notEmpty.notify_one();
			}

			//This function should only be called if we don't have a reference of the dataID inside the buffer
			//If we have it, an exception will be thrown
			void Push(int dataID, int formatIndex, unique_ptr<OCLMemory> inputMemory, unique_ptr<OCLMemory> targetMemory)
			{
				unique_lock<mutex> lock(mute);

				//Wait until the buffer is not full
				notFull.wait(lock, [this](){ return count != bufferSize;});

				//printf("Pushing data: %i, write position: %i \n", dataID, writePosition);

				//Assume here that the dataID has not been added before
				if (idAndReferences.find(dataID) != idAndReferences.end())
					throw invalid_argument("This function is not used correctly since you should not push data that already exist");

				//If the buffer is being filled for the first time, there's nothing to overwrite
				if (static_cast<int>(dataIDBuffer.size()) <= writePosition)
					dataIDBuffer.push_back(dataID);
				else
				{
					auto overwriteID = dataIDBuffer[writePosition];
					//If the reference count is 1, we may delete the OCL memory
					auto idAndReference = idAndReferences.find(overwriteID);

					if (idAndReference == idAndReferences.end())
						throw runtime_error("We must have a reference count of the IDs in the data buffer");

					if (idAndReference->second == 1)
					{
						idAndReferences.erase(overwriteID);
						idAndFormats.erase(overwriteID);
						idAndInputMemory.erase(overwriteID);
						idAndTargetMemory.erase(overwriteID);
					}
					else
						idAndReferences[overwriteID]--;

					dataIDBuffer[writePosition] = dataID;
				}

				idAndReferences.insert(make_pair(dataID, 1));
				idAndTargetMemory.insert(make_pair(dataID, move(targetMemory)));
				idAndFormats.insert(make_pair(dataID, formatIndex));
				idAndInputMemory.insert(make_pair(dataID, move(inputMemory)));

				writePosition = (writePosition + 1) % bufferSize;
				count++;

				// Manual unlocking is done before notifying, to avoid waking up
				// the waiting thread only to block again
				lock.unlock();
				notEmpty.notify_one();
			}

			void MoveReader()
			{
				unique_lock<mutex> lock(mute);
				readPosition = (readPosition + 1) % bufferSize;
				count--;
				// Manual unlocking is done before notifying, to avoid waking up
				// the waiting thread only to block again
				lock.unlock();
				notFull.notify_one();
			};

			//If this function returns 0, add the actual memory. Else increase the reference count of the dataID
			int ReferenceCount(int dataID)
			{
				unique_lock<mutex> lock(mute);
				if (idAndReferences.find(dataID) == idAndReferences.end())
					return 0;
				else
					return idAndReferences[dataID];
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

			void InnerLoopGDLowMemory(
				size_t layerCount,
				int sample,
				LayerKernel<T>* vectorKernel,
				const vector<vector<OCLMemory*>>& gradientsPointers, 
				OCLDevice* device,
				const vector<vector<OCLMemory*>>& accumulatedGradientsPointers);

			void UpdateGradient(size_t layerCount, const vector<vector<OCLMemory*>>& accumulatedGradientsPointers,
				LayerKernel<T>* scalarKernel, OCLDevice* device);

			void TrainNetworkGDLowMemory(unique_ptr<GradientDescentConfig<T>> gdConfig, unique_ptr<ConvNetTrainer<T>> trainer);
			void ReadInputDataAsync(ConvNetTrainer<T>* trainer);
		};

	} /* namespace MachineLearning */
} /* namespace Matuna */

#endif /* MATUNA_MATUNA_OCLCONVNET_OCLCONVNET_H_ */
