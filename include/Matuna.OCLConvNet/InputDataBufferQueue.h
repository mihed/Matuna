/*
* InputDataBufferQueue.h
*
*  Created on: Jun 29, 2015
*      Author: Mikael
*/

#ifndef MATUNA_MATUNA_OCLCONVNET_INPUTDATABUFFERQUEUE_H_
#define MATUNA_MATUNA_OCLCONVNET_INPUTDATABUFFERQUEUE_H_

#include "Matuna.OCLHelper/OCLMemory.h"

#include <unordered_map>
#include <memory>
#include <mutex>
#include <condition_variable>

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
			InputDataBufferQueue(int maxBufferSize);
			~InputDataBufferQueue();

			InputDataWrapper LockAndAcquire();
			void UnlockAcquire();
			void Push(int dataID);

			//This function should only be called if we don't have a reference of the dataID inside the buffer
			//If we have it, an exception will be thrown
			void Push(int dataID, int formatIndex, unique_ptr<OCLMemory> inputMemory, unique_ptr<OCLMemory> targetMemory);

			//If this function returns 0, add the actual memory. Else increase the reference count of the dataID
			int ReferenceCount(int dataID);
			void MoveReader();
		};

	} /* namespace MachineLearning */
} /* namespace Matuna */

#endif /* MATUNA_MATUNA_OCLCONVNET_INPUTDATABUFFERQUEUE_H_ */
