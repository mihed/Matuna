/*
* InputDataBufferQueue.cpp
*
*  Created on: Jun 29, 2015
*      Author: Mikael
*/

#include "InputDataBufferQueue.h"

namespace Matuna
{
	namespace MachineLearning
	{

		InputDataBufferQueue::InputDataBufferQueue(int maxBufferSize)
		{
			if (maxBufferSize == 0)
				throw invalid_argument("The buffer size has to be at least of size 1");

			this->bufferSize = maxBufferSize;
			readPosition = 0;
			writePosition = 0;
			count = 0;
			acquiredCalled = false;

		}

		InputDataBufferQueue::~InputDataBufferQueue()
		{

		}

		InputDataWrapper InputDataBufferQueue::LockAndAcquire()
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

		void InputDataBufferQueue::UnlockAcquire()
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

		void InputDataBufferQueue::Push(int dataID)
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

		void InputDataBufferQueue::Push(int dataID, int formatIndex, unique_ptr<OCLMemory> inputMemory, unique_ptr<OCLMemory> targetMemory)
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

		void InputDataBufferQueue::MoveReader()
		{
			unique_lock<mutex> lock(mute);
			readPosition = (readPosition + 1) % bufferSize;
			count--;
			// Manual unlocking is done before notifying, to avoid waking up
			// the waiting thread only to block again
			lock.unlock();
			notFull.notify_one();
		}

		int InputDataBufferQueue::ReferenceCount(int dataID)
		{
			unique_lock<mutex> lock(mute);
			if (idAndReferences.find(dataID) == idAndReferences.end())
				return 0;
			else
				return idAndReferences[dataID];
		}

	} /* namespace MachineLearning */
} /* namespace Matuna */
