/*
* IConvNetTrainer.h
*
*  Created on: May 8, 2015
*      Author: Mikael
*/

#ifndef MATUNA_MATUNA_CONVNET_CONVNETTRAINER_H_
#define MATUNA_MATUNA_CONVNET_CONVNETTRAINER_H_

#include "LayerDescriptions.h"
#include <vector>

using namespace std;

namespace Matuna
{
	namespace MachineLearning
	{

		template<class T>
		class TrainableConvNet;

		template<class T>
		class ConvNetTrainer
		{
		private:
			bool stopped;
			bool enableError;
			int bufferSize;

		protected:
			TrainableConvNet<T>* convNet;

		public:
			ConvNetTrainer(TrainableConvNet<T>* convNet);
			virtual ~ConvNetTrainer();

			virtual void MapInputAndTarget(int dataID, T*& input, T*& target,int& formatIndex) = 0;
			virtual void UnmapInputAndTarget(int dataID, T* input, T* target, int formatIndex) = 0;
			virtual void BatchFinished(T error) = 0;
			virtual void EpochFinished() = 0;
			virtual void EpochStarted() = 0;
			virtual void BatchStarted() = 0;

			//This function is called when the device wants to fill its buffer with the data corresponding to the ID
			//If the ID is already in the device buffer, then the map / unmap functions will be skipped yielding optimization .
			virtual int DataIDRequest() = 0;

			void SetBufferSize(int size);
			int GetBufferSize() const;

			void SetEnableError(bool value);
			bool GetEnableError() const;

			bool Stopping() const;
			void Stop();

		};

	} /* Matuna */
} /* MachineLearning */

#endif /* MATUNA_MATUNA_CONVNET_CONVNETTRAINER_H_ */
