/*
* ICNNTrainer.h
*
*  Created on: May 8, 2015
*      Author: Mikael
*/

#ifndef MATUNA_CNN_ICNNTRAINER_H_
#define MATUNA_CNN_ICNNTRAINER_H_

#include "LayerDescriptions.h"
#include <vector>

using namespace std;

namespace Matuna
{
	namespace MachineLearning
	{

		template<class T>
		class CNNTrainer
		{
		private:
			bool stopped;
			vector<LayerDataDescription> inputDataDescriptions;
			vector<LayerDataDescription> targetDataDescriptions;
			vector<LayerMemoryDescription> inputMemoryDescriptions;
			vector<LayerMemoryDescription> targetMemoryDescriptions;

		public:
			CNNTrainer( const vector<LayerDataDescription>& inputDataDescriptions,
				const vector<LayerDataDescription>& targetDataDescriptions,
				const vector<LayerMemoryDescription>& inputMemoryDescriptions,
				const vector<LayerMemoryDescription>& targetMemoryDescriptions);
			virtual ~CNNTrainer();

			//Assuming that the memory is aligned at the moment
			virtual void MapInputAndTarget(T*& input, T*& target,int& formatIndex) = 0;
			virtual void UnmapInputAndTarget(T* input, T* target,int formatIndex) = 0;
			virtual void BatchFinished(T error) = 0;
			virtual void EpochFinished() = 0;
			virtual void EpochStarted() = 0;
			virtual void BatchStarted() = 0;


			vector<LayerDataDescription> InputDataDescriptions() const;
			vector<LayerDataDescription> TargetDataDescriptions() const;
			vector<LayerMemoryDescription> InputMemoryDescriptions() const;
			vector<LayerMemoryDescription> TargetMemoryDescriptions() const;

			bool Stopping();
			void Stop();

		};

	} /* Matuna */
} /* MachineLearning */

#endif /* MATUNA_CNN_ICNNTRAINER_H_ */
