/*
* TestCNNTrainer.h
*
*  Created on: Jun 3, 2015
*      Author: Mikael
*/

#ifndef MATUNA_TEST_OCLCNNGDTRAININGTEST_TESTCNNTRAINER_H_
#define MATUNA_TEST_OCLCNNGDTRAININGTEST_TESTCNNTRAINER_H_

#include "CNN/CNNTrainer.h"
#include "Math/Matrix.h"
#include "CNNOCL/CNNOCL.h"

using namespace std;
using namespace Matuna::MachineLearning;
using namespace Matuna::Math;

namespace Matuna
{
	namespace MachineLearning
	{

		template<class T> 
		class TestCNNTrainer: public CNNTrainer<T>
		{
		private:
			CNNOCL<T>* network;
			T* input;
			T* target;
		public:
			TestCNNTrainer( const vector<LayerDataDescription>& inputDataDescriptions,
				const vector<LayerDataDescription>& targetDataDescriptions,
				const vector<LayerMemoryDescription>& inputMemoryDescriptions,
				const vector<LayerMemoryDescription>& targetMemoryDescriptions, CNNOCL<T>* network);
			~TestCNNTrainer();

			virtual void MapInputAndTarget(T*& input, T*& target,int& formatIndex) override;
			virtual void UnmapInputAndTarget(T* input, T* target,int formatIndex) override;
			virtual void BatchFinished(T error) override;
			virtual void EpochFinished() override;
			virtual void EpochStarted() override;
			virtual void BatchStarted() override;

			void SetInput(T* input);
			void SetTarget(T* target);
		};

	} /* namespace MachineLearning */
} /* namespace Matuna */

#endif /* MATUNA_TEST_OCLCNNGDTRAININGTEST_TESTCNNTRAINER_H_ */
