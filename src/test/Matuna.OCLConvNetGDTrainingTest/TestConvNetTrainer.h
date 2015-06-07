/*
* TestConvNetTrainer.h
*
*  Created on: Jun 3, 2015
*      Author: Mikael
*/

#ifndef MATUNA_TEST_OCLConvNetGDTRAININGTEST_TESTConvNetTRAINER_H_
#define MATUNA_TEST_OCLConvNetGDTRAININGTEST_TESTConvNetTRAINER_H_

#include "Matuna.ConvNet/ConvNetTrainer.h"
#include "Matuna.Math/Matrix.h"
#include "Matuna.OCLConvNet/OCLConvNet.h"

using namespace std;
using namespace Matuna::MachineLearning;
using namespace Matuna::Math;

namespace Matuna
{
	namespace MachineLearning
	{

		template<class T> 
		class TestConvNetTrainer: public ConvNetTrainer<T>
		{
		private:
			OCLConvNet<T>* network;
			T* input;
			T* target;
		public:
			TestConvNetTrainer( const vector<LayerDataDescription>& inputDataDescriptions,
				const vector<LayerDataDescription>& targetDataDescriptions,
				const vector<LayerMemoryDescription>& inputMemoryDescriptions,
				const vector<LayerMemoryDescription>& targetMemoryDescriptions, OCLConvNet<T>* network);
			~TestConvNetTrainer();

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

#endif /* MATUNA_TEST_OCLConvNetGDTRAININGTEST_TESTConvNetTRAINER_H_ */
