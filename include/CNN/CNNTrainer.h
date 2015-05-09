/*
 * ICNNTrainer.h
 *
 *  Created on: May 8, 2015
 *      Author: Mikael
 */

#ifndef ATML_CNN_ICNNTRAINER_H_
#define ATML_CNN_ICNNTRAINER_H_

namespace ATML
{
namespace MachineLearning
{

template<class T>
class CNNTrainer
{
public:
	CNNTrainer();
	virtual ~CNNTrainer();

	virtual void MapInput(T*& networkInput, int& formatIndex) = 0;
	virtual void UnmapInput(T* networkInput, int formatIndex) = 0;
	virtual void BatchFinished(T error) = 0;
	virtual void EpochFinished() = 0;

};

} /* ATML */
} /* MachineLearning */

#endif /* ATML_CNN_ICNNTRAINER_H_ */
