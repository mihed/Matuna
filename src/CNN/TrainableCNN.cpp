/*
 * TrainableCNN.cpp
 *
 *  Created on: May 9, 2015
 *      Author: Mikael
 */

#include "TrainableCNN.h"

namespace ATML
{
namespace MachineLearning
{

//Just add a type if the network is suppose to support more types.

template class TrainableCNN<float> ;
template class TrainableCNN<double> ;
template class TrainableCNN<long double> ;

template<class T>
TrainableCNN<T>::TrainableCNN(const CNNConfig& config) :
		CNN(config)
{

}

template<class T>
TrainableCNN<T>::~TrainableCNN()
{

}

} /* namespace MachineLearning */
} /* namespace ATML */
