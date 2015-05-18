/*
 * CNNTrainer.cpp
 *
 *  Created on: May 9, 2015
 *      Author: Mikael
 */
/*
 * ICNNTrainer.h
 *
 *  Created on: May 8, 2015
 *      Author: Mikael
 */
#include "CNNTrainer.h"

namespace ATML
{
namespace MachineLearning
{
template<class T>
CNNTrainer<T>::CNNTrainer()
{

}

template<class T>
CNNTrainer<T>::~CNNTrainer()
{

}

//Just add a type if the network is suppose to support more types.
template class CNNTrainer<float> ;
template class CNNTrainer<double> ;
template class CNNTrainer<long double> ;


}
/* ATML */
} /* MachineLearning */

