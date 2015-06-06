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

namespace Matuna
{
	namespace MachineLearning
	{
		template<class T>
		CNNTrainer<T>::CNNTrainer(const vector<LayerDataDescription>& inputDataDescriptions,
			const vector<LayerDataDescription>& targetDataDescriptions,
			const vector<LayerMemoryDescription>& inputMemoryDescriptions,
			const vector<LayerMemoryDescription>& targetMemoryDescriptions) :
		inputDataDescriptions(inputDataDescriptions),
			targetDataDescriptions(targetDataDescriptions),
			inputMemoryDescriptions(inputMemoryDescriptions),
			targetMemoryDescriptions(targetMemoryDescriptions)
		{
			stopped = false;
		}

		template<class T>
		CNNTrainer<T>::~CNNTrainer()
		{

		}

		template<class T>
		bool CNNTrainer<T>::Stopping()
		{
			return stopped;
		}

		template<class T>
		void CNNTrainer<T>::Stop()
		{
			stopped = true;
		}

		template<class T>
		vector<LayerDataDescription> CNNTrainer<T>::InputDataDescriptions() const
		{
			return inputDataDescriptions;
		}

		template<class T>
		vector<LayerDataDescription> CNNTrainer<T>::TargetDataDescriptions() const
		{
			return targetDataDescriptions;
		}

		template<class T>
		vector<LayerMemoryDescription> CNNTrainer<T>::InputMemoryDescriptions() const
		{
			return inputMemoryDescriptions;
		}

		template<class T>
		vector<LayerMemoryDescription> CNNTrainer<T>::TargetMemoryDescriptions() const
		{
			return targetMemoryDescriptions;
		}

		//Just add a type if the network is suppose to support more types.
		template class CNNTrainer<float> ;
		template class CNNTrainer<double> ;
		template class CNNTrainer<long double> ;


	}
	/* Matuna */
} /* MachineLearning */

