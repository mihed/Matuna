/*
* ConvNetTrainer.cpp
*
*  Created on: May 9, 2015
*      Author: Mikael
*/
/*
* IConvNetTrainer.h
*
*  Created on: May 8, 2015
*      Author: Mikael
*/
#include "ConvNetTrainer.h"

namespace Matuna
{
	namespace MachineLearning
	{
		template<class T>
		ConvNetTrainer<T>::ConvNetTrainer(const vector<LayerDataDescription>& inputDataDescriptions,
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
		ConvNetTrainer<T>::~ConvNetTrainer()
		{

		}

		template<class T>
		bool ConvNetTrainer<T>::Stopping()
		{
			return stopped;
		}

		template<class T>
		void ConvNetTrainer<T>::Stop()
		{
			stopped = true;
		}

		template<class T>
		vector<LayerDataDescription> ConvNetTrainer<T>::InputDataDescriptions() const
		{
			return inputDataDescriptions;
		}

		template<class T>
		vector<LayerDataDescription> ConvNetTrainer<T>::TargetDataDescriptions() const
		{
			return targetDataDescriptions;
		}

		template<class T>
		vector<LayerMemoryDescription> ConvNetTrainer<T>::InputMemoryDescriptions() const
		{
			return inputMemoryDescriptions;
		}

		template<class T>
		vector<LayerMemoryDescription> ConvNetTrainer<T>::TargetMemoryDescriptions() const
		{
			return targetMemoryDescriptions;
		}

		//Just add a type if the network is suppose to support more types.
		template class ConvNetTrainer<float> ;
		template class ConvNetTrainer<double> ;
		template class ConvNetTrainer<long double> ;


	}
	/* Matuna */
} /* MachineLearning */

