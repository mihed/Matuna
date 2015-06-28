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
		ConvNetTrainer<T>::ConvNetTrainer(TrainableConvNet<T>* convNet)
		{
			bufferSize = 1;
			enableError = false;
			stopped = false;
			this->convNet = convNet;
		}

		template<class T>
		ConvNetTrainer<T>::~ConvNetTrainer()
		{

		}

		template<class T>
		void ConvNetTrainer<T>::SetBufferSize(int size)
		{
			this->bufferSize = size;
		}

		template<class T>
		int ConvNetTrainer<T>::GetBufferSize() const
		{
			return bufferSize;
		}

		template<class T>
		void ConvNetTrainer<T>::SetEnableError(bool value)
		{
			this->enableError = value;
		}

		template<class T>
		bool ConvNetTrainer<T>::GetEnableError() const
		{
			return enableError;
		}

		template<class T>
		bool ConvNetTrainer<T>::Stopping() const
		{
			return stopped;
		}

		template<class T>
		void ConvNetTrainer<T>::Stop()
		{
			stopped = true;
		}

		//Just add a type if the network is suppose to support more types.
		template class ConvNetTrainer<float> ;
		template class ConvNetTrainer<double> ;
		template class ConvNetTrainer<long double> ;


	}
	/* Matuna */
} /* MachineLearning */

