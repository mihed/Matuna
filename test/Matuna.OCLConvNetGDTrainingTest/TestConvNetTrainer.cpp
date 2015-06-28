/*
* TestConvNetTrainer.cpp
*
*  Created on: Jun 3, 2015
*      Author: Mikael
*/

#include "TestConvNetTrainer.h"
#include <stdio.h>
#include <string>
#include <iostream>

namespace Matuna
{
	namespace MachineLearning
	{

		template<class T> 
		TestConvNetTrainer<T>::TestConvNetTrainer(OCLConvNet<T>* network) :
			ConvNetTrainer<T>(network)
		{

		}

		template<class T> 
		TestConvNetTrainer<T>::~TestConvNetTrainer()
		{

		}

		template<class T> 
		void TestConvNetTrainer<T>::MapInputAndTarget(int dataID, T*& input, T*& target,int& formatIndex)
		{
			target = this->target;
			input = this->input;
			formatIndex = 0;

			//cout << "Mapped" << endl;
		}

		template<class T> 
		int TestConvNetTrainer<T>::DataIDRequest()
		{
			return 0; //We only have one data in this sample
		}

		template<class T> 
		void TestConvNetTrainer<T>::UnmapInputAndTarget(int dataID, T*, T*, int)
		{
			//cout << "Unmaped" << endl;
		}

		template<class T> 
		void TestConvNetTrainer<T>::EpochStarted()
		{

		}

		template<class T> 
		void TestConvNetTrainer<T>::BatchStarted()
		{

		}

		template<class T> 
		void TestConvNetTrainer<T>::SetInput(T* input)
		{
			this->input = input;
		}

		template<class T> 
		void TestConvNetTrainer<T>::SetTarget(T* target)
		{
			this->target = target;
		}

		template<class T> 
		void TestConvNetTrainer<T>::BatchFinished(T)
		{
			//cout << "Batch finished" << endl;
		}

		template<class T> 
		void TestConvNetTrainer<T>::EpochFinished()
		{
			//cout << "Epoch finished" << endl;
		}

		template class TestConvNetTrainer<cl_float>;
		template class TestConvNetTrainer<cl_double>;

	} /* namespace MachineLearning */
} /* namespace Matuna */
