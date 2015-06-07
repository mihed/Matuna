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
		TestConvNetTrainer<T>::TestConvNetTrainer( const vector<LayerDataDescription>& inputDataDescriptions,
			const vector<LayerDataDescription>& targetDataDescriptions,
			const vector<LayerMemoryDescription>& inputMemoryDescriptions,
			const vector<LayerMemoryDescription>& targetMemoryDescriptions, OCLConvNet<T>* network) :
		ConvNetTrainer<T>(inputDataDescriptions, targetDataDescriptions, inputMemoryDescriptions, targetMemoryDescriptions),
			network(network)
		{

		}

		template<class T> 
		TestConvNetTrainer<T>::~TestConvNetTrainer()
		{

		}

		template<class T> 
		void TestConvNetTrainer<T>::MapInputAndTarget(T*& input, T*& target,int& formatIndex)
		{
			target = this->target;
			input = this->input;
			formatIndex = 0;

			//cout << "Mapped" << endl;
		}

		template<class T> 
		void TestConvNetTrainer<T>::UnmapInputAndTarget(T* input, T* target, int formatIndex)
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
		void TestConvNetTrainer<T>::BatchFinished(T error)
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
