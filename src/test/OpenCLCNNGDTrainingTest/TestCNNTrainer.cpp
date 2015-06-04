/*
* TestCNNTrainer.cpp
*
*  Created on: Jun 3, 2015
*      Author: Mikael
*/

#include "TestCNNTrainer.h"
#include <stdio.h>
#include <string>
#include <iostream>

namespace ATML
{
	namespace MachineLearning
	{

		template<class T> 
		TestCNNTrainer<T>::TestCNNTrainer( const vector<LayerDataDescription>& inputDataDescriptions,
			const vector<LayerDataDescription>& targetDataDescriptions,
			const vector<LayerMemoryDescription>& inputMemoryDescriptions,
			const vector<LayerMemoryDescription>& targetMemoryDescriptions, CNNOpenCL<T>* network) :
		CNNTrainer<T>(inputDataDescriptions, targetDataDescriptions, inputMemoryDescriptions, targetMemoryDescriptions),
			network(network)
		{

		}

		template<class T> 
		TestCNNTrainer<T>::~TestCNNTrainer()
		{

		}

		template<class T> 
		void TestCNNTrainer<T>::MapInputAndTarget(T*& input, T*& target,int& formatIndex)
		{
			target = this->target;
			input = this->input;
			formatIndex = 0;

			//cout << "Mapped" << endl;
		}

		template<class T> 
		void TestCNNTrainer<T>::UnmapInputAndTarget(T* input, T* target, int formatIndex)
		{
			//cout << "Unmaped" << endl;
		}

		template<class T> 
		void TestCNNTrainer<T>::EpochStarted()
		{

		}

		template<class T> 
		void TestCNNTrainer<T>::BatchStarted()
		{

		}

		template<class T> 
		void TestCNNTrainer<T>::SetInput(T* input)
		{
			this->input = input;
		}

		template<class T> 
		void TestCNNTrainer<T>::SetTarget(T* target)
		{
			this->target = target;
		}

		template<class T> 
		void TestCNNTrainer<T>::BatchFinished(T error)
		{
			//cout << "Batch finished" << endl;
		}

		template<class T> 
		void TestCNNTrainer<T>::EpochFinished()
		{
			//cout << "Epoch finished" << endl;
		}

		template class TestCNNTrainer<cl_float>;
		template class TestCNNTrainer<cl_double>;

	} /* namespace MachineLearning */
} /* namespace ATML */
