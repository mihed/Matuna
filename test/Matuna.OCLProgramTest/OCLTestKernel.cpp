/*
* OCLTestKernel.cpp
*
*  Created on: Jun 10, 2015
*      Author: Mikael
*/

#include "OCLTestKernel.h"

OCLTestKernel::OCLTestKernel(string name)
{
	this->name = name;
}

OCLTestKernel::~OCLTestKernel()
{

}

string OCLTestKernel::Name() const 
{
	return name;
}

const vector<size_t>& OCLTestKernel::GlobalWorkSize() const
{
	return globalWorkSize;
}

const vector<size_t>& OCLTestKernel::LocalWorkSize() const
{
	return localWorkSize;
}

