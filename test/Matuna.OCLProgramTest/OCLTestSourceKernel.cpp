/*
* OCLTestSourceKernel.cpp
*
*  Created on: Jun 10, 2015
*      Author: Mikael
*/

#include "OCLTestSourceKernel.h"
#include "Matuna.OCLHelper/OCLProgram.h"

OCLTestSourceKernel::OCLTestSourceKernel()
{
	this->name = "ConvolutionKernel";
}

OCLTestSourceKernel::~OCLTestSourceKernel()
{

}

string OCLTestSourceKernel::Name() const
{
	return name;
}

const vector<size_t>& OCLTestSourceKernel::GlobalWorkSize() const
{
	return globalWorkSize;
}

const vector<size_t>& OCLTestSourceKernel::LocalWorkSize() const
{
	return localWorkSize;
}

vector<string> OCLTestSourceKernel::GetIncludePaths() const
{
	vector<string> result;
	result.push_back(OCLProgram::DefaultSourceLocation);
	return result;
}

vector<string> OCLTestSourceKernel::GetSourcePaths() const
{
	vector<string> result;
	result.push_back(OCLProgram::DefaultSourceLocation + "/ConvolutionKernel.cl");
	return result;
}

