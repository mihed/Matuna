/*
 * PerceptronKernel.cpp
 *
 *  Created on: May 5, 2015
 *      Author: Mikael
 */

#include "ForwardPerceptronKernel.h"
#include "Helper/Path.h"
#include <sstream>

namespace ATML
{
namespace MachineLearning
{

ForwardPerceptronKernel::ForwardPerceptronKernel() :
		OpenCLKernel()
{
	stringstream stringStream;
	//We are using unique kernels with macros. So the name will be unique for every kernel
	stringStream << "PerceptronProgram";
	stringStream << instanceCounter;
	programName = stringStream.str();
	kernelName = "ForwardPerceptronKernel";
}

ForwardPerceptronKernel::~ForwardPerceptronKernel()
{

}

string ForwardPerceptronKernel::ProgramName() const
{
	return programName;
}

string ForwardPerceptronKernel::ProgramCode() const
{
	return GetTextFromPath(Path::Combine("kernels", "ForwardPerceptronKernel.cl"));
}
string ForwardPerceptronKernel::KernelName() const
{
	return kernelName;
}
const vector<tuple<int, shared_ptr<OpenCLMemory>>>& ForwardPerceptronKernel::GetMemoryArguments() const
{
	return memoryArguments;
}
const vector<tuple<int, size_t, void*>>& ForwardPerceptronKernel::GetOtherArguments() const
{
	return otherArguments;
}
const vector<size_t>& ForwardPerceptronKernel::GlobalWorkSize() const
{
	return globalWorkSize;
}
const vector<size_t>& ForwardPerceptronKernel::LocalWorkSize() const
{
	return localWorkSize;
}

} /* namespace MachineLearning */
} /* namespace ATML */
