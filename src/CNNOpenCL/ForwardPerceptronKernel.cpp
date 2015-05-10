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

ForwardPerceptronKernel::ForwardPerceptronKernel()
{
	stringstream stringStream;

	stringStream << "ForwardPerceptronProgram";
	stringStream << instanceCounter;
	programName = stringStream.str();
	kernelName = "ForwardPerceptronKernel";
}

ForwardPerceptronKernel::~ForwardPerceptronKernel()
{

}

void ForwardPerceptronKernel::SetArguments()
{

}

string ForwardPerceptronKernel::ProgramName() const
{
	return programName;
}

vector<string> ForwardPerceptronKernel::GetProgramCode() const
{
	vector<string> result;
	result.push_back(
			GetTextFromPath(
					Path::Combine("kernels", "ForwardPerceptronKernel.cl")));
	return result;
}
string ForwardPerceptronKernel::KernelName() const
{
	return kernelName;
}

string ForwardPerceptronKernel::GetCompilerOptions() const
{
	return string();
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
