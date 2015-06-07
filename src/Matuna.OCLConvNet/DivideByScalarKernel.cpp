/*
 * DivideByScalarKernel.cpp
 *
 *  Created on: May 19, 2015
 *      Author: Mikael
 */

#include "DivideByScalarKernel.h"
#include "Matuna.OCLHelper/OCLUtility.h"
#include "Matuna.Helper/FileHelper.h"
#include "Matuna.Helper/Path.h"

namespace Matuna
{
namespace MachineLearning
{

template<class T>
DivideByScalarKernel<T>::DivideByScalarKernel(int inputCount) :
		inputCount(inputCount)
{
	stringstream stringStream;

	stringStream << "DivideByScalarProgram";
	stringStream << instanceCounter;
	programName = stringStream.str();
	kernelName = "DivideByScalarKernel";

	stringStream.str("");
	stringStream.clear();

	if (is_same<cl_double, T>::value)
		stringStream << "-D" << "DOUBLE_PRECISION ";
	else if (!is_same<cl_float, T>::value)
		throw runtime_error(
				"The template type is not valid. This is an indication of programming error");

	compilerOptions = stringStream.str();
	globalWorkSize.push_back(inputCount);
}

template<class T>
DivideByScalarKernel<T>::~DivideByScalarKernel()
{

}

template<class T>
void DivideByScalarKernel<T>::SetInputOutput(OCLMemory* inputOutput)
{
	auto rawInput = inputOutput->GetCLMemory();
	CheckOCLError(
			clSetKernelArg(this->GetKernel(), 0, sizeof(cl_mem), &rawInput),
			"Could not set the kernel arguments");
}

template<class T>
void DivideByScalarKernel<T>::SetScalar(OCLMemory* scalar)
{
	auto rawOutput = scalar->GetCLMemory();
	CheckOCLError(
			clSetKernelArg(this->GetKernel(), 1, sizeof(cl_mem), &rawOutput),
			"Could not set the kernel arguments");
}

template<class T>
string DivideByScalarKernel<T>::ProgramName() const
{
	return programName;
}

template<class T>
string DivideByScalarKernel<T>::GetCompilerOptions() const
{
	return compilerOptions;
}

template<class T>
vector<string> DivideByScalarKernel<T>::GetProgramCode() const
{
	vector<string> result;
	result.push_back(
			FileHelper::GetTextFromPath(
					Path::Combine(
							Path::GetDirectoryPath(
									FileHelper::GetExecutablePath()), "kernels",
							"DivideByScalarKernel.cl")));
	return result;
}

template<class T>
string DivideByScalarKernel<T>::KernelName() const
{
	return kernelName;
}

template<class T>
const vector<size_t>& DivideByScalarKernel<T>::GlobalWorkSize() const
{
	return globalWorkSize;
}

template<class T>
const vector<size_t>& DivideByScalarKernel<T>::LocalWorkSize() const
{
	return localWorkSize;
}

template class DivideByScalarKernel<cl_float> ;
template class DivideByScalarKernel<cl_double>;

} /* namespace MachineLearning */
} /* namespace Matuna */
