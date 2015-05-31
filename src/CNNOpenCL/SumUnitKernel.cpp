/*
 * SumUnitKernel.cpp
 *
 *  Created on: May 31, 2015
 *      Author: Mikael
 */

#include "SumUnitKernel.h"
#include "OpenCLHelper/OpenCLUtility.h"
#include "Helper/FileHelper.h"
#include "Helper/Path.h"

namespace ATML
{
namespace MachineLearning
{

template<class T>
SumUnitKernel<T>::SumUnitKernel(int inputStride, int inputMemoryHeight,
		int inputWidthOffset, int inputHeightOffset, int inputUnitOffset,
		int outputOffset, int globalUnits) :
		inputStride(inputStride), inputMemoryHeight(inputMemoryHeight), inputWidthOffset(
				inputWidthOffset), inputHeightOffset(inputHeightOffset), inputUnitOffset(
				inputUnitOffset), outputOffset(outputOffset), globalUnits(
				globalUnits)
{
	stringstream stringStream;

	stringStream << "SumUnitKernelProgram";
	stringStream << instanceCounter;
	programName = stringStream.str();

	kernelName = "SumUnitKernel";

	useConstantInput = false;
	useRelaxedMath = false;

	globalWorkSize.push_back(globalUnits);
}

template<class T>
SumUnitKernel<T>::~SumUnitKernel()
{

}

template<class T>
void SumUnitKernel<T>::InitializeCompilerOptions()
{

}

template<class T>
void SumUnitKernel<T>::SetInput(OpenCLMemory* input)
{
	auto rawInput = input->GetCLMemory();
	CheckOpenCLError(
			clSetKernelArg(this->GetKernel(), 0, sizeof(cl_mem), &rawInput),
			"Could not set the kernel arguments");
}

template<class T>
void SumUnitKernel<T>::SetConstantInput(bool value)
{
	useConstantInput = value;
}

template<class T>
void SumUnitKernel<T>::SetRelaxedMath(bool value)
{
	useRelaxedMath = value;
}

template<class T>
string SumUnitKernel<T>::ProgramName() const
{
	return programName;
}

template<class T>
string SumUnitKernel<T>::GetCompilerOptions() const
{
	return compilerOptions;
}

template<class T>
vector<string> SumUnitKernel<T>::GetProgramCode() const
{
	vector<string> result;
	result.push_back(
			FileHelper::GetTextFromPath(
					Path::Combine(
							Path::GetDirectoryPath(
									FileHelper::GetExecutablePath()), "kernels",
							"SumUnitKernel.cl")));
	return result;
}

template<class T>
string SumUnitKernel<T>::KernelName() const
{
	return kernelName;
}

template<class T>
const vector<size_t>& SumUnitKernel<T>::GlobalWorkSize() const
{
	return globalWorkSize;
}

template<class T>
const vector<size_t>& SumUnitKernel<T>::LocalWorkSize() const
{
	return localWorkSize;
}


template class SumUnitKernel<cl_float> ;
template class SumUnitKernel<cl_double> ;

} /* namespace MachineLearning */
} /* namespace ATML */
