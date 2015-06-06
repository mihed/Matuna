/*
 * BackConvolutionKernel.cpp
 *
 *  Created on: May 24, 2015
 *      Author: Mikael
 */

#include "BackConvolutionKernel.h"
#include "OCLHelper/OCLUtility.h"
#include "Helper/FileHelper.h"
#include "Helper/Path.h"

namespace Matuna
{
namespace MachineLearning
{

template<class T>
BackConvolutionKernel<T>::BackConvolutionKernel(int globalWidth,
		int globalHeight, int globalUnits, int filterWidth, int filterHeight,
		int inputUnitOfffset, int inputWidthOffset, int inputHeightOffset,
		int outputWidthOffset, int outputHeightOffset, int inputStride,
		int outputStride, int inputMemoryHeight, bool useLocalmemory) :
		globalWidth(globalWidth), globalHeight(globalHeight), globalUnits(
				globalUnits), filterWidth(filterWidth), filterHeight(
				filterHeight), inputUnitOfffset(inputUnitOfffset), inputWidthOffset(
				inputWidthOffset), inputHeightOffset(inputHeightOffset), outputWidthOffset(
				outputWidthOffset), outputHeightOffset(outputHeightOffset), inputStride(
				inputStride), outputStride(outputStride), inputMemoryHeight(
				inputMemoryHeight), useLocalMemory(useLocalmemory)
{
	useRelaxedMath = false;
	useConstantDeltaInput = false;
	useConstantFilters = false;

	kernelName = "BackPropConvolutionKernel";
	stringstream stringStream;

	stringStream << "BackPropConvolutionKernelProgram";
	stringStream << instanceCounter;
	programName = stringStream.str();

	globalWorkSize.push_back(globalWidth);
	globalWorkSize.push_back(globalHeight);
}

template<class T>
BackConvolutionKernel<T>::~BackConvolutionKernel()
{

}

template<class T>
void BackConvolutionKernel<T>::InitializeCompilerOptions()
{
	stringstream stringStream;

	if (useConstantFilters)
		stringStream << "-D" << "CONSTANT_FILTERS ";
	if (useConstantDeltaInput)
		stringStream << "-D" << "CONSTANT_INPUT_DELTA ";
	if (useLocalMemory)
		stringStream << "-D" << "USE_LOCAL_MEMORY ";

	stringStream << "-D" << "INPUT_UNIT_COUNT=" << globalUnits << " ";
	stringStream << "-D" << "FILTER_WIDTH=" << filterWidth << " ";
	stringStream << "-D" << "FILTER_HEIGHT=" << filterHeight << " ";
	stringStream << "-D" << "INPUT_UNIT_OFFSET=" << inputUnitOfffset << " ";
	stringStream << "-D" << "INPUT_UNIT_LIMIT="
			<< (inputUnitOfffset + globalUnits) << " ";
	stringStream << "-D" << "INPUT_STRIDE=" << inputStride << " ";
	stringStream << "-D" << "OUTPUT_STRIDE=" << outputStride << " ";
	stringStream << "-D" << "INPUT_WIDTH_OFFSET=" << inputWidthOffset << " ";
	stringStream << "-D" << "INPUT_HEIGHT_OFFSET=" << inputHeightOffset << " ";
	stringStream << "-D" << "OUTPUT_WIDTH_OFFSET=" << outputWidthOffset << " ";
	stringStream << "-D" << "OUTPUT_HEIGHT_OFFSET=" << outputHeightOffset
			<< " ";
	stringStream << "-D" << "INPUT_UNIT_ELEMENT_COUNT_INC_PADDING="
			<< (inputStride * inputMemoryHeight) << " ";
	stringStream << "-D" << "FILTER_UNIT_ELEMENT_COUNT_INC_PADDING="
			<< (filterWidth * filterHeight) << " ";

	if (is_same<cl_double, T>::value)
		stringStream << "-D" << "DOUBLE_PRECISION ";
	else if (!is_same<cl_float, T>::value)
		throw runtime_error(
				"The template type is not valid. This is an indication of programming error");

	if (useRelaxedMath)
		stringStream << "-cl-fast-relaxed-math";

	compilerOptions = stringStream.str();
}

template<class T>
void BackConvolutionKernel<T>::SetFilters(OCLMemory* filters)
{
	auto rawFilters = filters->GetCLMemory();
	CheckOCLError(
			clSetKernelArg(this->GetKernel(), 2, sizeof(cl_mem), &rawFilters),
			"Could not set the kernel arguments");
}

template<class T>
void BackConvolutionKernel<T>::SetDeltaInput(OCLMemory* deltaInput)
{
	auto rawInput = deltaInput->GetCLMemory();
	CheckOCLError(
			clSetKernelArg(this->GetKernel(), 0, sizeof(cl_mem), &rawInput),
			"Could not set the kernel arguments");
}

template<class T>
void BackConvolutionKernel<T>::SetOutput(OCLMemory* output)
{
	auto rawOutput = output->GetCLMemory();
	CheckOCLError(
			clSetKernelArg(this->GetKernel(), 1, sizeof(cl_mem), &rawOutput),
			"Could not set the kernel arguments");
}

template<class T>
void BackConvolutionKernel<T>::SetLocalWorkGroup(int width, int height)
{
	if (!useLocalMemory)
		throw invalid_argument(
				"Local memory is not used so you may not set any local memory.");

	CheckOCLError(
			clSetKernelArg(this->GetKernel(), 3,
					sizeof(T) * (width + filterWidth - 1)
							* (height + filterHeight - 1) * globalUnits,
					nullptr), "Could not set the kernel arguments");
	localWorkSize.clear();
	localWorkSize.push_back(width);
	localWorkSize.push_back(height);
}

template<class T>
void BackConvolutionKernel<T>::SetConstantInputDelta(bool value)
{
	useConstantDeltaInput = value;
}

template<class T>
void BackConvolutionKernel<T>::SetConstantFilters(bool value)
{
	useConstantFilters = value;
}

template<class T>
void BackConvolutionKernel<T>::SetRelaxedMath(bool value)
{
	useRelaxedMath = value;
}

template<class T>
string BackConvolutionKernel<T>::ProgramName() const
{
	return programName;
}

template<class T>
string BackConvolutionKernel<T>::GetCompilerOptions() const
{
	return compilerOptions;
}

template<class T>
vector<string> BackConvolutionKernel<T>::GetProgramCode() const
{
	vector<string> result;
	result.push_back(
			FileHelper::GetTextFromPath(
					Path::Combine(
							Path::GetDirectoryPath(
									FileHelper::GetExecutablePath()), "kernels",
							"BackPropConvolutionKernel.cl")));
	return result;
}

template<class T>
string BackConvolutionKernel<T>::KernelName() const
{
	return kernelName;
}

template<class T>
const vector<size_t>& BackConvolutionKernel<T>::GlobalWorkSize() const
{
	return globalWorkSize;
}

template<class T>
const vector<size_t>& BackConvolutionKernel<T>::LocalWorkSize() const
{
	return localWorkSize;
}

template class BackConvolutionKernel<cl_float> ;
template class BackConvolutionKernel<cl_double> ;

} /* namespace MachineLearning */
} /* namespace Matuna */
