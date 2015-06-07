/*
 * SumAllUnitsKernel.cpp
 *
 *  Created on: May 20, 2015
 *      Author: Mikael
 */

#include "SumAllUnitsKernel.h"
#include "Matuna.OCLHelper/OCLUtility.h"
#include "Matuna.Helper/FileHelper.h"
#include "Matuna.Helper/Path.h"

namespace Matuna {
namespace MachineLearning {

template<class T>
SumAllUnitsKernel<T>::SumAllUnitsKernel(int globalWidth, int globalHeight,
		int unitsToSum, int inputWidthOffset, int inputHeightOffset,
		int inputUnitOffset, int inputStride, int inputMemoryHeight,
		int outputWidthOffset, int outputHeightOffset, int outputStride,
		int outputMemoryHeight) :
		globalWidth(globalWidth), globalHeight(globalHeight), unitsToSum(
				unitsToSum), inputWidthOffset(inputWidthOffset), inputHeightOffset(
				inputHeightOffset), inputUnitOffset(inputUnitOffset), inputStride(
				inputStride), inputMemoryHeight(inputMemoryHeight), outputWidthOffset(
				outputWidthOffset), outputHeightOffset(outputHeightOffset), outputStride(
				outputStride), outputMemoryHeight(outputMemoryHeight) {
	stringstream stringStream;

	stringStream << "SumAllUnitsKernelProgram";
	stringStream << instanceCounter;
	programName = stringStream.str();

	useRelaxedMath = false;
	useConstantInput = false;

	globalWorkSize.push_back(globalWidth);
	globalWorkSize.push_back(globalHeight);

	kernelName = "SumAllUnitsKernel";

}
template<class T>
SumAllUnitsKernel<T>::~SumAllUnitsKernel() {

}

template<class T>
void SumAllUnitsKernel<T>::InitializeCompilerOptions() {
	stringstream stringStream;

	if (useConstantInput)
		stringStream << "-D" << "CONSTANT_INPUT ";

	if (is_same<cl_double, T>::value)
		stringStream << "-D" << "DOUBLE_PRECISION ";
	else if (!is_same<cl_float, T>::value)
		throw runtime_error(
				"The template type is not valid. This is an indication of programming error");

	stringStream << "-D" << "UNIT_COUNT_INC_PADDING="
			<< (inputUnitOffset + unitsToSum) << " ";
	stringStream << "-D" << "UNIT_INPUT_OFFSET=" << inputUnitOffset << " ";
	stringStream << "-D" << "WIDTH_INPUT_OFFSET=" << inputWidthOffset << " ";
	stringStream << "-D" << "HEIGHT_INPUT_OFFSET=" << inputHeightOffset << " ";
	stringStream << "-D" << "WIDTH_OUTPUT_OFFSET=" << outputWidthOffset << " ";
	stringStream << "-D" << "HEIGHT_OUTPUT_OFFSET=" << outputHeightOffset
			<< " ";
	stringStream << "-D" << "WIDTH_INPUT=" << inputStride << " ";
	stringStream << "-D" << "WIDTH_OUTPUT=" << outputStride << " ";
	stringStream << "-D" << "INPUT_UNIT_ELEMENT_COUNT_INC_PADDING="
			<< (inputStride * inputMemoryHeight) << " ";

	if (useRelaxedMath)
		stringStream << "-cl-fast-relaxed-math ";

	string folderPath = Path::Combine(
			Path::GetDirectoryPath(FileHelper::GetExecutablePath()), "kernels");
	stringStream << "-I" << folderPath << " ";

	compilerOptions = stringStream.str();
}

template<class T>
void SumAllUnitsKernel<T>::SetInput(OCLMemory* input) {
	auto rawInput = input->GetCLMemory();
	CheckOCLError(
			clSetKernelArg(this->GetKernel(), 0, sizeof(cl_mem), &rawInput),
			"Could not set the kernel arguments");
}

template<class T>
void SumAllUnitsKernel<T>::SetOutput(OCLMemory* output) {
	auto rawOutput = output->GetCLMemory();
	CheckOCLError(
			clSetKernelArg(this->GetKernel(), 1, sizeof(cl_mem), &rawOutput),
			"Could not set the kernel arguments");
}

template<class T>
void SumAllUnitsKernel<T>::SetUseConstantInput(bool value) {
	this->useConstantInput = value;
}

template<class T>
void SumAllUnitsKernel<T>::SetUseRelaxedMath(bool value) {
	this->useRelaxedMath = value;
}

template<class T>
string SumAllUnitsKernel<T>::ProgramName() const {
	return programName;
}

template<class T>
string SumAllUnitsKernel<T>::GetCompilerOptions() const {
	return compilerOptions;
}

template<class T>
vector<string> SumAllUnitsKernel<T>::GetProgramCode() const {
	vector<string> result;
	string folderPath = Path::Combine(
			Path::GetDirectoryPath(FileHelper::GetExecutablePath()), "kernels");
	result.push_back(
			FileHelper::GetTextFromPath(
					Path::Combine(folderPath, "RealType.h")));
	result.push_back(
			FileHelper::GetTextFromPath(
					Path::Combine(folderPath, "SumAllUnitsKernel.cl")));
	return result;
}

template<class T>
string SumAllUnitsKernel<T>::KernelName() const {
	return kernelName;
}

template<class T>
const vector<size_t>& SumAllUnitsKernel<T>::GlobalWorkSize() const {
	return globalWorkSize;
}

template<class T>
const vector<size_t>& SumAllUnitsKernel<T>::LocalWorkSize() const {
	return localWorkSize;
}

template class SumAllUnitsKernel<cl_float> ;
template class SumAllUnitsKernel<cl_double> ;

} /* namespace MachineLearning */
} /* namespace Matuna */
