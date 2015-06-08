/*
 * SumUnitKernel.cpp
 *
 *  Created on: May 31, 2015
 *      Author: Mikael
 */

#include "SumUnitKernel.h"
#include "Matuna.OCLHelper/OCLUtility.h"
#include "Matuna.Helper/FileHelper.h"
#include "Matuna.Helper/Path.h"

namespace Matuna {
namespace MachineLearning {

template<class T>
SumUnitKernel<T>::SumUnitKernel(int inputStride, int inputMemoryHeight,
		int inputWidthOffset, int inputHeightOffset, int inputUnitOffset,
		int outputOffset, int globalUnits, int dataWidth, int dataHeight) :
		inputStride(inputStride), inputMemoryHeight(inputMemoryHeight), inputWidthOffset(
				inputWidthOffset), inputHeightOffset(inputHeightOffset), inputUnitOffset(
				inputUnitOffset), outputOffset(outputOffset), globalUnits(
				globalUnits), dataWidth(dataWidth), dataHeight(dataHeight) {
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
SumUnitKernel<T>::~SumUnitKernel() {

}

template<class T>
void SumUnitKernel<T>::InitializeCompilerOptions() {
	stringstream stringStream;

	if (is_same<cl_double, T>::value)
		stringStream << " -D" << "DOUBLE_PRECISION ";
	else if (!is_same<cl_float, T>::value)
		throw runtime_error(
				"The template type is not valid. This is an indication of programming error");

	stringStream << "-D" << "INPUT_STRIDE=" << inputStride << " ";
	stringStream << "-D" << "INPUT_WIDTH_OFFSET=" << inputWidthOffset << " ";
	stringStream << "-D" << "WIDTH_LIMIT=" << (inputWidthOffset + dataWidth)
			<< " ";
	stringStream << "-D" << "HEIGHT_LIMIT=" << (inputHeightOffset + dataHeight)
			<< " ";
	stringStream << "-D" << "INPUT_HEIGHT_OFFSET=" << inputHeightOffset << " ";
	stringStream << "-D" << "INPUT_UNIT_OFFSET=" << inputUnitOffset << " ";
	stringStream << "-D" << "INPUT_UNIT_ELEMENT_COUNT_INC_PADDING="
			<< (inputStride * inputMemoryHeight) << " ";
	stringStream << "-D" << "OUTPUT_OFFSET=" << outputOffset << " ";

	if (useConstantInput)
		stringStream << "-D" << "CONSTANT_INPUT" << " ";

	if (useRelaxedMath)
		stringStream << "-cl-fast-relaxed-math ";

	string folderPath = Path::Combine(
			Path::GetDirectoryPath(FileHelper::GetExecutablePath()), "kernels/");
	stringStream << "-I" << folderPath << " ";

	compilerOptions = stringStream.str();
}

template<class T>
void SumUnitKernel<T>::SetInput(OCLMemory* input) {
	auto rawInput = input->GetCLMemory();
	CheckOCLError(
			clSetKernelArg(this->GetKernel(), 0, sizeof(cl_mem), &rawInput),
			"Could not set the kernel arguments");
}

template<class T>
void SumUnitKernel<T>::SetOutput(OCLMemory* output) {
	auto rawOutput = output->GetCLMemory();
	CheckOCLError(
			clSetKernelArg(this->GetKernel(), 1, sizeof(cl_mem), &rawOutput),
			"Could not set the kernel arguments");
}

template<class T>
void SumUnitKernel<T>::SetConstantInput(bool value) {
	useConstantInput = value;
}

template<class T>
void SumUnitKernel<T>::SetRelaxedMath(bool value) {
	useRelaxedMath = value;
}

template<class T>
string SumUnitKernel<T>::ProgramName() const {
	return programName;
}

template<class T>
string SumUnitKernel<T>::GetCompilerOptions() const {
	return compilerOptions;
}

template<class T>
vector<string> SumUnitKernel<T>::GetProgramCode() const {
	vector<string> result;
	string folderPath = Path::Combine(
			Path::GetDirectoryPath(FileHelper::GetExecutablePath()), "kernels");
	result.push_back(
			FileHelper::GetTextFromPath(
					Path::Combine(folderPath, "RealType.h")));
	result.push_back(
			FileHelper::GetTextFromPath(
					Path::Combine(folderPath, "SumUnitKernel.cl")));
	return result;
}

template<class T>
string SumUnitKernel<T>::KernelName() const {
	return kernelName;
}

template<class T>
const vector<size_t>& SumUnitKernel<T>::GlobalWorkSize() const {
	return globalWorkSize;
}

template<class T>
const vector<size_t>& SumUnitKernel<T>::LocalWorkSize() const {
	return localWorkSize;
}

template class SumUnitKernel<cl_float> ;
template class SumUnitKernel<cl_double> ;

} /* namespace MachineLearning */
} /* namespace Matuna */
