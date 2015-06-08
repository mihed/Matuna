/*
 * MultiplyWithOffsetKernel.cpp
 *
 *  Created on: May 31, 2015
 *      Author: Mikael
 */

#include "MultiplyWithOffsetKernel.h"
#include "Matuna.OCLHelper/OCLUtility.h"
#include "Matuna.Helper/FileHelper.h"
#include "Matuna.Helper/Path.h"

namespace Matuna {
namespace MachineLearning {

template<class T>
MultiplyWithOffsetKernel<T>::MultiplyWithOffsetKernel(int globalWidth,
		int globalHeight, int globalUnits, int dataWidth, int dataHeight,
		int inputDeltaStride, int inputDeltaMemoryHeight, int outputStride,
		int outputMemoryHeight, int inputStride, int inputWidthOffset,
		int inputHeightOffset, int inputDeltaWidthOffset,
		int inputDeltaHeightOffset, int inputDeltaUnitOffset,
		int outputWidthoffset, int outputHeightOffset, int outputUnitOffset) :
		globalWidth(globalWidth), globalHeight(globalHeight), globalUnits(
				globalUnits), dataWidth(dataWidth), dataHeight(dataHeight), inputDeltaStride(
				inputDeltaStride), inputDeltaMemoryHeight(
				inputDeltaMemoryHeight), outputStride(outputStride), outputMemoryHeight(
				outputMemoryHeight), inputStride(inputStride), inputWidthOffset(
				inputWidthOffset), inputHeightOffset(inputHeightOffset), inputDeltaWidthOffset(
				inputDeltaWidthOffset), inputDeltaHeightOffset(
				inputDeltaHeightOffset), inputDeltaUnitOffset(
				inputDeltaUnitOffset), outputWidthoffset(outputWidthoffset), outputHeightOffset(
				outputHeightOffset), outputUnitOffset(outputUnitOffset) {
	stringstream stringStream;

	stringStream << "MultiplyWithOffsetKernelProgram";
	stringStream << instanceCounter;
	programName = stringStream.str();

	kernelName = "MultiplyWithOffsetKernel";

	useConstantInput = false;
	useConstantInputDelta = false;
	useRelaxedMath = false;

	globalWorkSize.push_back(globalWidth);
	globalWorkSize.push_back(globalHeight);
	globalWorkSize.push_back(globalUnits);
}

template<class T>
MultiplyWithOffsetKernel<T>::~MultiplyWithOffsetKernel() {

}

template<class T>
void MultiplyWithOffsetKernel<T>::InitializeCompilerOptions() {
	stringstream stringStream;

	if (is_same<cl_double, T>::value)
		stringStream << " -D" << "DOUBLE_PRECISION ";
	else if (!is_same<cl_float, T>::value)
		throw runtime_error(
				"The template type is not valid. This is an indication of programming error");

	stringStream << "-D" << "INPUT_DELTA_STRIDE=" << inputDeltaStride << " ";
	stringStream << "-D" << "OUTPUT_STRIDE=" << outputStride << " ";
	stringStream << "-D" << "INPUT_STRIDE=" << inputStride << " ";
	stringStream << "-D" << "INPUT_WIDTH_OFFSET=" << inputWidthOffset << " ";
	stringStream << "-D" << "INPUT_HEIGHT_OFFSET=" << inputHeightOffset << " ";
	stringStream << "-D" << "INPUT_DELTA_WIDTH_OFFSET=" << inputDeltaWidthOffset
			<< " ";
	stringStream << "-D" << "INPUT_DELTA_HEIGHT_OFFSET="
			<< inputDeltaHeightOffset << " ";
	stringStream << "-D" << "INPUT_DELTA_UNIT_OFFSET=" << inputDeltaUnitOffset
			<< " ";
	stringStream << "-D" << "WIDTH_LIMIT=" << dataWidth << " ";
	stringStream << "-D" << "HEIGHT_LIMIT=" << dataHeight << " ";
	stringStream << "-D" << "OUTPUT_UNIT_ELEMENT_COUNT_INC_PADDING="
			<< (outputStride * outputMemoryHeight) << " ";
	stringStream << "-D" << "INPUT_DELTA_UNIT_ELEMENT_COUNT_INC_PADDING="
			<< (inputDeltaStride * inputDeltaMemoryHeight) << " ";
	stringStream << "-D" << "OUTPUT_WIDTH_OFFSET=" << outputWidthoffset << " ";
	stringStream << "-D" << "OUTPUT_HEIGHT_OFFSET=" << outputHeightOffset
			<< " ";
	stringStream << "-D" << "OUTPUT_UNIT_OFFSET=" << outputUnitOffset << " ";

	if (useConstantInput)
		stringStream << "-D" << "CONSTANT_INPUT" << " ";

	if (useConstantInputDelta)
		stringStream << "-D" << "CONSTANT_INPUT_DELTA" << " ";

	if (useRelaxedMath)
		stringStream << "-cl-fast-relaxed-math ";

	string folderPath = Path::Combine(
			Path::GetDirectoryPath(FileHelper::GetExecutablePath()), "kernels/");
	stringStream << "-I" << folderPath << " ";

	compilerOptions = stringStream.str();
}

template<class T>
void MultiplyWithOffsetKernel<T>::SetInput(OCLMemory* input) {
	auto rawInput = input->GetCLMemory();
	CheckOCLError(
			clSetKernelArg(this->GetKernel(), 0, sizeof(cl_mem), &rawInput),
			"Could not set the kernel arguments");
}

template<class T>
void MultiplyWithOffsetKernel<T>::SetInputDelta(OCLMemory* inputDelta) {
	auto rawInput = inputDelta->GetCLMemory();
	CheckOCLError(
			clSetKernelArg(this->GetKernel(), 1, sizeof(cl_mem), &rawInput),
			"Could not set the kernel arguments");
}

template<class T>
void MultiplyWithOffsetKernel<T>::SetOutput(OCLMemory* output) {
	auto rawOutput = output->GetCLMemory();
	CheckOCLError(
			clSetKernelArg(this->GetKernel(), 2, sizeof(cl_mem), &rawOutput),
			"Could not set the kernel arguments");
}

template<class T>
void MultiplyWithOffsetKernel<T>::SetConstantInput(bool value) {
	useConstantInput = value;
}

template<class T>
void MultiplyWithOffsetKernel<T>::SetConstantInputDelta(bool value) {
	useConstantInputDelta = value;
}

template<class T>
void MultiplyWithOffsetKernel<T>::SetRelaxedMath(bool value) {
	useRelaxedMath = value;
}

template<class T>
string MultiplyWithOffsetKernel<T>::ProgramName() const {
	return programName;
}

template<class T>
string MultiplyWithOffsetKernel<T>::GetCompilerOptions() const {
	return compilerOptions;
}

template<class T>
vector<string> MultiplyWithOffsetKernel<T>::GetProgramCode() const {
	vector<string> result;
	string folderPath = Path::Combine(
			Path::GetDirectoryPath(FileHelper::GetExecutablePath()), "kernels");
	result.push_back(
			FileHelper::GetTextFromPath(
					Path::Combine(folderPath, "RealType.h")));
	result.push_back(
			FileHelper::GetTextFromPath(
					Path::Combine(folderPath, "MultiplyWithOffsetKernel.cl")));
	return result;
}

template<class T>
string MultiplyWithOffsetKernel<T>::KernelName() const {
	return kernelName;
}

template<class T>
const vector<size_t>& MultiplyWithOffsetKernel<T>::GlobalWorkSize() const {
	return globalWorkSize;
}

template<class T>
const vector<size_t>& MultiplyWithOffsetKernel<T>::LocalWorkSize() const {
	return localWorkSize;
}

template class MultiplyWithOffsetKernel<cl_float> ;
template class MultiplyWithOffsetKernel<cl_double> ;

} /* namespace MachineLearning */
} /* namespace Matuna */
