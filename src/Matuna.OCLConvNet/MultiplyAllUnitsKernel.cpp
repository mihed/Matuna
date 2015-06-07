/*
 * MultiplyAllUnitsKernel.cpp
 *
 *  Created on: May 24, 2015
 *      Author: Mikael
 */

#include "MultiplyAllUnitsKernel.h"
#include "Matuna.OCLHelper/OCLUtility.h"
#include "Matuna.Helper/FileHelper.h"
#include "Matuna.Helper/Path.h"

namespace Matuna {
namespace MachineLearning {

template<class T>
MultiplyAllUnitsKernel<T>::MultiplyAllUnitsKernel(int globalWidth,
		int globalHeight, int globalUnits, int inputDeltaStride,
		int outputStride, int inputStride, int inputDeltaWidthOffset,
		int inputDeltaHeightOffset, int outputWidthOffset,
		int outputHeightOffset, int outputUnitOffset, int inputWidthOffset,
		int inputHeightOffset, int inputUnitOffset, int outputMemoryHeight,
		int inputMemoryHeight) :
		globalWidth(globalWidth), globalHeight(globalHeight), globalUnits(
				globalUnits), inputDeltaStride(inputDeltaStride), outputStride(
				outputStride), inputStride(inputStride), inputDeltaWidthOffset(
				inputDeltaWidthOffset), inputDeltaHeightOffset(
				inputDeltaHeightOffset), outputWidthOffset(outputWidthOffset), outputHeightOffset(
				outputHeightOffset), outputUnitOffset(outputUnitOffset), inputWidthOffset(
				inputWidthOffset), inputHeightOffset(inputHeightOffset), inputUnitOffset(
				inputUnitOffset), outputMemoryHeight(outputMemoryHeight), inputMemoryHeight(
				inputMemoryHeight) {
	stringstream stringStream;

	stringStream << "MultiplyAllUnitsKernelProgram";
	stringStream << instanceCounter;
	programName = stringStream.str();

	useRelaxedMath = false;
	useConstantInput = false;
	useConstantInputDelta = false;
	activationFunction = MatunaLinearActivation;

	kernelName = "MultiplyAllUnitsKernel";

	globalWorkSize.push_back(globalWidth);
	globalWorkSize.push_back(globalHeight);
	globalWorkSize.push_back(globalUnits);
}

template<class T>
MultiplyAllUnitsKernel<T>::~MultiplyAllUnitsKernel() {

}

template<class T>
void MultiplyAllUnitsKernel<T>::InitializeCompilerOptions() {
	stringstream stringStream;

	if (is_same<cl_double, T>::value)
		stringStream << "-D" << "DOUBLE_PRECISION ";
	else if (!is_same<cl_float, T>::value)
		throw runtime_error(
				"The template type is not valid. This is an indication of programming error");

	if (useConstantInput)
		stringStream << "-D" << "CONSTANT_INPUT ";
	if (useConstantInputDelta)
		stringStream << "-D" << "CONSTANT_INPUT_DELTA ";

	stringStream << "-D" << "INPUT_UNIT_ELEMENT_COUNT_INC_PADDING="
			<< (inputStride * inputMemoryHeight) << " ";
	stringStream << "-D" << "OUTPUT_UNIT_ELEMENT_COUNT_INC_PADDING="
			<< (outputStride * outputMemoryHeight) << " ";
	stringStream << "-D" << "INPUT_DELTA_STRIDE=" << inputDeltaStride << " ";
	stringStream << "-D" << "OUTPUT_STRIDE=" << outputStride << " ";
	stringStream << "-D" << "INPUT_STRIDE=" << inputStride << " ";
	stringStream << "-D" << "INPUT_DELTA_WIDTH_OFFSET=" << inputDeltaWidthOffset
			<< " ";
	stringStream << "-D" << "INPUT_DELTA_HEIGHT_OFFSET="
			<< inputDeltaHeightOffset << " ";
	stringStream << "-D" << "OUTPUT_WIDTH_OFFSET=" << outputWidthOffset << " ";
	stringStream << "-D" << "OUTPUT_HEIGHT_OFFSET=" << outputHeightOffset
			<< " ";
	stringStream << "-D" << "OUTPUT_UNIT_OFFSET=" << outputUnitOffset << " ";
	stringStream << "-D" << "INPUT_WIDTH_OFFSET=" << inputWidthOffset << " ";
	stringStream << "-D" << "INPUT_HEIGHT_OFFSET=" << inputHeightOffset << " ";
	stringStream << "-D" << "INPUT_UNIT_OFFSET=" << inputUnitOffset << " ";

	if (activationFunction == MatunaSigmoidActivation)
		stringStream << "-D" << "SIGMOID ";
	else if (activationFunction == MatunaTanhActivation)
		stringStream << "-D" << "TANH ";

	if (useRelaxedMath)
		stringStream << "-cl-fast-relaxed-math ";

	string folderPath = Path::Combine(
			Path::GetDirectoryPath(FileHelper::GetExecutablePath()), "kernels");
	stringStream << "-I" << folderPath << " ";

	compilerOptions = stringStream.str();
}

template<class T>
void MultiplyAllUnitsKernel<T>::SetUseRelaxedMath(bool value) {
	useRelaxedMath = value;
}

template<class T>
void MultiplyAllUnitsKernel<T>::SetUseConstantInput(bool value) {
	useConstantInput = value;
}

template<class T>
void MultiplyAllUnitsKernel<T>::SetUseConstantInputDelta(bool value) {
	useConstantInputDelta = value;
}

template<class T>
void MultiplyAllUnitsKernel<T>::SetActivationFunction(
		MatunaActivationFunction activationFunction) {
	this->activationFunction = activationFunction;
}

template<class T>
void MultiplyAllUnitsKernel<T>::SetInput(OCLMemory* input) {
	auto rawInput = input->GetCLMemory();
	CheckOCLError(
			clSetKernelArg(this->GetKernel(), 0, sizeof(cl_mem), &rawInput),
			"Could not set the kernel arguments");
}

template<class T>
void MultiplyAllUnitsKernel<T>::SetInputDelta(OCLMemory* inputDelta) {
	auto rawInputDelta = inputDelta->GetCLMemory();
	CheckOCLError(
			clSetKernelArg(this->GetKernel(), 1, sizeof(cl_mem),
					&rawInputDelta), "Could not set the kernel arguments");
}

template<class T>
void MultiplyAllUnitsKernel<T>::SetOutput(OCLMemory* output) {
	auto rawOutput = output->GetCLMemory();
	CheckOCLError(
			clSetKernelArg(this->GetKernel(), 2, sizeof(cl_mem), &rawOutput),
			"Could not set the kernel arguments");
}

template<class T>
string MultiplyAllUnitsKernel<T>::ProgramName() const {
	return programName;
}

template<class T>
string MultiplyAllUnitsKernel<T>::GetCompilerOptions() const {
	return compilerOptions;
}

template<class T>
vector<string> MultiplyAllUnitsKernel<T>::GetProgramCode() const {
	vector<string> result;
	string folderPath = Path::Combine(
			Path::GetDirectoryPath(FileHelper::GetExecutablePath()), "kernels");
	result.push_back(
			FileHelper::GetTextFromPath(
					Path::Combine(folderPath, "RealType.h")));
	result.push_back(
			FileHelper::GetTextFromPath(
					Path::Combine(folderPath, "ActivationFunction.h")));
	result.push_back(
			FileHelper::GetTextFromPath(
					Path::Combine(folderPath, "MultiplyAllUnitsKernel.cl")));
	return result;
}

template<class T>
string MultiplyAllUnitsKernel<T>::KernelName() const {
	return kernelName;
}

template<class T>
const vector<size_t>& MultiplyAllUnitsKernel<T>::GlobalWorkSize() const {
	return globalWorkSize;
}

template<class T>
const vector<size_t>& MultiplyAllUnitsKernel<T>::LocalWorkSize() const {
	return localWorkSize;
}

template class MultiplyAllUnitsKernel<cl_float> ;
template class MultiplyAllUnitsKernel<cl_double> ;

} /* namespace MachineLearning */
} /* namespace Matuna */
