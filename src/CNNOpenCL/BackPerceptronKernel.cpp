/*
 * BackPerceptronKernel.cpp
 *
 *  Created on: May 16, 2015
 *      Author: Mikael
 */

#include "BackPerceptronKernel.h"
#include "OpenCLHelper/OpenCLUtility.h"
#include "Helper/FileHelper.h"
#include "Helper/Path.h"

namespace ATML {
namespace MachineLearning {

template<class T>
BackPerceptronKernel<T>::BackPerceptronKernel(int inputUnits, int outputUnits,
		int inputDeltaOffset, int inputOffset, int outputOffset) :
		inputUnits(inputUnits), outputUnits(outputUnits), inputDeltaOffset(
				inputDeltaOffset), inputOffset(inputOffset), outputOffset(
				outputOffset) {
	useRelaxedMath = false;
	useConstantInput = false;
	useConstantDeltaInput = false;
	useConstantWeights = false;
	kernelName = "BackPerceptronKernel";
	activationFunction = ATMLLinearActivation;
	weights = nullptr;
	stringstream stringStream;

	stringStream << "BackPerceptronProgram";
	stringStream << instanceCounter;
	programName = stringStream.str();

	globalWorkSize.push_back(outputUnits);
}

template<class T>
BackPerceptronKernel<T>::~BackPerceptronKernel() {

}

template<class T>
void BackPerceptronKernel<T>::SetUseConstantWeights(bool value) {
	useConstantWeights = value;
}

template<class T>
void BackPerceptronKernel<T>::SetUseConstantInput(bool value) {
	useConstantInput = value;
}

template<class T>
void BackPerceptronKernel<T>::SetUseConstantDeltaInput(bool value) {
	useConstantDeltaInput = value;
}

template<class T>
void BackPerceptronKernel<T>::SetUseRelaxedMath(bool value) {
	useRelaxedMath = value;
}

template<class T>
void BackPerceptronKernel<T>::SetActivationFunction(
		ATMLActivationFunction activationFunction) {
	this->activationFunction = activationFunction;
}

template<class T>
void BackPerceptronKernel<T>::SetWeights(OpenCLMemory* weights) {
	this->weights = weights;
}

template<class T>
void BackPerceptronKernel<T>::SetInput(OpenCLMemory* input) {
	auto rawInput = input->GetCLMemory();
	CheckOpenCLError(
			clSetKernelArg(this->GetKernel(), 0, sizeof(cl_mem), &rawInput),
			"Could not set the kernel arguments");
}

template<class T>
void BackPerceptronKernel<T>::SetDeltaInput(OpenCLMemory* deltaInput) {
	auto rawInput = deltaInput->GetCLMemory();
	CheckOpenCLError(
			clSetKernelArg(this->GetKernel(), 1, sizeof(cl_mem), &rawInput),
			"Could not set the kernel arguments");
}

template<class T>
void BackPerceptronKernel<T>::SetOutput(OpenCLMemory* output) {
	auto rawOutput = output->GetCLMemory();
	CheckOpenCLError(
			clSetKernelArg(this->GetKernel(), 2, sizeof(cl_mem), &rawOutput),
			"Could not set the kernel arguments");
}

template<class T>
void BackPerceptronKernel<T>::InitializeArguments() {
	auto rawWeights = weights->GetCLMemory();
	CheckOpenCLError(
			clSetKernelArg(this->GetKernel(), 3, sizeof(cl_mem), &rawWeights),
			"Could not set the kernel arguments");
}

template<class T>
void BackPerceptronKernel<T>::InitializeCompilerOptions() {
	stringstream stringStream;

	if (useConstantWeights)
		stringStream << "-D" << "CONSTANT_WEIGHTS ";
	if (useConstantInput)
		stringStream << "-D" << "CONSTANT_INPUT ";
	if (useConstantDeltaInput)
		stringStream << "-D" << "CONSTANT_INPUT_DELTA ";

	stringStream << "-D" << "INPUT_DELTA_COUNT=" << inputUnits << " ";
	stringStream << "-D" << "WEIGHT_COLUMN_COUNT=" << outputUnits << " ";
	stringStream << "-D" << "INPUT_OFFSET" << inputOffset << " ";
	stringStream << "-D" << "INPUT_DELTA_OFFSET" << inputDeltaOffset << " ";
	stringStream << "-D" << "OUTPUT_DELTA_OFFSET" << outputOffset << " ";

	if (is_same<cl_double, T>::value)
		stringStream << "-D" << "DOUBLE_PRECISION ";
	else if (!is_same<cl_float, T>::value)
		throw runtime_error(
				"The template type is not valid. This is an indication of programming error");

	if (activationFunction == ATMLSigmoidActivation)
		stringStream << "-D" << "SIGMOID ";
	else if (activationFunction == ATMLTanhActivation)
		stringStream << "-D" << "TANH ";

	if (useRelaxedMath)
		stringStream << "-cl-fast-relaxed-math";

	compilerOptions = stringStream.str();
}

template<class T>
string BackPerceptronKernel<T>::ProgramName() const {
	return programName;
}

template<class T>
string BackPerceptronKernel<T>::GetCompilerOptions() const {
	return compilerOptions;
}

template<class T>
vector<string> BackPerceptronKernel<T>::GetProgramCode() const {
	vector<string> result;
	result.push_back(
			FileHelper::GetTextFromPath(
					Path::Combine(
							Path::GetDirectoryPath(
									FileHelper::GetExecutablePath()), "kernels",
							"BackPropPerceptronKernel.cl")));
	return result;
}

template<class T>
string BackPerceptronKernel<T>::KernelName() const {
	return kernelName;
}

template<class T>
const vector<size_t>& BackPerceptronKernel<T>::GlobalWorkSize() const {
	return globalWorkSize;
}

template<class T>
const vector<size_t>& BackPerceptronKernel<T>::LocalWorkSize() const {
	return localWorkSize;
}

template class BackPerceptronKernel<cl_float> ;
template class BackPerceptronKernel<cl_double> ;

} /* namespace MachineLearning */
} /* namespace ATML */
