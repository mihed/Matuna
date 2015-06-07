/*
 * PerceptronKernel.cpp
 *
 *  Created on: May 5, 2015
 *      Author: Mikael
 */

#include "ForwardPerceptronKernel.h"
#include "Matuna.OCLHelper/OCLUtility.h"
#include "Matuna.Helper/Path.h"
#include "Matuna.Helper/FileHelper.h"
#include <sstream>
#include <type_traits>

namespace Matuna {
namespace MachineLearning {

template<class T>
ForwardPerceptronKernel<T>::ForwardPerceptronKernel(int inputUnitsCount,
		int unitsCount) :
		inputUnitsCount(inputUnitsCount), unitsCount(unitsCount) {
	stringstream stringStream;

	stringStream << "ForwardPerceptronProgram";
	stringStream << instanceCounter;
	programName = stringStream.str();
	kernelName = "ForwardPerceptronKernel";

	useConstantWeights = false;
	useConstantInput = false;
	useConstantBiases = false;
	useRelaxedMath = false;
	activationFunction = MatunaSigmoidActivation;
	computationPrecision = MatunaNormalPrecision;
	biases = nullptr;
	weights = nullptr;

	globalWorkSize.push_back(unitsCount);
}

template<class T>
ForwardPerceptronKernel<T>::~ForwardPerceptronKernel() {

}

template<class T>
void ForwardPerceptronKernel<T>::InitializeArguments() {
	auto rawWeights = weights->GetCLMemory();
	auto rawBiases = biases->GetCLMemory();
	CheckOCLError(
			clSetKernelArg(this->GetKernel(), 2, sizeof(cl_mem), &rawWeights),
			"Could not set the kernel arguments");
	CheckOCLError(
			clSetKernelArg(this->GetKernel(), 3, sizeof(cl_mem), &rawBiases),
			"Could not set the kernel arguments");
}

template<class T>
void ForwardPerceptronKernel<T>::InitializeCompilerOptions() {
	stringstream stringStream;

	if (useConstantBiases)
		stringStream << "-D" << "CONSTANT_BIASES ";
	if (useConstantWeights)
		stringStream << "-D" << "CONSTANT_WEIGHTS ";
	if (useConstantInput)
		stringStream << "-D" << "CONSTANT_INPUT ";

	stringStream << "-D" << "INPUT_COUNT=" << inputUnitsCount << " ";

	if (is_same<cl_double, T>::value)
		stringStream << "-D" << "DOUBLE_PRECISION ";
	else if (!is_same<cl_float, T>::value)
		throw runtime_error(
				"The template type is not valid. This is an indication of programming error");

	if (activationFunction == MatunaSigmoidActivation)
		stringStream << "-D" << "SIGMOID ";
	else if (activationFunction == MatunaTanhActivation)
		stringStream << "-D" << "TANH ";
	else if (activationFunction == MatunaSoftMaxActivation)
		stringStream << "-D" << "SOFTMAX ";

	if (computationPrecision == MatunaNativePrecision)
		stringStream << "-D" << "NATIVE_MATH ";
	else if (computationPrecision == MatunaHalfPrecision)
		stringStream << "-D" << "HALF_MATH ";

	if (useRelaxedMath)
		stringStream << "-cl-fast-relaxed-math";

	compilerOptions = stringStream.str();
}

template<class T>
void ForwardPerceptronKernel<T>::SetUseRelaxedMath(bool value) {
	useRelaxedMath = value;
}

template<class T>
void ForwardPerceptronKernel<T>::SetInput(OCLMemory* input) {
	auto rawInput = input->GetCLMemory();
	CheckOCLError(
			clSetKernelArg(this->GetKernel(), 0, sizeof(cl_mem), &rawInput),
			"Could not set the kernel arguments");
}

template<class T>
void ForwardPerceptronKernel<T>::SetOutput(OCLMemory* output) {
	auto rawOutput = output->GetCLMemory();
	CheckOCLError(
			clSetKernelArg(this->GetKernel(), 1, sizeof(cl_mem), &rawOutput),
			"Could not set the kernel arguments");
}

template<class T>
void ForwardPerceptronKernel<T>::SetWeights(OCLMemory* weights) {
	this->weights = weights;
}

template<class T>
void ForwardPerceptronKernel<T>::SetBiases(OCLMemory* biases) {
	this->biases = biases;
}

template<class T>
void ForwardPerceptronKernel<T>::SetUseConstantWeights(bool value) {
	useConstantWeights = value;
}

template<class T>
void ForwardPerceptronKernel<T>::SetUseConstantInput(bool value) {
	useConstantInput = value;
}

template<class T>
void ForwardPerceptronKernel<T>::SetUseConstantBiases(bool value) {
	useConstantBiases = value;
}

template<class T>
void ForwardPerceptronKernel<T>::SetActivationFunction(
		MatunaActivationFunction activationFunction) {
	this->activationFunction = activationFunction;
}

template<class T>
void ForwardPerceptronKernel<T>::SetComputationPrecision(
		MatunaComputationPrecision computationPrecision) {
	this->computationPrecision = computationPrecision;
}

template<class T>
string ForwardPerceptronKernel<T>::ProgramName() const {
	return programName;
}

template<class T>
vector<string> ForwardPerceptronKernel<T>::GetProgramCode() const {
	vector<string> result;
	result.push_back(
			FileHelper::GetTextFromPath(
					Path::Combine(
							Path::GetDirectoryPath(
									FileHelper::GetExecutablePath()), "kernels",
							"ForwardPerceptronKernel.cl")));
	return result;
}

template<class T>
string ForwardPerceptronKernel<T>::KernelName() const {
	return kernelName;
}

template<class T>
string ForwardPerceptronKernel<T>::GetCompilerOptions() const {
	return compilerOptions;
}

template<class T>
const vector<size_t>& ForwardPerceptronKernel<T>::GlobalWorkSize() const {
	return globalWorkSize;
}

template<class T>
const vector<size_t>& ForwardPerceptronKernel<T>::LocalWorkSize() const {
	return localWorkSize;
}

template class ForwardPerceptronKernel<cl_float> ;
template class ForwardPerceptronKernel<cl_double> ;

} /* namespace MachineLearning */
} /* namespace Matuna */
