/*
 * OutputKernel.cpp
 *
 *  Created on: May 15, 2015
 *      Author: Mikael
 */

#include "OutputKernel.h"
#include "Matuna.OCLHelper/OCLUtility.h"
#include "Matuna.Helper/FileHelper.h"
#include "Matuna.Helper/Path.h"

namespace Matuna {
namespace MachineLearning {

template<class T>
OutputKernel<T>::OutputKernel(int unitsCount, int inputOffset, int outputOffset) :
		unitsCount(unitsCount), inputOffset(inputOffset), outputOffset(
				outputOffset) {
	stringstream stringStream;

	stringStream << "OutputBackPropProgram";
	stringStream << instanceCounter;
	programName = stringStream.str();
	kernelName = "BackPropagation";

	useConstantInput = false;
	useConstantTarget = false;
	useRelaxedMath = false;
	activationFunction = MatunaSigmoidActivation;
	computationPrecision = MatunaNormalPrecision;
	errorFunction = MatunaMeanSquareError;

	globalWorkSize.push_back(unitsCount);
}

template<class T>
OutputKernel<T>::~OutputKernel() {

}

template<class T>
void OutputKernel<T>::SetConstantInput(bool value) {
	useConstantInput = value;
}

template<class T>
void OutputKernel<T>::SetConstantTarget(bool value) {
	useConstantTarget = value;
}

template<class T>
void OutputKernel<T>::SetUseRelaxedMath(bool value) {
	useRelaxedMath = value;
}

template<class T>
void OutputKernel<T>::SetActivationFunction(
		MatunaActivationFunction activationFunction) {
	this->activationFunction = activationFunction;
}

template<class T>
void OutputKernel<T>::SetComputationPrecision(
		MatunaComputationPrecision computationPrecision) {
	this->computationPrecision = computationPrecision;
}

template<class T>
void OutputKernel<T>::SetErrorFunction(MatunaErrorFunction errorFunction) {
	this->errorFunction = errorFunction;
}

template<class T>
void OutputKernel<T>::SetInput(OCLMemory* input) {
	auto rawInput = input->GetCLMemory();
	CheckOCLError(
			clSetKernelArg(this->GetKernel(), 0, sizeof(cl_mem), &rawInput),
			"Could not set the kernel arguments");
}

template<class T>
void OutputKernel<T>::SetTarget(OCLMemory* target) {
	auto rawTarget = target->GetCLMemory();
	CheckOCLError(
			clSetKernelArg(this->GetKernel(), 1, sizeof(cl_mem), &rawTarget),
			"Could not set the kernel arguments");
}

template<class T>
void OutputKernel<T>::SetOutput(OCLMemory* output) {
	auto rawOutput = output->GetCLMemory();
	CheckOCLError(
			clSetKernelArg(this->GetKernel(), 2, sizeof(cl_mem), &rawOutput),
			"Could not set the kernel arguments");
}

template<class T>
void OutputKernel<T>::InitializeCompilerOptions() {
	stringstream stringStream;

	if (useConstantInput)
		stringStream << "-D" << "CONSTANT_INPUT ";
	if (useConstantTarget)
		stringStream << "-D" << "CONSTANT_TARGET ";

	//Refer to the notes for this
	if (errorFunction == MatunaMeanSquareError) {
		if (activationFunction == MatunaLinearActivation)
			stringStream << "-D" << "DIFFERENCE ";
		else
			stringStream << "-D" << "MSE_ANY ";
	} else if (errorFunction == MatunaCrossEntropy) {
		if (unitsCount == 1) {
			if (activationFunction == MatunaSigmoidActivation)
				stringStream << "-D" << "DIFFERENCE ";
			else
				stringStream << "-D" << "CE_BINARY_ANY ";
		} else {
			if (activationFunction == MatunaSoftMaxActivation)
				stringStream << "-D" << "DIFFERENCE ";
			else
				stringStream << "-D" << "CE_ANY ";
		}
	} else
		throw invalid_argument(
				"The error function is not supported by the output kernel");

	stringStream << "-D" << "INPUT_UNIT_OFFSET=" << inputOffset << " ";
	stringStream << "-D" << "OUTPUT_UNIT_OFFSET=" << outputOffset << " ";

	if (is_same<cl_double, T>::value)
		stringStream << "-D" << "DOUBLE_PRECISION ";
	else if (!is_same<cl_float, T>::value)
		throw runtime_error(
				"The template type is not valid. This is an indication of programming error");

	if (activationFunction == MatunaSigmoidActivation)
		stringStream << "-D" << "SIGMOID ";
	else if (activationFunction == MatunaTanhActivation)
		stringStream << "-D" << "TANH ";

	if (computationPrecision == MatunaNativePrecision)
		stringStream << "-D" << "NATIVE_MATH ";
	else if (computationPrecision == MatunaHalfPrecision)
		stringStream << "-D" << "HALF_MATH ";

	if (useRelaxedMath)
		stringStream << "-cl-fast-relaxed-math";

	compilerOptions = stringStream.str();

}

template<class T>
string OutputKernel<T>::ProgramName() const {
	return programName;
}

template<class T>
string OutputKernel<T>::GetCompilerOptions() const {
	return compilerOptions;
}

template<class T>
vector<string> OutputKernel<T>::GetProgramCode() const {
	vector<string> result;
	result.push_back(
			FileHelper::GetTextFromPath(
					Path::Combine(
							Path::GetDirectoryPath(
									FileHelper::GetExecutablePath()), "kernels",
							"OutputBackProp.cl")));
	return result;
}

template<class T>
string OutputKernel<T>::KernelName() const {
	return kernelName;
}

template<class T>
const vector<size_t>& OutputKernel<T>::GlobalWorkSize() const {
	return globalWorkSize;
}

template<class T>
const vector<size_t>& OutputKernel<T>::LocalWorkSize() const {
	return localWorkSize;
}

template class OutputKernel<cl_float> ;
template class OutputKernel<cl_double> ;

} /* namespace MachineLearning */
} /* namespace Matuna */
