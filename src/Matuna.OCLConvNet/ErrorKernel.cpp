/*
 * ErrorKernel.cpp
 *
 *  Created on: May 15, 2015
 *      Author: Mikael
 */

#include "ErrorKernel.h"
#include "Matuna.OCLHelper/OCLUtility.h"
#include "Matuna.Helper/FileHelper.h"
#include "Matuna.Helper/Path.h"

namespace Matuna {
namespace MachineLearning {

template<class T>
ErrorKernel<T>::ErrorKernel(int units, int unitOffset) :
		units(units), unitOffset(unitOffset) {
	stringstream stringStream;

	stringStream << "OutputErrorProgram";
	stringStream << instanceCounter;
	programName = stringStream.str();
	kernelName = "Error";

	useConstantInput = false;
	useConstantTarget = false;
	useRelaxedMath = false;
	errorFunction = MatunaMeanSquareError;
	computationPrecision = MatunaNormalPrecision;
}

template<class T>
ErrorKernel<T>::~ErrorKernel() {

}

template<class T>
void ErrorKernel<T>::SetConstantInput(bool value) {
	useConstantInput = value;
}

template<class T>
void ErrorKernel<T>::SetConstantTarget(bool value) {
	useConstantTarget = value;
}

template<class T>
void ErrorKernel<T>::SetUseRelaxedMath(bool value) {
	useRelaxedMath = value;
}

template<class T>
void ErrorKernel<T>::SetErrorFunction(MatunaErrorFunction errorFunction) {
	this->errorFunction = errorFunction;
}

template<class T>
void ErrorKernel<T>::SetComputationPrecision(
		MatunaComputationPrecision computationPrecision) {
	this->computationPrecision = computationPrecision;
}

template<class T>
void ErrorKernel<T>::SetInput(OCLMemory* input) {
	auto rawInput = input->GetCLMemory();
	CheckOCLError(
			clSetKernelArg(this->GetKernel(), 0, sizeof(cl_mem), &rawInput),
			"Could not set the kernel arguments");
}

template<class T>
void ErrorKernel<T>::SetTarget(OCLMemory* target) {
	auto rawTarget = target->GetCLMemory();
	CheckOCLError(
			clSetKernelArg(this->GetKernel(), 1, sizeof(cl_mem), &rawTarget),
			"Could not set the kernel arguments");
}

template<class T>
void ErrorKernel<T>::SetError(OCLMemory* error) {
	auto rawError = error->GetCLMemory();
	CheckOCLError(
			clSetKernelArg(this->GetKernel(), 2, sizeof(cl_mem), &rawError),
			"Could not set the kernel arguments");
}

template<class T>
void ErrorKernel<T>::InitializeCompilerOptions() {
	stringstream stringStream;

	if (useConstantInput)
		stringStream << "-D" << "CONSTANT_INPUT ";
	if (useConstantTarget)
		stringStream << "-D" << "CONSTANT_TARGET ";

	//Refer to the notes for this
	if (errorFunction == MatunaMeanSquareError) {
		stringStream << "-D" << "MSE ";
	} else if (errorFunction == MatunaCrossEntropy) {
		if (units == 1)
			stringStream << "-D" << "CE_BINARY ";
		else
			stringStream << "-D" << "CE ";
	} else
		throw invalid_argument(
				"The error function is not supported by the output kernel");

	stringStream << "-D" << "INPUT_UNIT_OFFSET=" << unitOffset << " ";

	if (is_same<cl_double, T>::value)
		stringStream << "-D" << "DOUBLE_PRECISION ";
	else if (!is_same<cl_float, T>::value)
		throw runtime_error(
				"The template type is not valid. This is an indication of programming error");

	stringStream << "-D" << "INPUT_COUNT=" << units << " ";

	if (computationPrecision == MatunaNativePrecision)
		stringStream << "-D" << "NATIVE_MATH ";
	else if (computationPrecision == MatunaHalfPrecision)
		stringStream << "-D" << "HALF_MATH ";

	if (useRelaxedMath)
		stringStream << "-cl-fast-relaxed-math ";

	string folderPath = Path::Combine(
			Path::GetDirectoryPath(FileHelper::GetExecutablePath()), "kernels/");
	stringStream << "-I " << folderPath << " ";

	compilerOptions = stringStream.str();
}

template<class T>
string ErrorKernel<T>::ProgramName() const {
	return programName;
}

template<class T>
string ErrorKernel<T>::GetCompilerOptions() const {
	return compilerOptions;
}

template<class T>
vector<string> ErrorKernel<T>::GetProgramCode() const {
	vector<string> result;
	string folderPath = Path::Combine(
			Path::GetDirectoryPath(FileHelper::GetExecutablePath()), "kernels");
	result.push_back(
			FileHelper::GetTextFromPath(
					Path::Combine(folderPath, "RealType.h")));
	result.push_back(
			FileHelper::GetTextFromPath(
					Path::Combine(folderPath, "OutputError.cl")));
	return result;
}

template<class T>
string ErrorKernel<T>::KernelName() const {
	return kernelName;
}

template<class T>
const vector<size_t>& ErrorKernel<T>::GlobalWorkSize() const {
	return globalWorkSize;
}

template<class T>
const vector<size_t>& ErrorKernel<T>::LocalWorkSize() const {
	return localWorkSize;
}

template class ErrorKernel<cl_float> ;
template class ErrorKernel<cl_double> ;

} /* namespace MachineLearning */
} /* namespace Matuna */
