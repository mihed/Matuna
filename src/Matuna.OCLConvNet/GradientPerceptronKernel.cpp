/*
 * GradientPerceptronKernel.cpp
 *
 *  Created on: May 17, 2015
 *      Author: Mikael
 */

#include "GradientPerceptronKernel.h"
#include "Matuna.OCLHelper/OCLUtility.h"
#include "Matuna.Helper/FileHelper.h"
#include "Matuna.Helper/Path.h"

namespace Matuna {
namespace MachineLearning {

template<class T>
GradientPerceptronKernel<T>::GradientPerceptronKernel(int inputUnits, int units) :
		inputUnits(inputUnits), units(units) {
	stringstream stringStream;

	stringStream << "GradientPerceptronKernelProgram";
	stringStream << instanceCounter;
	programName = stringStream.str();
	kernelName = "GradientPerceptronKernel";

	useConstantInput = false;
	useConstantInputDelta = false;
	useRelaxedMath = false;

	globalWorkSize.push_back(inputUnits);
	globalWorkSize.push_back(units);

}
template<class T>
GradientPerceptronKernel<T>::~GradientPerceptronKernel() {

}

template<class T>
void GradientPerceptronKernel<T>::SetConstantInput(bool value) {
	useConstantInput = value;
}

template<class T>
void GradientPerceptronKernel<T>::SetConstantInputDelta(bool value) {
	useConstantInputDelta = value;
}

template<class T>
void GradientPerceptronKernel<T>::SetUseRelaxedMath(bool value) {
	useRelaxedMath = value;
}

template<class T>
void GradientPerceptronKernel<T>::SetInput(OCLMemory* input) {
	auto rawInput = input->GetCLMemory();
	CheckOCLError(
			clSetKernelArg(this->GetKernel(), 0, sizeof(cl_mem), &rawInput),
			"Could not set the kernel arguments");
}

template<class T>
void GradientPerceptronKernel<T>::SetInputDelta(OCLMemory* inputDelta) {
	auto rawInput = inputDelta->GetCLMemory();
	CheckOCLError(
			clSetKernelArg(this->GetKernel(), 1, sizeof(cl_mem), &rawInput),
			"Could not set the kernel arguments");
}

template<class T>
void GradientPerceptronKernel<T>::SetGradient(OCLMemory* gradient) {
	auto rawGradient = gradient->GetCLMemory();
	CheckOCLError(
			clSetKernelArg(this->GetKernel(), 2, sizeof(cl_mem), &rawGradient),
			"Could not set the kernel arguments");
}

template<class T>
void GradientPerceptronKernel<T>::InitializeCompilerOptions() {
	stringstream stringStream;

	if (useConstantInput)
		stringStream << "-D" << "CONSTANT_INPUT ";
	if (useConstantInputDelta)
		stringStream << "-D" << "CONSTANT_INPUT_DELTA ";

	if (is_same<cl_double, T>::value)
		stringStream << "-D" << "DOUBLE_PRECISION ";
	else if (!is_same<cl_float, T>::value)
		throw runtime_error(
				"The template type is not valid. This is an indication of programming error");

	stringStream << "-D" << "WEIGHT_COLUMN_COUNT=" << inputUnits << " ";

	if (useRelaxedMath)
		stringStream << "-cl-fast-relaxed-math ";

	string folderPath = Path::Combine(
			Path::GetDirectoryPath(FileHelper::GetExecutablePath()), "kernels/");
	stringStream << "-I" << folderPath << " ";

	compilerOptions = stringStream.str();
}

template<class T>
string GradientPerceptronKernel<T>::ProgramName() const {
	return programName;
}

template<class T>
string GradientPerceptronKernel<T>::GetCompilerOptions() const {
	return compilerOptions;
}

template<class T>
vector<string> GradientPerceptronKernel<T>::GetProgramCode() const {
	vector<string> result;
	string folderPath = Path::Combine(
			Path::GetDirectoryPath(FileHelper::GetExecutablePath()), "kernels");
	result.push_back(
			FileHelper::GetTextFromPath(
					Path::Combine(folderPath, "RealType.h")));
	result.push_back(
			FileHelper::GetTextFromPath(
					Path::Combine(folderPath, "GradientPerceptronKernel.cl")));
	return result;
}

template<class T>
string GradientPerceptronKernel<T>::KernelName() const {
	return kernelName;
}

template<class T>
const vector<size_t>& GradientPerceptronKernel<T>::GlobalWorkSize() const {
	return globalWorkSize;
}

template<class T>
const vector<size_t>& GradientPerceptronKernel<T>::LocalWorkSize() const {
	return localWorkSize;
}

template class GradientPerceptronKernel<cl_float> ;
template class GradientPerceptronKernel<cl_double> ;

} /* namespace MachineLearning */
} /* namespace Matuna */