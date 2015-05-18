/*
 * GradientPerceptronKernel.cpp
 *
 *  Created on: May 17, 2015
 *      Author: Mikael
 */

#include "GradientPerceptronKernel.h"
#include "OpenCLHelper/OpenCLUtility.h"
#include "Helper/FileHelper.h"
#include "Helper/Path.h"

namespace ATML {
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
void GradientPerceptronKernel<T>::SetInput(OpenCLMemory* input) {
	auto rawInput = input->GetCLMemory();
	CheckOpenCLError(
			clSetKernelArg(this->GetKernel(), 0, sizeof(cl_mem), &rawInput),
			"Could not set the kernel arguments");
}

template<class T>
void GradientPerceptronKernel<T>::SetInputDelta(OpenCLMemory* inputDelta) {
	auto rawInput = inputDelta->GetCLMemory();
	CheckOpenCLError(
			clSetKernelArg(this->GetKernel(), 1, sizeof(cl_mem), &rawInput),
			"Could not set the kernel arguments");
}

template<class T>
void GradientPerceptronKernel<T>::SetGradient(OpenCLMemory* gradient) {
	auto rawGradient = gradient->GetCLMemory();
	CheckOpenCLError(
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
		stringStream << "-cl-fast-relaxed-math";

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
	result.push_back(
			FileHelper::GetTextFromPath(
					Path::Combine(
							Path::GetDirectoryPath(
									FileHelper::GetExecutablePath()), "kernels",
							"GradientPerceptronKernel.cl")));
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
} /* namespace ATML */
