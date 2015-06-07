/*
 * SimpleSumKernel.cpp
 *
 *  Created on: May 19, 2015
 *      Author: Mikael
 */

#include "SimpleSumKernel.h"
#include "Matuna.OCLHelper/OCLUtility.h"
#include "Matuna.Helper/FileHelper.h"
#include "Matuna.Helper/Path.h"

namespace Matuna {
namespace MachineLearning {

template<class T>
SimpleSumKernel<T>::SimpleSumKernel(int inputCount) :
		inputCount(inputCount) {
	stringstream stringStream;

	stringStream << "SimpleSumProgram";
	stringStream << instanceCounter;
	programName = stringStream.str();
	kernelName = "SimpleSumKernel";

	stringStream.str("");
	stringStream.clear();

	stringStream << "-D" << "INPUT_COUNT=" << inputCount;
	if (is_same<cl_double, T>::value)
		stringStream << " -D" << "DOUBLE_PRECISION ";
	else if (!is_same<cl_float, T>::value)
		throw runtime_error(
				"The template type is not valid. This is an indication of programming error");

	string folderPath = Path::Combine(
			Path::GetDirectoryPath(FileHelper::GetExecutablePath()), "kernels");
	stringStream << " -I" << folderPath << " ";

	compilerOptions = stringStream.str();
}

template<class T>
SimpleSumKernel<T>::~SimpleSumKernel() {

}

template<class T>
void SimpleSumKernel<T>::SetInput(OCLMemory* input) {
	auto rawInput = input->GetCLMemory();
	CheckOCLError(
			clSetKernelArg(this->GetKernel(), 0, sizeof(cl_mem), &rawInput),
			"Could not set the kernel arguments");
}

template<class T>
void SimpleSumKernel<T>::SetOutput(OCLMemory* output) {
	auto rawOutput = output->GetCLMemory();
	CheckOCLError(
			clSetKernelArg(this->GetKernel(), 1, sizeof(cl_mem), &rawOutput),
			"Could not set the kernel arguments");
}

template<class T>
string SimpleSumKernel<T>::ProgramName() const {
	return programName;
}

template<class T>
string SimpleSumKernel<T>::GetCompilerOptions() const {
	return compilerOptions;
}

template<class T>
vector<string> SimpleSumKernel<T>::GetProgramCode() const {
	vector<string> result;
	string folderPath = Path::Combine(
			Path::GetDirectoryPath(FileHelper::GetExecutablePath()), "kernels");
	result.push_back(
			FileHelper::GetTextFromPath(
					Path::Combine(folderPath, "RealType.h")));
	result.push_back(
			FileHelper::GetTextFromPath(
					Path::Combine(folderPath, "SimpleSumKernel.cl")));
	return result;
}

template<class T>
string SimpleSumKernel<T>::KernelName() const {
	return kernelName;
}

template<class T>
const vector<size_t>& SimpleSumKernel<T>::GlobalWorkSize() const {
	return globalWorkSize;
}

template<class T>
const vector<size_t>& SimpleSumKernel<T>::LocalWorkSize() const {
	return localWorkSize;
}

template class SimpleSumKernel<cl_float> ;
template class SimpleSumKernel<cl_double> ;

} /* namespace MachineLearning */
} /* namespace Matuna */
