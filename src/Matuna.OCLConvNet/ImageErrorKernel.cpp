/*
 * ImageErrorKernel.cpp
 *
 *  Created on: Jun 1, 2015
 *      Author: Mikael
 */

#include "ImageErrorKernel.h"
#include "Matuna.OCLHelper/OCLUtility.h"
#include "Matuna.Helper/FileHelper.h"
#include "Matuna.Helper/Path.h"

namespace Matuna {
namespace MachineLearning {

template<class T>
ImageErrorKernel<T>::ImageErrorKernel(int dataWidth, int dataHeight,
		int dataUnits, int inputWidthOffset, int inputHeightOffset,
		int inputUnitOffset, int inputStride, int inputMemoryHeight) :
		dataWidth(dataWidth), dataHeight(dataHeight), dataUnits(dataUnits), inputWidthOffset(
				inputWidthOffset), inputHeightOffset(inputHeightOffset), inputUnitOffset(
				inputUnitOffset), inputStride(inputStride), inputMemoryHeight(
				inputMemoryHeight) {
	stringstream stringStream;

	stringStream << "ImageOutputErrorProgram";
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
ImageErrorKernel<T>::~ImageErrorKernel() {
	// TODO Auto-generated destructor stub
}

template<class T>
void ImageErrorKernel<T>::SetConstantInput(bool value) {
	useConstantInput = value;
}

template<class T>
void ImageErrorKernel<T>::SetConstantTarget(bool value) {
	useConstantTarget = value;
}

template<class T>
void ImageErrorKernel<T>::SetUseRelaxedMath(bool value) {
	useRelaxedMath = value;
}

template<class T>
void ImageErrorKernel<T>::SetErrorFunction(MatunaErrorFunction errorFunction) {
	this->errorFunction = errorFunction;
}

template<class T>
void ImageErrorKernel<T>::SetComputationPrecision(
		MatunaComputationPrecision computationPrecision) {
	this->computationPrecision = computationPrecision;
}

template<class T>
void ImageErrorKernel<T>::SetInput(OCLMemory* input) {
	auto rawInput = input->GetCLMemory();
	CheckOCLError(
			clSetKernelArg(this->GetKernel(), 0, sizeof(cl_mem), &rawInput),
			"Could not set the kernel arguments");
}

template<class T>
void ImageErrorKernel<T>::SetTarget(OCLMemory* target) {
	auto rawTarget = target->GetCLMemory();
	CheckOCLError(
			clSetKernelArg(this->GetKernel(), 1, sizeof(cl_mem), &rawTarget),
			"Could not set the kernel arguments");
}

template<class T>
void ImageErrorKernel<T>::SetError(OCLMemory* error) {
	auto rawError = error->GetCLMemory();
	CheckOCLError(
			clSetKernelArg(this->GetKernel(), 2, sizeof(cl_mem), &rawError),
			"Could not set the kernel arguments");
}

template<class T>
void ImageErrorKernel<T>::InitializeCompilerOptions() {
	stringstream stringStream;

	if (useConstantInput)
		stringStream << "-D" << "CONSTANT_INPUT ";
	if (useConstantTarget)
		stringStream << "-D" << "CONSTANT_TARGET ";

	//Refer to the notes for this
	if (errorFunction == MatunaMeanSquareError) {
		stringStream << "-D" << "MSE ";
	} else if (errorFunction == MatunaCrossEntropy) {
		if (dataUnits == 1 && dataWidth == 1 && dataHeight == 1)
			stringStream << "-D" << "CE_BINARY ";
		else
			stringStream << "-D" << "CE ";
	} else
		throw invalid_argument(
				"The error function is not supported by the output kernel");

	stringStream << "-D" << "INPUT_OFFSET_WIDTH=" << inputWidthOffset << " ";
	stringStream << "-D" << "INPUT_WIDTH_LIMIT="
			<< (inputWidthOffset + dataWidth) << " ";
	stringStream << "-D" << "INPUT_HEIGHT_LIMIT="
			<< (inputHeightOffset + dataHeight) << " ";
	stringStream << "-D" << "INPUT_OFFSET_HEIGHT=" << inputHeightOffset << " ";
	stringStream << "-D" << "INPUT_UNIT_OFFSET=" << inputUnitOffset << " ";
	stringStream << "-D" << "INPUT_STRIDE=" << inputStride << " ";
	stringStream << "-D" << "INPUT_UNIT_LIMIT=" << (inputUnitOffset + dataUnits)
			<< " ";
	stringStream << "-D" << "INPUT_UNIT_ELEMENT_COUNT_INC_PADDING="
			<< (inputStride * inputMemoryHeight) << " ";

	if (is_same<cl_double, T>::value)
		stringStream << "-D" << "DOUBLE_PRECISION ";
	else if (!is_same<cl_float, T>::value)
		throw runtime_error(
				"The template type is not valid. This is an indication of programming error");

	if (computationPrecision == MatunaNativePrecision)
		stringStream << "-D" << "NATIVE_MATH ";
	else if (computationPrecision == MatunaHalfPrecision)
		stringStream << "-D" << "HALF_MATH ";

	if (useRelaxedMath)
		stringStream << "-cl-fast-relaxed-math ";

	string folderPath = Path::Combine(
			Path::GetDirectoryPath(FileHelper::GetExecutablePath()), "kernels/");
	stringStream << "-I" << folderPath << " ";

	compilerOptions = stringStream.str();
}

template<class T>
string ImageErrorKernel<T>::ProgramName() const {
	return programName;
}

template<class T>
string ImageErrorKernel<T>::GetCompilerOptions() const {
	return compilerOptions;
}

template<class T>
vector<string> ImageErrorKernel<T>::GetProgramCode() const {
	vector<string> result;
	string folderPath = Path::Combine(
			Path::GetDirectoryPath(FileHelper::GetExecutablePath()), "kernels");
	result.push_back(
			FileHelper::GetTextFromPath(
					Path::Combine(folderPath, "RealType.h")));
	result.push_back(
			FileHelper::GetTextFromPath(
					Path::Combine(folderPath, "ImageOutputError.cl")));
	return result;
}

template<class T>
string ImageErrorKernel<T>::KernelName() const {
	return kernelName;
}

template<class T>
const vector<size_t>& ImageErrorKernel<T>::GlobalWorkSize() const {
	return globalWorkSize;
}

template<class T>
const vector<size_t>& ImageErrorKernel<T>::LocalWorkSize() const {
	return localWorkSize;
}

template class ImageErrorKernel<cl_float> ;
template class ImageErrorKernel<cl_double> ;

} /* namespace MachineLearning */
} /* namespace Matuna */
