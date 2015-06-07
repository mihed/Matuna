/*
 * ImageGradientPerceptronKernel.cpp
 *
 *  Created on: Jun 3, 2015
 *      Author: Mikael
 */

#include "ImageGradientPerceptronKernel.h"
#include "Matuna.OCLHelper/OCLUtility.h"
#include "Matuna.Helper/FileHelper.h"
#include "Matuna.Helper/Path.h"

namespace Matuna
{
namespace MachineLearning
{

template<class T>
ImageGradientPerceptronKernel<T>::ImageGradientPerceptronKernel(int weightWidth,
		int weightHeight, int inputDataWidth, int inputDataHeight,
		int inputWidthOffset, int inputHeightOffset, int inputUnitOffset,
		int inputStride, int inputMemoryHeight, int inputDeltaOffset,
		int weightColumnCount) :
		weightWidth(weightWidth), weightHeight(weightHeight), inputDataWidth(
				inputDataWidth), inputDataHeight(inputDataHeight), inputWidthOffset(
				inputWidthOffset), inputHeightOffset(inputHeightOffset), inputUnitOffset(
				inputUnitOffset), inputStride(inputStride), inputMemoryHeight(
				inputMemoryHeight), inputDeltaOffset(inputDeltaOffset), weightColumnCount(
				weightColumnCount)
{
	stringstream stringStream;

	stringStream << "ImageGradientPerceptronKernelProgram";
	stringStream << instanceCounter;
	programName = stringStream.str();
	kernelName = "ImageGradientPerceptronKernel";

	useConstantInput = false;
	useConstantInputDelta = false;
	useRelaxedMath = false;
	globalWorkSize.push_back(weightWidth);
	globalWorkSize.push_back(weightHeight);
}

template<class T>
ImageGradientPerceptronKernel<T>::~ImageGradientPerceptronKernel()
{

}

template<class T>
void ImageGradientPerceptronKernel<T>::SetConstantInput(bool value)
{
	useConstantInput = value;
}

template<class T>
void ImageGradientPerceptronKernel<T>::SetConstantInputDelta(bool value)
{
	useConstantInputDelta = value;
}

template<class T>
void ImageGradientPerceptronKernel<T>::SetUseRelaxedMath(bool value)
{
	useRelaxedMath = value;
}

template<class T>
void ImageGradientPerceptronKernel<T>::SetInput(OCLMemory* input)
{
	auto rawInput = input->GetCLMemory();
	CheckOCLError(
			clSetKernelArg(this->GetKernel(), 0, sizeof(cl_mem), &rawInput),
			"Could not set the kernel arguments");
}

template<class T>
void ImageGradientPerceptronKernel<T>::SetInputDelta(OCLMemory* inputDelta)
{
	auto rawInput = inputDelta->GetCLMemory();
	CheckOCLError(
			clSetKernelArg(this->GetKernel(), 1, sizeof(cl_mem), &rawInput),
			"Could not set the kernel arguments");
}

template<class T>
void ImageGradientPerceptronKernel<T>::SetGradient(OCLMemory* gradient)
{
	auto rawGradient = gradient->GetCLMemory();
	CheckOCLError(
			clSetKernelArg(this->GetKernel(), 2, sizeof(cl_mem), &rawGradient),
			"Could not set the kernel arguments");
}

template<class T>
void ImageGradientPerceptronKernel<T>::InitializeCompilerOptions()
{
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

	stringStream << "-D" << "INPUT_DATA_WIDTH=" << inputDataWidth << " ";
	stringStream << "-D" << "INPUT_UNIT_ELEMENT_COUNT="
			<< (inputDataHeight * inputDataWidth) << " ";
	stringStream << "-D" << "INPUT_WIDTH_OFFSET=" << inputWidthOffset << " ";
	stringStream << "-D" << "INPUT_HEIGHT_OFFSET=" << inputHeightOffset << " ";
	stringStream << "-D" << "INPUT_UNIT_OFFSET=" << inputUnitOffset << " ";
	stringStream << "-D" << "INPUT_STRIDE=" << inputStride << " ";
	stringStream << "-D" << "INPUT_UNIT_ELEMENT_COUNT_INC_PADDING="
			<< (inputStride * inputMemoryHeight) << " ";
	stringStream << "-D" << "INPUT_DELTA_OFFSET=" << inputDeltaOffset << " ";
	stringStream << "-D" << "WEIGHT_COLUMN_COUNT=" << weightColumnCount << " ";

	if (useRelaxedMath)
		stringStream << "-cl-fast-relaxed-math";

	compilerOptions = stringStream.str();
}

template<class T>
string ImageGradientPerceptronKernel<T>::ProgramName() const
{
	return programName;
}

template<class T>
string ImageGradientPerceptronKernel<T>::GetCompilerOptions() const
{
	return compilerOptions;
}

template<class T>
vector<string> ImageGradientPerceptronKernel<T>::GetProgramCode() const
{
	vector<string> result;
	result.push_back(
			FileHelper::GetTextFromPath(
					Path::Combine(
							Path::GetDirectoryPath(
									FileHelper::GetExecutablePath()), "kernels",
							"ImageGradientPerceptronKernel.cl")));
	return result;
}

template<class T>
string ImageGradientPerceptronKernel<T>::KernelName() const
{
	return kernelName;
}

template<class T>
const vector<size_t>& ImageGradientPerceptronKernel<T>::GlobalWorkSize() const
{
	return globalWorkSize;
}

template<class T>
const vector<size_t>& ImageGradientPerceptronKernel<T>::LocalWorkSize() const
{
	return localWorkSize;
}

template class ImageGradientPerceptronKernel<cl_float> ;
template class ImageGradientPerceptronKernel<cl_double> ;

} /* namespace MachineLearning */
} /* namespace Matuna */
