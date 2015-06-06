/*
 * ImageBackPerceptronKernel.cpp
 *
 *  Created on: Jun 2, 2015
 *      Author: Mikael
 */

#include "ImageBackPerceptronKernel.h"
#include "OpenCLHelper/OpenCLUtility.h"
#include "Helper/FileHelper.h"
#include "Helper/Path.h"

namespace Matuna
{
namespace MachineLearning
{

template<class T>
ImageBackPerceptronKernel<T>::ImageBackPerceptronKernel(int globalWidth,
		int globalHeight, int globalUnits, int outputWidthOffset,
		int outputHeightOffset, int outputUnitOffset, int inputWidthOffset,
		int inputHeightOffset, int inputUnitOffset, int outputStride,
		int outputMemoryHeight, int inputStride, int inputMemoryHeight,
		int weightColumnCount, int inputDeltaOffset, int inputDeltaDataUnits) :
		globalWidth(globalWidth), globalHeight(globalHeight), globalUnits(
				globalUnits), outputWidthOffset(outputWidthOffset), outputHeightOffset(
				outputHeightOffset), outputUnitOffset(outputUnitOffset), inputWidthOffset(
				inputWidthOffset), inputHeightOffset(inputHeightOffset), inputUnitOffset(
				inputUnitOffset), outputStride(outputStride), outputMemoryHeight(
				outputMemoryHeight), inputStride(inputStride), inputMemoryHeight(
				inputMemoryHeight), weightColumnCount(weightColumnCount), inputDeltaOffset(
				inputDeltaOffset), inputDeltaDataUnits(inputDeltaDataUnits)
{
	useRelaxedMath = false;
	useConstantInput = false;
	useConstantDeltaInput = false;
	useConstantWeights = false;
	kernelName = "BackPerceptronKernel";
	activationFunction = MatunaLinearActivation;
	stringstream stringStream;

	stringStream << "ImageBackPerceptronProgram";
	stringStream << instanceCounter;
	programName = stringStream.str();

	globalWorkSize.push_back(globalWidth);
	globalWorkSize.push_back(globalHeight);
	globalWorkSize.push_back(globalUnits);
}

template<class T>
ImageBackPerceptronKernel<T>::~ImageBackPerceptronKernel()
{

}

template<class T>
void ImageBackPerceptronKernel<T>::SetUseConstantWeights(bool value)
{
	useConstantWeights = value;
}

template<class T>
void ImageBackPerceptronKernel<T>::SetUseConstantInput(bool value)
{
	useConstantInput = value;
}

template<class T>
void ImageBackPerceptronKernel<T>::SetUseConstantDeltaInput(bool value)
{
	useConstantDeltaInput = value;
}

template<class T>
void ImageBackPerceptronKernel<T>::SetUseRelaxedMath(bool value)
{
	useRelaxedMath = value;
}

template<class T>
void ImageBackPerceptronKernel<T>::SetActivationFunction(
		MatunaActivationFunction activationFunction)
{
	this->activationFunction = activationFunction;
}

template<class T>
void ImageBackPerceptronKernel<T>::SetWeights(OpenCLMemory* weights)
{
	auto rawWeights = weights->GetCLMemory();
	CheckOpenCLError(
			clSetKernelArg(this->GetKernel(), 3, sizeof(cl_mem), &rawWeights),
			"Could not set the kernel arguments");
}

template<class T>
void ImageBackPerceptronKernel<T>::SetInput(OpenCLMemory* input)
{
	auto rawInput = input->GetCLMemory();
	CheckOpenCLError(
			clSetKernelArg(this->GetKernel(), 0, sizeof(cl_mem), &rawInput),
			"Could not set the kernel arguments");
}

template<class T>
void ImageBackPerceptronKernel<T>::SetDeltaInput(OpenCLMemory* deltaInput)
{
	auto rawInput = deltaInput->GetCLMemory();
	CheckOpenCLError(
			clSetKernelArg(this->GetKernel(), 1, sizeof(cl_mem), &rawInput),
			"Could not set the kernel arguments");
}

template<class T>
void ImageBackPerceptronKernel<T>::SetOutput(OpenCLMemory* output)
{
	auto rawOutput = output->GetCLMemory();
	CheckOpenCLError(
			clSetKernelArg(this->GetKernel(), 2, sizeof(cl_mem), &rawOutput),
			"Could not set the kernel arguments");
}

template<class T>
void ImageBackPerceptronKernel<T>::InitializeCompilerOptions()
{
	stringstream stringStream;

	if (useConstantWeights)
		stringStream << "-D" << "CONSTANT_WEIGHTS ";
	if (useConstantInput)
		stringStream << "-D" << "CONSTANT_INPUT ";
	if (useConstantDeltaInput)
		stringStream << "-D" << "CONSTANT_INPUT_DELTA ";

	stringStream << "-D" << "OUTPUT_WIDTH_OFFSET=" << outputWidthOffset << " ";
	stringStream << "-D" << "OUTPUT_HEIGHT_OFFSET=" << outputHeightOffset
			<< " ";
	stringStream << "-D" << "OUTPUT_UNIT_OFFSET=" << outputUnitOffset << " ";
	stringStream << "-D" << "OUTPUT_STRIDE=" << outputStride << " ";
	stringStream << "-D" << "OUTPUT_UNIT_ELEMENT_COUNT_INC_PADDING="
			<< (outputStride * outputMemoryHeight) << " ";
	stringStream << "-D" << "INPUT_WIDTH_OFFSET=" << inputWidthOffset << " ";
	stringStream << "-D" << "INPUT_HEIGHT_OFFSET=" << inputHeightOffset << " ";
	stringStream << "-D" << "INPUT_UNIT_OFFSET=" << inputUnitOffset << " ";
	stringStream << "-D" << "INPUT_STRIDE=" << inputStride << " ";
	stringStream << "-D" << "INPUT_UNIT_ELEMENT_COUNT_INC_PADDING="
			<< (inputStride * inputMemoryHeight) << " ";
	stringStream << "-D" << "INPUT_DELTA_OFFSET=" << inputDeltaOffset << " ";
	stringStream << "-D" << "INPUT_DELTA_LIMIT="
			<< (inputDeltaDataUnits + inputDeltaOffset) << " ";
	stringStream << "-D" << "WEIGHT_COLUMN_COUNT=" << weightColumnCount << " ";

	if (is_same<cl_double, T>::value)
		stringStream << "-D" << "DOUBLE_PRECISION ";
	else if (!is_same<cl_float, T>::value)
		throw runtime_error(
				"The template type is not valid. This is an indication of programming error");

	if (activationFunction == MatunaSigmoidActivation)
		stringStream << "-D" << "SIGMOID ";
	else if (activationFunction == MatunaTanhActivation)
		stringStream << "-D" << "TANH ";

	if (useRelaxedMath)
		stringStream << "-cl-fast-relaxed-math";

	compilerOptions = stringStream.str();
}

template<class T>
string ImageBackPerceptronKernel<T>::ProgramName() const
{
	return programName;
}

template<class T>
string ImageBackPerceptronKernel<T>::GetCompilerOptions() const
{
	return compilerOptions;
}

template<class T>
vector<string> ImageBackPerceptronKernel<T>::GetProgramCode() const
{
	vector<string> result;
	result.push_back(
			FileHelper::GetTextFromPath(
					Path::Combine(
							Path::GetDirectoryPath(
									FileHelper::GetExecutablePath()), "kernels",
							"ImageBackPropPerceptronKernel.cl")));
	return result;
}

template<class T>
string ImageBackPerceptronKernel<T>::KernelName() const
{
	return kernelName;
}

template<class T>
const vector<size_t>& ImageBackPerceptronKernel<T>::GlobalWorkSize() const
{
	return globalWorkSize;
}

template<class T>
const vector<size_t>& ImageBackPerceptronKernel<T>::LocalWorkSize() const
{
	return localWorkSize;
}

template class ImageBackPerceptronKernel<cl_float> ;
template class ImageBackPerceptronKernel<cl_double> ;

} /* namespace MachineLearning */
} /* namespace Matuna */
