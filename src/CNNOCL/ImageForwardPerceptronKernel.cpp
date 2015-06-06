/*
 * ImageForwardPerceptronKernel.cpp
 *
 *  Created on: Jun 2, 2015
 *      Author: Mikael
 */

#include "ImageForwardPerceptronKernel.h"
#include "OCLHelper/OCLUtility.h"
#include "Helper/Path.h"
#include "Helper/FileHelper.h"
#include <sstream>
#include <type_traits>

namespace Matuna
{
namespace MachineLearning
{

template<class T>
ImageForwardPerceptronKernel<T>::ImageForwardPerceptronKernel(int globalUnits,
		int inputDataUnits, int inputDataWidth, int inputDataHeight,
		int inputWidthOffset, int inputHeightOffset, int inputUnitOffset,
		int inputStride, int inputMemoryHeight, int weightColumnCount,
		int outputUnitOffset) :
		globalUnits(globalUnits), inputDataUnits(inputDataUnits), inputDataWidth(
				inputDataWidth), inputDataHeight(inputDataHeight), inputWidthOffset(
				inputWidthOffset), inputHeightOffset(inputHeightOffset), inputUnitOffset(
				inputUnitOffset), inputStride(inputStride), inputMemoryHeight(
				inputMemoryHeight), weightColumnCount(weightColumnCount), outputUnitOffset(
				outputUnitOffset)
{
	stringstream stringStream;

	stringStream << "ImageForwardPerceptronProgram";
	stringStream << instanceCounter;
	programName = stringStream.str();
	kernelName = "ForwardPerceptronKernel";

	useConstantWeights = false;
	useConstantInput = false;
	useConstantBiases = false;
	useRelaxedMath = false;
	activationFunction = MatunaSigmoidActivation;
	computationPrecision = MatunaNormalPrecision;

	globalWorkSize.push_back(globalUnits);
}

template<class T>
ImageForwardPerceptronKernel<T>::~ImageForwardPerceptronKernel()
{

}

template<class T>
void ImageForwardPerceptronKernel<T>::InitializeCompilerOptions()
{
	stringstream stringStream;

	if (useConstantBiases)
		stringStream << "-D" << "CONSTANT_BIASES ";
	if (useConstantWeights)
		stringStream << "-D" << "CONSTANT_WEIGHTS ";
	if (useConstantInput)
		stringStream << "-D" << "CONSTANT_INPUT ";

	if (is_same<cl_double, T>::value)
		stringStream << "-D" << "DOUBLE_PRECISION ";
	else if (!is_same<cl_float, T>::value)
		throw runtime_error(
				"The template type is not valid. This is an indication of programming error");

	stringStream << "-D" << "INPUT_UNITS_LIMIT=" << (inputUnitOffset + inputDataUnits )<< " ";
	stringStream << "-D" << "INPUT_WIDTH_LIMIT=" << (inputWidthOffset + inputDataWidth) << " ";
	stringStream << "-D" << "INPUT_HEIGHT_LIMIT=" << (inputHeightOffset + inputDataHeight) << " ";
	stringStream << "-D" << "INPUT_UNITS_OFFSET=" << inputUnitOffset << " ";
	stringStream << "-D" << "INPUT_WIDTH_OFFSET=" << inputWidthOffset << " ";
	stringStream << "-D" << "INPUT_HEIGHT_OFFSET=" << inputHeightOffset << " ";
	stringStream << "-D" << "COLUMN_COUNT=" << weightColumnCount << " ";
	stringStream << "-D" << "OUTPUT_UNIT_OFFSET=" << outputUnitOffset << " ";
	stringStream << "-D" << "INPUT_UNIT_ELEMENT_COUNT_INC_PADDING=" << (inputStride * inputMemoryHeight) << " ";
	stringStream << "-D" << "INPUT_MEMORY_WIDTH=" << inputStride << " ";

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
void ImageForwardPerceptronKernel<T>::SetUseRelaxedMath(bool value)
{
	useRelaxedMath = value;
}

template<class T>
void ImageForwardPerceptronKernel<T>::SetInput(OCLMemory* input)
{
	auto rawInput = input->GetCLMemory();
	CheckOCLError(
			clSetKernelArg(this->GetKernel(), 0, sizeof(cl_mem), &rawInput),
			"Could not set the kernel arguments");
}

template<class T>
void ImageForwardPerceptronKernel<T>::SetOutput(OCLMemory* output)
{
	auto rawOutput = output->GetCLMemory();
	CheckOCLError(
			clSetKernelArg(this->GetKernel(), 1, sizeof(cl_mem), &rawOutput),
			"Could not set the kernel arguments");
}

template<class T>
void ImageForwardPerceptronKernel<T>::SetWeights(OCLMemory* weights)
{
	auto rawWeights = weights->GetCLMemory();
	CheckOCLError(
			clSetKernelArg(this->GetKernel(), 2, sizeof(cl_mem), &rawWeights),
			"Could not set the kernel arguments");
}

template<class T>
void ImageForwardPerceptronKernel<T>::SetBiases(OCLMemory* biases)
{
	auto rawBiases = biases->GetCLMemory();
	CheckOCLError(
			clSetKernelArg(this->GetKernel(), 3, sizeof(cl_mem), &rawBiases),
			"Could not set the kernel arguments");
}

template<class T>
void ImageForwardPerceptronKernel<T>::SetUseConstantWeights(bool value)
{
	useConstantWeights = value;
}

template<class T>
void ImageForwardPerceptronKernel<T>::SetUseConstantInput(bool value)
{
	useConstantInput = value;
}

template<class T>
void ImageForwardPerceptronKernel<T>::SetUseConstantBiases(bool value)
{
	useConstantBiases = value;
}

template<class T>
void ImageForwardPerceptronKernel<T>::SetActivationFunction(
		MatunaActivationFunction activationFunction)
{
	this->activationFunction = activationFunction;
}

template<class T>
void ImageForwardPerceptronKernel<T>::SetComputationPrecision(
		MatunaComputationPrecision computationPrecision)
{
	this->computationPrecision = computationPrecision;
}

template<class T>
string ImageForwardPerceptronKernel<T>::ProgramName() const
{
	return programName;
}

template<class T>
vector<string> ImageForwardPerceptronKernel<T>::GetProgramCode() const
{
	vector<string> result;
	result.push_back(
			FileHelper::GetTextFromPath(
					Path::Combine(
							Path::GetDirectoryPath(
									FileHelper::GetExecutablePath()), "kernels",
							"ImageForwardPerceptronKernel.cl")));
	return result;
}

template<class T>
string ImageForwardPerceptronKernel<T>::KernelName() const
{
	return kernelName;
}

template<class T>
string ImageForwardPerceptronKernel<T>::GetCompilerOptions() const
{
	return compilerOptions;
}

template<class T>
const vector<size_t>& ImageForwardPerceptronKernel<T>::GlobalWorkSize() const
{
	return globalWorkSize;
}

template<class T>
const vector<size_t>& ImageForwardPerceptronKernel<T>::LocalWorkSize() const
{
	return localWorkSize;
}

template class ImageForwardPerceptronKernel<cl_float> ;
template class ImageForwardPerceptronKernel<cl_double> ;

} /* namespace MachineLearning */
} /* namespace Matuna */
