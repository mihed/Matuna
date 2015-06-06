/*
 * ImageOutputKernel.cpp
 *
 *  Created on: May 27, 2015
 *      Author: Mikael
 */

#include "ImageOutputKernel.h"
#include "OCLHelper/OCLUtility.h"
#include "Helper/FileHelper.h"
#include "Helper/Path.h"

namespace Matuna
{
namespace MachineLearning
{

template<class T>
ImageOutputKernel<T>::ImageOutputKernel(int globalWidth, int globalHeight,
		int globalUnits, int inputOffsetWidth, int inputOffsetHeight,
		int outputOffsetWidth, int outputOffsetHeight, int inputUnitOffset,
		int outputUnitOffset, int inputStride, int outputStride,
		int outputMemoryHeight, int inputMemoryHeight) :
		globalWidth(globalWidth), globalHeight(globalHeight), globalUnits(
				globalUnits), inputOffsetWidth(inputOffsetWidth), inputOffsetHeight(
				inputOffsetHeight), outputOffsetWidth(outputOffsetWidth), outputOffsetHeight(
				outputOffsetHeight), inputUnitOffset(inputUnitOffset), outputUnitOffset(
				outputUnitOffset), inputStride(inputStride), outputStride(
				outputStride), outputMemoryHeight(outputMemoryHeight), inputMemoryHeight(
				inputMemoryHeight)
{
	stringstream stringStream;

	stringStream << "ImageOutputBackPropProgram";
	stringStream << instanceCounter;
	programName = stringStream.str();
	kernelName = "BackPropagation";

	useConstantInput = false;
	useConstantTarget = false;
	useRelaxedMath = false;
	activationFunction = MatunaSigmoidActivation;
	computationPrecision = MatunaNormalPrecision;
	errorFunction = MatunaMeanSquareError;

	globalWorkSize.push_back(globalWidth);
	globalWorkSize.push_back(globalHeight);
	globalWorkSize.push_back(globalUnits);
}

template<class T>
ImageOutputKernel<T>::~ImageOutputKernel()
{

}

template<class T>
void ImageOutputKernel<T>::SetConstantInput(bool value)
{
	useConstantInput = value;
}

template<class T>
void ImageOutputKernel<T>::SetConstantTarget(bool value)
{
	useConstantTarget = value;
}

template<class T>
void ImageOutputKernel<T>::SetUseRelaxedMath(bool value)
{
	useRelaxedMath = value;
}

template<class T>
void ImageOutputKernel<T>::SetActivationFunction(
		MatunaActivationFunction activationFunction)
{
	this->activationFunction = activationFunction;
}

template<class T>
void ImageOutputKernel<T>::SetComputationPrecision(
		MatunaComputationPrecision computationPrecision)
{
	this->computationPrecision = computationPrecision;
}

template<class T>
void ImageOutputKernel<T>::SetErrorFunction(MatunaErrorFunction errorFunction)
{
	this->errorFunction = errorFunction;
}

template<class T>
void ImageOutputKernel<T>::SetInput(OCLMemory* input)
{
	auto rawInput = input->GetCLMemory();
	CheckOCLError(
			clSetKernelArg(this->GetKernel(), 0, sizeof(cl_mem), &rawInput),
			"Could not set the kernel arguments");
}

template<class T>
void ImageOutputKernel<T>::SetTarget(OCLMemory* target)
{
	auto rawTarget = target->GetCLMemory();
	CheckOCLError(
			clSetKernelArg(this->GetKernel(), 1, sizeof(cl_mem), &rawTarget),
			"Could not set the kernel arguments");
}

template<class T>
void ImageOutputKernel<T>::SetOutput(OCLMemory* output)
{
	auto rawOutput = output->GetCLMemory();
	CheckOCLError(
			clSetKernelArg(this->GetKernel(), 2, sizeof(cl_mem), &rawOutput),
			"Could not set the kernel arguments");
}

template<class T>
void ImageOutputKernel<T>::InitializeCompilerOptions()
{
	stringstream stringStream;

	if (useConstantInput)
		stringStream << "-D" << "CONSTANT_INPUT ";
	if (useConstantTarget)
		stringStream << "-D" << "CONSTANT_TARGET ";

	//Refer to the notes for this
	if (errorFunction == MatunaMeanSquareError)
	{
		if (activationFunction == MatunaLinearActivation)
			stringStream << "-D" << "DIFFERENCE ";
		else
			stringStream << "-D" << "MSE_ANY ";
	}
	else if (errorFunction == MatunaCrossEntropy)
	{
		if (globalUnits == 1 && globalHeight == 1 && globalWidth == 1)
		{
			if (activationFunction == MatunaSigmoidActivation)
				stringStream << "-D" << "DIFFERENCE ";
			else
				stringStream << "-D" << "CE_BINARY_ANY ";
		}
		else
		{
			if (activationFunction == MatunaSoftMaxActivation)
				stringStream << "-D" << "DIFFERENCE ";
			else
				stringStream << "-D" << "CE_ANY ";
		}
	}
	else
		throw invalid_argument(
				"The error function is not supported by the output kernel");

	stringStream << "-D" << "INPUT_OFFSET_WIDTH=" << inputOffsetWidth << " ";
	stringStream << "-D" << "INPUT_OFFSET_HEIGHT=" << inputOffsetHeight << " ";
	stringStream << "-D" << "OUTPUT_OFFSET_WIDTH=" << outputOffsetWidth << " ";
	stringStream << "-D" << "OUTPUT_OFFSET_HEIGHT=" << outputOffsetHeight
			<< " ";
	stringStream << "-D" << "INPUT_UNIT_OFFSET=" << inputUnitOffset << " ";
	stringStream << "-D" << "INPUT_STRIDE=" << inputStride << " ";
	stringStream << "-D" << "OUTPUT_STRIDE=" << outputStride << " ";
	stringStream << "-D" << "OUTPUT_UNIT_ELEMENT_COUNT_INC_PADDING="
			<< (outputStride * outputMemoryHeight) << " ";
	stringStream << "-D" << "INPUT_UNIT_ELEMENT_COUNT_INC_PADDING="
			<< (inputStride * inputMemoryHeight) << " ";
	stringStream << "-D" << "OUTPUT_UNIT_OFFSET=" << outputUnitOffset << " ";

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
string ImageOutputKernel<T>::ProgramName() const
{
	return programName;
}

template<class T>
string ImageOutputKernel<T>::GetCompilerOptions() const
{
	return compilerOptions;
}

template<class T>
vector<string> ImageOutputKernel<T>::GetProgramCode() const
{
	vector<string> result;
	result.push_back(
			FileHelper::GetTextFromPath(
					Path::Combine(
							Path::GetDirectoryPath(
									FileHelper::GetExecutablePath()), "kernels",
							"ImageOutputBackProp.cl")));
	return result;
}

template<class T>
string ImageOutputKernel<T>::KernelName() const
{
	return kernelName;
}

template<class T>
const vector<size_t>& ImageOutputKernel<T>::GlobalWorkSize() const
{
	return globalWorkSize;
}

template<class T>
const vector<size_t>& ImageOutputKernel<T>::LocalWorkSize() const
{
	return localWorkSize;
}

template class ImageOutputKernel<cl_double> ;
template class ImageOutputKernel<cl_float> ;

} /* namespace MachineLearning */
} /* namespace Matuna */
