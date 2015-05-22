/*
 * ConvolutionKernel.cpp
 *
 *  Created on: May 20, 2015
 *      Author: Mikael
 */

#include "ConvolutionKernel.h"
#include "OpenCLHelper/OpenCLUtility.h"
#include "Helper/FileHelper.h"
#include "Helper/Path.h"

namespace ATML
{
namespace MachineLearning
{

template<class T>
ConvolutionKernel<T>::ConvolutionKernel(int dataOutputUnits,
		int dataOutputWidth, int dataOutputHeight, int filterWidth,
		int filterHeight, int inputOffsetWidth, int inputOffsetHeight,
		int outputOffsetWidth, int outputOffsetHeight, int outputOffsetUnit,
		int outputStride, int inputStride, int outputUnitMemoryCount,
		int filterUnitElementCount, bool useLocalMemory) :
		dataOutputUnits(dataOutputUnits), dataOutputWidth(dataOutputWidth), dataOutputHeight(
				dataOutputHeight), filterWidth(filterWidth), filterHeight(
				filterHeight), inputOffsetWidth(inputOffsetWidth), inputOffsetHeight(
				inputOffsetHeight), outputOffsetWidth(outputOffsetWidth), outputOffsetHeight(
				outputOffsetHeight), outputOffsetUnit(outputOffsetUnit), outputStride(
				outputStride), inputStride(inputStride), outputUnitMemoryCount(
				outputUnitMemoryCount), filterUnitElementCount(
				filterUnitElementCount), useLocalMemory(useLocalMemory)
{
	stringstream stringStream;

	stringStream << "ConvolutionKernelProgram";
	stringStream << instanceCounter;
	programName = stringStream.str();

	kernelName = "ConvolutionKernel";

	useConstantInput = false;
	useConstantFilters = false;
	useConstantBias = false;
	useRelaxedMath = false;

	activation = ATMLSigmoidActivation;
	precision = ATMLNormalPrecision;

	globalWorkSize.push_back(dataOutputWidth);
	globalWorkSize.push_back(dataOutputHeight);
	globalWorkSize.push_back(dataOutputUnits);

}

template<class T>
ConvolutionKernel<T>::~ConvolutionKernel()
{

}

template<class T>
void ConvolutionKernel<T>::InitializeCompilerOptions()
{
	stringstream stringStream;

	if (useConstantBias)
		stringStream << "-D" << "CONSTANT_BIAS ";
	if (useConstantFilters)
		stringStream << "-D" << "CONSTANT_FILTERS ";
	if (useConstantInput)
		stringStream << "-D" << "CONSTANT_INPUT ";

	if (useLocalMemory)
		stringStream << "-D" << "USE_LOCAL_MEMORY ";

	stringStream << "-D" << "FILTER_WIDTH=" << filterWidth << " ";
	stringStream << "-D" << "FILTER_HEIGHT=" << filterHeight << " ";
	stringStream << "-D" << "INPUT_OFFSET_WIDTH=" << inputOffsetWidth << " ";
	stringStream << "-D" << "INPUT_OFFSET_HEIGHT=" << inputOffsetHeight << " ";
	stringStream << "-D" << "OUTPUT_OFFSET_WIDTH=" << outputOffsetWidth << " ";
	stringStream << "-D" << "OUTPUT_OFFSET_HEIGHT=" << outputOffsetHeight << " ";
	stringStream << "-D" << "OUTPUT_OFFSET_UNIT=" << outputOffsetUnit << " ";
	stringStream << "-D" << "OUTPUT_WIDTH=" << outputStride << " ";
	stringStream << "-D" << "INPUT_WIDTH=" << inputStride << " ";
	stringStream << "-D" << "OUTPUT_UNIT_ELEMENT_COUNT_INC_PADDING=" << (outputStride * (dataOutputHeight + outputOffsetHeight)) << " ";
	stringStream << "-D" << "FILTER_UNIT_ELEMENT_COUNT_INC_PADDING=" << (filterWidth *  filterHeight) << " ";

	if (is_same<cl_double, T>::value)
		stringStream << "-D" << "DOUBLE_PRECISION ";
	else if (!is_same<cl_float, T>::value)
		throw runtime_error(
		"The template type is not valid. This is an indication of programming error");

	if (activation == ATMLSigmoidActivation)
		stringStream << "-D" << "SIGMOID ";
	else if (activation == ATMLTanhActivation)
		stringStream << "-D" << "TANH ";

	if (precision == ATMLNativePrecision)
		stringStream << "-D" << "NATIVE_MATH ";
	else if (precision == ATMLHalfPrecision)
		stringStream << "-D" << "HALF_MATH ";

	if (useRelaxedMath)
		stringStream << "-cl-fast-relaxed-math";

	compilerOptions = stringStream.str();
}

template<class T>
void ConvolutionKernel<T>::SetInput(OpenCLMemory* input)
{
	auto rawInput = input->GetCLMemory();
	CheckOpenCLError(
			clSetKernelArg(this->GetKernel(), 0, sizeof(cl_mem), &rawInput),
			"Could not set the kernel arguments");
}

template<class T>
void ConvolutionKernel<T>::SetOutput(OpenCLMemory* output)
{
	auto rawOutput = output->GetCLMemory();
	CheckOpenCLError(
			clSetKernelArg(this->GetKernel(), 1, sizeof(cl_mem), &rawOutput),
			"Could not set the kernel arguments");
}

template<class T>
void ConvolutionKernel<T>::SetBiases(OpenCLMemory* biases)
{
	auto rawBias = biases->GetCLMemory();
	CheckOpenCLError(
			clSetKernelArg(this->GetKernel(), 3, sizeof(cl_mem), &rawBias),
			"Could not set the kernel arguments");
}

template<class T>
void ConvolutionKernel<T>::SetFilters(OpenCLMemory* filters)
{
	auto rawFilter = filters->GetCLMemory();
	CheckOpenCLError(
			clSetKernelArg(this->GetKernel(), 2, sizeof(cl_mem), &rawFilter),
			"Could not set the kernel arguments");
}

template<class T>
void ConvolutionKernel<T>::SetLocalWorkGroup(int width, int height)
{
	if (!useLocalMemory)
		throw invalid_argument(
				"Local memory is not used so you may not set any local memory.");

	CheckOpenCLError(
			clSetKernelArg(this->GetKernel(), 4,
					sizeof(T) * (width + filterWidth - 1)* (height + filterHeight - 1), nullptr),
			"Could not set the kernel arguments");
	localWorkSize.clear();
	localWorkSize.push_back(width);
	localWorkSize.push_back(height);
	localWorkSize.push_back(1);
}

template<class T>
void ConvolutionKernel<T>::SetConstantInput(bool value)
{
	useConstantInput = value;
}

template<class T>
void ConvolutionKernel<T>::SetConstantFilters(bool value)
{
	useConstantFilters = value;
}

template<class T>
void ConvolutionKernel<T>::SetConstantBias(bool value)
{
	useConstantBias = value;
}

template<class T>
void ConvolutionKernel<T>::SetRelaxedMath(bool value)
{
	useRelaxedMath = value;
}

template<class T>
void ConvolutionKernel<T>::SetActivationFunction(
		ATMLActivationFunction activation)
{
	this->activation = activation;
}

template<class T>
void ConvolutionKernel<T>::SetComputationPrecision(
		ATMLComputationPrecision precision)
{
	this->precision = precision;
}

template<class T>
string ConvolutionKernel<T>::ProgramName() const
{
	return programName;
}

template<class T>
string ConvolutionKernel<T>::GetCompilerOptions() const
{
	return compilerOptions;
}

template<class T>
vector<string> ConvolutionKernel<T>::GetProgramCode() const
{
	vector<string> result;
	result.push_back(
			FileHelper::GetTextFromPath(
					Path::Combine(
							Path::GetDirectoryPath(
									FileHelper::GetExecutablePath()), "kernels",
							"ConvolutionKernel.cl")));
	return result;
}

template<class T>
string ConvolutionKernel<T>::KernelName() const
{
	return kernelName;
}

template<class T>
const vector<size_t>& ConvolutionKernel<T>::GlobalWorkSize() const
{
	return globalWorkSize;
}

template<class T>
const vector<size_t>& ConvolutionKernel<T>::LocalWorkSize() const
{
	return localWorkSize;
}

template class ConvolutionKernel<cl_float> ;
template class ConvolutionKernel<cl_double> ;

} /* namespace MachineLearning */
} /* namespace ATML */
