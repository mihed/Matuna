/*
 * PerceptronKernel.cpp
 *
 *  Created on: May 5, 2015
 *      Author: Mikael
 */

#include "ForwardPerceptronKernel.h"
#include "Helper/Path.h"
#include <sstream>
#include <type_traits>

namespace ATML
{
namespace MachineLearning
{

template class ForwardPerceptronKernel<cl_float> ;
template class ForwardPerceptronKernel<cl_double> ;

template<class T>
ForwardPerceptronKernel<T>::ForwardPerceptronKernel(int inputUnitsCount,
		int unitsCount) :
		inputUnitsCount(inputUnitsCount), unitsCount(unitsCount)
{
	stringstream stringStream;

	stringStream << "ForwardPerceptronProgram";
	stringStream << instanceCounter;
	programName = stringStream.str();
	kernelName = "ForwardPerceptronKernel";

	useConstantWeights = false;
	useConstantInput = false;
	useConstantBiases = false;
	useRelaxedMath = false;
	activationFunction = ATMLSigmoidActivation;
	computationPrecision = ATMLNormalPrecision;
	biases = nullptr;
	weights = nullptr;

	globalWorkSize.push_back(unitsCount);
}

template<class T>
ForwardPerceptronKernel<T>::~ForwardPerceptronKernel()
{

}

template<class T>
void ForwardPerceptronKernel<T>::InitializeArgumentsAndCompilerOptions()
{
	//Arguments that do not change between executions
	auto rawWeights = weights->GetCLMemory();
	auto rawBiases = biases->GetCLMemory();
	clSetKernelArg(this->kernel, 2, sizeof(cl_mem), &rawWeights);
	clSetKernelArg(this->kernel, 3, sizeof(cl_mem), &rawBiases);

	//Constructing compiler options string
	stringstream stringStream;

	if (useConstantBiases)
		stringStream << "-D" << "CONSTANT_BIASES ";
	if (useConstantWeights)
		stringStream << "-D" << "CONSTANT_WEIGHTS ";
	if (useConstantInput)
		stringStream << "-D" << "CONSTANT_INPUT ";

	stringStream << "-D" << "INPUT_COUNT=" << inputUnitsCount << " ";

	if (is_same<cl_double, T>::value)
		stringStream << "-D" << "DOUBLE_PRECISION ";
	else if (!is_same<cl_float, T>::value)
		throw runtime_error(
				"The template type is not valid. This is an indication of programming error");

	if (activationFunction == ATMLSigmoidActivation)
		stringStream << "-D" << "SIGMOID ";
	else if (activationFunction == ATMLTanhActivation)
		stringStream << "-D" << "TANH ";

	if (computationPrecision == ATMLNativePrecision)
		stringStream << "-D" << "NATIVE_MATH ";
	else if (computationPrecision == ATMLHalfPrecision)
		stringStream << "-D" << "HALF_MATH ";

	if (useRelaxedMath)
		stringStream << "-cl-fast-relaxed-math";

	compilerOptions = stringStream.str();
}

template<class T>
void ForwardPerceptronKernel<T>::SetUseRelaxedMath(bool value)
{
	useRelaxedMath = value;
}

template<class T>
void ForwardPerceptronKernel<T>::SetInput(OpenCLMemory* input)
{
	auto rawInput = input->GetCLMemory();
	clSetKernelArg(this->kernel, 0, sizeof(cl_mem), &rawInput);
}

template<class T>
void ForwardPerceptronKernel<T>::SetOutput(OpenCLMemory* output)
{
	auto rawOutput = output->GetCLMemory();
	clSetKernelArg(this->kernel, 1, sizeof(cl_mem), &rawOutput);
}

template<class T>
void ForwardPerceptronKernel<T>::SetWeights(OpenCLMemory* weights)
{
	this->weights = weights;
}

template<class T>
void ForwardPerceptronKernel<T>::SetBiases(OpenCLMemory* biases)
{
	this->biases = biases;
}

template<class T>
void ForwardPerceptronKernel<T>::SetUseConstantWeights(bool value)
{
	useConstantWeights = value;
}

template<class T>
void ForwardPerceptronKernel<T>::SetUseConstantInput(bool value)
{
	useConstantInput = value;
}

template<class T>
void ForwardPerceptronKernel<T>::SetUseConstantBiases(bool value)
{
	useConstantBiases = value;
}

template<class T>
void ForwardPerceptronKernel<T>::SetActivationFunction(
		ATMLActivationFunction activationFunction)
{
	this->activationFunction = activationFunction;
}

template<class T>
void ForwardPerceptronKernel<T>::SetComputationPrecision(
		ATMLComputationPrecision computationPrecision)
{
	this->computationPrecision = computationPrecision;
}

template<class T>
string ForwardPerceptronKernel<T>::ProgramName() const
{
	return programName;
}

template<class T>
vector<string> ForwardPerceptronKernel<T>::GetProgramCode() const
{
	vector<string> result;
	result.push_back(
			GetTextFromPath(
					Path::Combine("kernels", "ForwardPerceptronKernel.cl")));
	return result;
}

template<class T>
string ForwardPerceptronKernel<T>::KernelName() const
{
	return kernelName;
}

template<class T>
string ForwardPerceptronKernel<T>::GetCompilerOptions() const
{
	return compilerOptions;
}

template<class T>
const vector<size_t>& ForwardPerceptronKernel<T>::GlobalWorkSize() const
{
	return globalWorkSize;
}

template<class T>
const vector<size_t>& ForwardPerceptronKernel<T>::LocalWorkSize() const
{
	return localWorkSize;
}

} /* namespace MachineLearning */
} /* namespace ATML */
