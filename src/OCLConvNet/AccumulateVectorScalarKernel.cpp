/*
 * AccumulateVectorScalarKernel.cpp
 *
 *  Created on: Jun 4, 2015
 *      Author: Mikael
 */

#include "AccumulateVectorScalarKernel.h"
#include "OCLHelper/OCLUtility.h"

namespace Matuna
{
namespace MachineLearning
{

template<class T>
AccumulateVectorScalarKernel<T>::AccumulateVectorScalarKernel(
		string programName) :
		programName(programName)
{
	kernelName = "AccumulateVectorWithScalarKernel";
}

template<class T>
AccumulateVectorScalarKernel<T>::~AccumulateVectorScalarKernel()
{

}

template<class T>
void AccumulateVectorScalarKernel<T>::SetGlobalWorkSize(int globalUnits)
{
	globalWorkSize.clear();
	globalWorkSize.push_back(globalUnits);
}

template<class T>
void AccumulateVectorScalarKernel<T>::SetInput(OCLMemory* input)
{
	auto rawInput = input->GetCLMemory();
	CheckOCLError(
			clSetKernelArg(this->GetKernel(), 0, sizeof(cl_mem), &rawInput),
			"Could not set the kernel arguments");
}

template<class T>
void AccumulateVectorScalarKernel<T>::SetAccumulator(OCLMemory* accumulator)
{
	auto rawInput = accumulator->GetCLMemory();
	CheckOCLError(
			clSetKernelArg(this->GetKernel(), 1, sizeof(cl_mem), &rawInput),
			"Could not set the kernel arguments");
}

template<class T>
void AccumulateVectorScalarKernel<T>::SetScalar(T scalar)
{
	CheckOCLError(clSetKernelArg(this->GetKernel(), 2, sizeof(T), &scalar),
			"Could not set the kernel arguments");
}

template<class T>
string AccumulateVectorScalarKernel<T>::ProgramName() const
{
	return programName;
}

template<class T>
string AccumulateVectorScalarKernel<T>::KernelName() const
{
	return kernelName;
}

template<class T>
const vector<size_t>& AccumulateVectorScalarKernel<T>::GlobalWorkSize() const
{
	return globalWorkSize;
}

template<class T>
const vector<size_t>& AccumulateVectorScalarKernel<T>::LocalWorkSize() const
{
	return localWorkSize;
}

template class AccumulateVectorScalarKernel<cl_float> ;
template class AccumulateVectorScalarKernel<cl_double> ;

} /* namespace MachineLearning */
} /* namespace Matuna */
