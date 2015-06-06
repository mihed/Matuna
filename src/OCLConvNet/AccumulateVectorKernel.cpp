/*
 * AccumulateVectorKernel.cpp
 *
 *  Created on: Jun 4, 2015
 *      Author: Mikael
 */

#include "AccumulateVectorKernel.h"
#include "OCLHelper/OCLUtility.h"

namespace Matuna
{
namespace MachineLearning
{

template<class T>
AccumulateVectorKernel<T>::AccumulateVectorKernel(
		string programName) :
		programName(programName)
{
	kernelName = "AccumulateVectorKernel";
}

template<class T>
void AccumulateVectorKernel<T>::SetGlobalWorkSize(int globalUnits)
{
	globalWorkSize.clear();
	globalWorkSize.push_back(globalUnits);
}

template<class T>
AccumulateVectorKernel<T>::~AccumulateVectorKernel()
{

}

template<class T>
void AccumulateVectorKernel<T>::SetInput(OCLMemory* input)
{
	auto rawInput = input->GetCLMemory();
	CheckOCLError(
			clSetKernelArg(this->GetKernel(), 0, sizeof(cl_mem), &rawInput),
			"Could not set the kernel arguments");
}

template<class T>
void AccumulateVectorKernel<T>::SetAccumulator(OCLMemory* accumulator)
{
	auto rawInput = accumulator->GetCLMemory();
	CheckOCLError(
			clSetKernelArg(this->GetKernel(), 1, sizeof(cl_mem), &rawInput),
			"Could not set the kernel arguments");
}

template<class T>
string AccumulateVectorKernel<T>::ProgramName() const
{
	return programName;
}

template<class T>
string AccumulateVectorKernel<T>::KernelName() const
{
	return kernelName;
}

template<class T>
const vector<size_t>& AccumulateVectorKernel<T>::GlobalWorkSize() const
{
	return globalWorkSize;
}

template<class T>
const vector<size_t>& AccumulateVectorKernel<T>::LocalWorkSize() const
{
	return localWorkSize;
}

template class AccumulateVectorKernel<cl_float> ;
template class AccumulateVectorKernel<cl_double> ;

} /* namespace MachineLearning */
} /* namespace Matuna */
