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
ConvolutionKernel<T>::ConvolutionKernel()
{
	stringstream stringStream;

	stringStream << "ConvolutionKernelProgram";
	stringStream << instanceCounter;
	programName = stringStream.str();

	kernelName = "ConvolutionKernel";

}

template<class T>
ConvolutionKernel<T>::~ConvolutionKernel()
{

}

template<class T>
void ConvolutionKernel<T>::InitializeCompilerOptions() {

}

template<class T>
string ConvolutionKernel<T>::ProgramName() const {
	return programName;
}

template<class T>
string ConvolutionKernel<T>::GetCompilerOptions() const {
	return compilerOptions;
}

template<class T>
vector<string> ConvolutionKernel<T>::GetProgramCode() const {
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
string ConvolutionKernel<T>::KernelName() const {
	return kernelName;
}

template<class T>
const vector<size_t>& ConvolutionKernel<T>::GlobalWorkSize() const {
	return globalWorkSize;
}

template<class T>
const vector<size_t>& ConvolutionKernel<T>::LocalWorkSize() const {
	return localWorkSize;
}

template class ConvolutionKernel<cl_float> ;
template class ConvolutionKernel<cl_double> ;

} /* namespace MachineLearning */
} /* namespace ATML */
