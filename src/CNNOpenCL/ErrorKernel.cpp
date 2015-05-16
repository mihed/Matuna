/*
 * ErrorKernel.cpp
 *
 *  Created on: May 15, 2015
 *      Author: Mikael
 */

#include "ErrorKernel.h"

namespace ATML
{
namespace MachineLearning
{

template class ErrorKernel<cl_float>;
template class ErrorKernel<cl_double>;

template<class T>
ErrorKernel<T>::ErrorKernel()
{

}

template<class T>
ErrorKernel<T>::~ErrorKernel()
{

}

template<class T>
string ErrorKernel<T>::ProgramName() const
{
	return programName;
}

template<class T>
string ErrorKernel<T>::GetCompilerOptions() const
{
	return compilerOptions;
}

template<class T>
vector<string> ErrorKernel<T>::GetProgramCode() const
{
	return vector<string>();
}

template<class T>
string ErrorKernel<T>::KernelName() const
{
	return kernelName;
}

template<class T>
const vector<size_t>& ErrorKernel<T>::GlobalWorkSize() const
{
	return globalWorkSize;
}

template<class T>
const vector<size_t>& ErrorKernel<T>::LocalWorkSize() const
{
	return localWorkSize;
}

} /* namespace MachineLearning */
} /* namespace ATML */
