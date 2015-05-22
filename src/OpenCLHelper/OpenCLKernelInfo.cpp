/*
 * OpenCLKernelInfo.cpp
 *
 *  Created on: May 22, 2015
 *      Author: Mikael
 */

#include "OpenCLKernelInfo.h"

namespace ATML
{
namespace Helper
{

OpenCLKernelInfo::OpenCLKernelInfo(size_t kernelWorkGroupSize,
		vector<size_t> compileWorkGroupSize, cl_ulong localMemorySize) :
		kernelWorkGroupSize(kernelWorkGroupSize), compileWorkGroupSize(
				compileWorkGroupSize), localMemorySize(localMemorySize)
{

}

OpenCLKernelInfo::~OpenCLKernelInfo()
{

}

size_t OpenCLKernelInfo::KernelWorkGroupSize() const
{
	return kernelWorkGroupSize;
}

vector<size_t> OpenCLKernelInfo::CompileWorkGroupSize() const
{
	return compileWorkGroupSize;
}

cl_ulong OpenCLKernelInfo::LocalMemorySize() const
{
	return localMemorySize;
}

} /* namespace Helper */
} /* namespace ATML */
