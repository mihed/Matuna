/*
* OCLKernelInfo.cpp
*
*  Created on: May 22, 2015
*      Author: Mikael
*/

#include "OCLKernelInfo.h"

namespace Matuna
{
	namespace Helper
	{

		OCLKernelInfo::OCLKernelInfo(size_t kernelWorkGroupSize,
			vector<size_t> compileWorkGroupSize,
			cl_ulong localMemorySize,
			size_t preferredWorkGroupMultiple)
		{
			this->kernelWorkGroupSize = kernelWorkGroupSize;
			this->compileWorkGroupSize = compileWorkGroupSize;
			this->localMemorySize = localMemorySize;
			this->preferredWorkGroupMultiple = preferredWorkGroupMultiple;
		}

		OCLKernelInfo::~OCLKernelInfo()
		{

		}

		size_t OCLKernelInfo::PreferredWorkGroupMultiple() const
		{
			return preferredWorkGroupMultiple;
		}

		size_t OCLKernelInfo::KernelWorkGroupSize() const
		{
			return kernelWorkGroupSize;
		}

		vector<size_t> OCLKernelInfo::CompileWorkGroupSize() const
		{
			return compileWorkGroupSize;
		}

		cl_ulong OCLKernelInfo::LocalMemorySize() const
		{
			return localMemorySize;
		}

	} /* namespace Helper */
} /* namespace Matuna */
