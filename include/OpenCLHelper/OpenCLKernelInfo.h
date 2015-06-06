/*
 * OpenCLKernelInfo.h
 *
 *  Created on: May 22, 2015
 *      Author: Mikael
 */

#ifndef MATUNA_OPENCLHELPER_OPENCLKERNELINFO_H_
#define MATUNA_OPENCLHELPER_OPENCLKERNELINFO_H_

#include "OpenCLInclude.h"
#include <vector>

using namespace std;

namespace Matuna
{
namespace Helper
{

class OpenCLKernelInfo
{
private:
	size_t kernelWorkGroupSize;
	vector<size_t> compileWorkGroupSize;
	cl_ulong localMemorySize;
public:
	OpenCLKernelInfo(size_t kernelWorkGroupSize,
			vector<size_t> compileWorkGroupSize, cl_ulong localMemorySize);
	~OpenCLKernelInfo();

	size_t KernelWorkGroupSize() const;
	vector<size_t> CompileWorkGroupSize() const;
	cl_ulong LocalMemorySize() const;
};

} /* namespace Helper */
} /* namespace Matuna */

#endif /* MATUNA_OPENCLHELPER_OPENCLKERNELINFO_H_ */
