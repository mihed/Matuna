/*
 * OCLKernelInfo.h
 *
 *  Created on: May 22, 2015
 *      Author: Mikael
 */

#ifndef MATUNA_MATUNA_OCLHELPER_OCLKERNELINFO_H_
#define MATUNA_MATUNA_OCLHELPER_OCLKERNELINFO_H_

#include "OCLInclude.h"
#include <vector>

using namespace std;

namespace Matuna
{
namespace Helper
{

class OCLKernelInfo
{
private:
	size_t kernelWorkGroupSize;
	vector<size_t> compileWorkGroupSize;
	cl_ulong localMemorySize;
public:
	OCLKernelInfo(size_t kernelWorkGroupSize,
			vector<size_t> compileWorkGroupSize, cl_ulong localMemorySize);
	~OCLKernelInfo();

	size_t KernelWorkGroupSize() const;
	vector<size_t> CompileWorkGroupSize() const;
	cl_ulong LocalMemorySize() const;
};

} /* namespace Helper */
} /* namespace Matuna */

#endif /* MATUNA_MATUNA_OCLHELPER_OCLKERNELINFO_H_ */
