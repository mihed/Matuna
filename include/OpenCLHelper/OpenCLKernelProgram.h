/*
 * OpenCLKernelProgram.h
 *
 *  Created on: May 7, 2015
 *      Author: Mikael
 */

#ifndef ATML_OPENCLHELPER_OPENCLKERNELPROGRAM_H_
#define ATML_OPENCLHELPER_OPENCLKERNELPROGRAM_H_

#include "OpenCLKernel.h"
#include <string>
#include <vector>

namespace ATML
{
namespace Helper
{

class OpenCLKernelProgram: public OpenCLKernel
{
public:
	OpenCLKernelProgram();
	virtual ~OpenCLKernelProgram();
	virtual string GetCompilerOptions() const = 0;
	virtual vector<string> GetProgramCode() const = 0;
};

} /* namespace Helper */
} /* namespace ATML */

#endif /* ATML_OPENCLHELPER_OPENCLKERNELPROGRAM_H_ */
