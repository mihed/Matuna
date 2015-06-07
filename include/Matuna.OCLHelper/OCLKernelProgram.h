/*
 * OCLKernelProgram.h
 *
 *  Created on: May 7, 2015
 *      Author: Mikael
 */

#ifndef MATUNA_OCLHELPER_OCLKERNELPROGRAM_H_
#define MATUNA_OCLHELPER_OCLKERNELPROGRAM_H_

#include "OCLKernel.h"
#include <string>
#include <vector>

namespace Matuna
{
namespace Helper
{

class OCLKernelProgram: public OCLKernel
{
public:
	OCLKernelProgram();
	virtual ~OCLKernelProgram();
	virtual string GetCompilerOptions() const = 0;
	virtual vector<string> GetProgramCode() const = 0;
};

} /* namespace Helper */
} /* namespace Matuna */

#endif /* MATUNA_OCLHELPER_OCLKERNELPROGRAM_H_ */
