/*
 * OpenCLKernel.h
 *
 *  Created on: Apr 28, 2015
 *      Author: Mikael
 */

#ifndef ATML_OPENCLHELPER_OPENCLKERNEL_H_
#define ATML_OPENCLHELPER_OPENCLKERNEL_H_

#include <string>
#include <vector>
#include <memory>
#include <tuple>

#include "OpenCLMemory.h"

namespace ATML {
namespace Helper {

class OpenCLKernel {
public:
	OpenCLKernel();
	virtual ~OpenCLKernel();

	virtual string ProgramName() const = 0;
	virtual string ProgramCode() const = 0;
	virtual string KernelName() const = 0;
	virtual vector<tuple<int, shared_ptr<OpenCLMemory>>> GetMemoryArguments() const = 0;
	virtual vector<tuple<int, size_t, void*>> GetOtherArguments() const = 0;
	virtual vector<size_t> GlobalWorkSize() const = 0;
	virtual vector<size_t> LocalWorkSize() const = 0;
};

} /* namespace Helper */
} /* namespace ATML */

#endif /* ATML_OPENCLHELPER_OPENCLKERNEL_H_ */
