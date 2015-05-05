/*
 * PerceptronKernel.h
 *
 *  Created on: May 5, 2015
 *      Author: Mikael
 */

#ifndef ATML_CNNOPENCL_PERCEPTRONKERNEL_H_
#define ATML_CNNOPENCL_PERCEPTRONKERNEL_H_

#include "OpenCLHelper/OpenCLKernel.h"
#include "OpenCLHelper/OpenCLMemory.h"
#include <memory>
#include <tuple>
#include <string>
#include <vector>

using namespace ATML::Helper;
using namespace std;

namespace ATML
{
namespace MachineLearning
{

class ForwardPerceptronKernel: public OpenCLKernel
{
private:
	vector<tuple<int, shared_ptr<OpenCLMemory>>> memoryArguments;
	vector<tuple<int, size_t, void*>> otherArguments;
	vector<size_t> globalWorkSize;
	vector<size_t> localWorkSize;

	string kernelName;
	string programName;

public:
	ForwardPerceptronKernel();
	~ForwardPerceptronKernel();

	virtual string ProgramName() const override;
	virtual string ProgramCode() const override;
	virtual string KernelName() const override;
	virtual const vector<tuple<int, shared_ptr<OpenCLMemory>>>&GetMemoryArguments() const override;
	virtual const vector<tuple<int, size_t, void*>>& GetOtherArguments() const override;
	virtual const vector<size_t>& GlobalWorkSize() const override;
	virtual const vector<size_t>& LocalWorkSize() const override;
};

}
/* namespace MachineLearning */
} /* namespace ATML */

#endif /* ATML_CNNOPENCL_PERCEPTRONKERNEL_H_ */
