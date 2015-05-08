/*
 * PerceptronKernel.h
 *
 *  Created on: May 5, 2015
 *      Author: Mikael
 */

#ifndef ATML_CNNOPENCL_PERCEPTRONKERNEL_H_
#define ATML_CNNOPENCL_PERCEPTRONKERNEL_H_

#include "OpenCLHelper/OpenCLKernelProgram.h"
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

class ForwardPerceptronKernel: public OpenCLKernelProgram
{
private:
	vector<size_t> globalWorkSize;
	vector<size_t> localWorkSize;

	string kernelName;
	string programName;

public:
	ForwardPerceptronKernel();
	~ForwardPerceptronKernel();

	virtual string ProgramName() const override;
	virtual string GetCompilerOptions() const override;
	virtual vector<string> GetProgramCode() const override;
	virtual string KernelName() const override;
	virtual void SetArguments() override;
	virtual const vector<size_t>& GlobalWorkSize() const override;
	virtual const vector<size_t>& LocalWorkSize() const override;
};

}
/* namespace MachineLearning */
} /* namespace ATML */

#endif /* ATML_CNNOPENCL_PERCEPTRONKERNEL_H_ */
