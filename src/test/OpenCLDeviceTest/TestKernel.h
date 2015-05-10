/*
 * TestKernel.h
 *
 *  Created on: Apr 29, 2015
 *      Author: Mikael
 */

#ifndef ATML_TEST_OPENCLDEVICETEST_TESTKERNEL_H_
#define ATML_TEST_OPENCLDEVICETEST_TESTKERNEL_H_

#include <CL/cl.h>
#include <OpenCLHelper/OpenCLKernelProgram.h>
#include <OpenCLHelper/OpenCLMemory.h>
#include <memory>
#include <vector>
#include <tuple>

using namespace ATML::Helper;

class TestKernel : public OpenCLKernelProgram
{
public:
	static const string programCode;

private:
	vector<tuple<cl_uint, shared_ptr<OpenCLMemory>>> memoryArguments;
	vector<size_t> globalWorkSize;
	vector<size_t> localWorkSize;
public:
	TestKernel();
	virtual ~TestKernel();

	void SetMemorySize(size_t size);
	void SetInput1(shared_ptr<OpenCLMemory> input1);
	void SetInput2(shared_ptr<OpenCLMemory> input2);
	void SetOutput(shared_ptr<OpenCLMemory> output);

	virtual string GetCompilerOptions() const override;
	virtual vector<string> GetProgramCode() const override;

	virtual string ProgramName() const override;
	virtual string KernelName() const override;
	virtual const vector<size_t>& GlobalWorkSize() const override;
	virtual const vector<size_t>& LocalWorkSize() const override;
};

#endif /* ATML_TEST_OPENCLDEVICETEST_TESTKERNEL_H_ */
