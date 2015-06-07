/*
 * TestKernel.h
 *
 *  Created on: Apr 29, 2015
 *      Author: Mikael
 */

#ifndef MATUNA_TEST_OCLDEVICETEST_TESTKERNEL_H_
#define MATUNA_TEST_OCLDEVICETEST_TESTKERNEL_H_

#include <Matuna.OCLHelper/OCLKernelProgram.h>
#include <Matuna.OCLHelper/OCLMemory.h>
#include <memory>
#include <vector>
#include <tuple>

using namespace Matuna::Helper;

class TestKernel : public OCLKernelProgram
{
public:
	static const string programCode;

private:
	vector<size_t> globalWorkSize;
	vector<size_t> localWorkSize;
public:
	TestKernel();
	virtual ~TestKernel();

	void SetMemorySize(size_t size);
	void SetInput1(shared_ptr<OCLMemory> input1);
	void SetInput2(shared_ptr<OCLMemory> input2);
	void SetOutput(shared_ptr<OCLMemory> output);

	virtual string GetCompilerOptions() const override;
	virtual vector<string> GetProgramCode() const override;

	virtual string ProgramName() const override;
	virtual string KernelName() const override;
	virtual const vector<size_t>& GlobalWorkSize() const override;
	virtual const vector<size_t>& LocalWorkSize() const override;
};

#endif /* MATUNA_TEST_OCLDEVICETEST_TESTKERNEL_H_ */
