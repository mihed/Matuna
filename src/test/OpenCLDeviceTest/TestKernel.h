/*
 * TestKernel.h
 *
 *  Created on: Apr 29, 2015
 *      Author: Mikael
 */

#ifndef ATML_TEST_OPENCLDEVICETEST_TESTKERNEL_H_
#define ATML_TEST_OPENCLDEVICETEST_TESTKERNEL_H_

#include <OpenCLHelper/OpenCLKernel.h>
#include <memory>
#include <vector>
#include <tuple>

using namespace ATML::Helper;

class TestKernel: public OpenCLKernel
{
private:
	vector<tuple<int, shared_ptr<OpenCLMemory>>> memoryArguments;
	vector<tuple<int, size_t, void*>> otherArguments;
	vector<size_t> globalWorkSize;
	vector<size_t> localWorkSize;
	size_t size;
public:
	TestKernel();
	virtual ~TestKernel();

	void SetMemorySize(size_t size);
	void SetInput1(shared_ptr<OpenCLMemory> input1);
	void SetInput2(shared_ptr<OpenCLMemory> input2);
	void SetOutput(shared_ptr<OpenCLMemory> output);

	virtual string ProgramName() const override;
	virtual string ProgramCode() const override;
	virtual string KernelName() const override;
	virtual const vector<tuple<int, shared_ptr<OpenCLMemory>>>& GetMemoryArguments() const override;
	virtual const vector<tuple<int, size_t, void*>>& GetOtherArguments() const override;
	virtual const vector<size_t>& GlobalWorkSize() const override;
	virtual const vector<size_t>& LocalWorkSize() const override;
};

#endif /* ATML_TEST_OPENCLDEVICETEST_TESTKERNEL_H_ */
