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

namespace ATML {
namespace Helper {

class TestKernel: public OpenCLKernel {
private:
	vector<tuple<int, shared_ptr<OpenCLMemory>>> memoryArguments;
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
	virtual vector<tuple<int, shared_ptr<OpenCLMemory>>> GetMemoryArguments() const override;
	virtual vector<tuple<int, size_t, void*>> GetOtherArguments() const override;
	virtual vector<size_t> GlobalWorkSize() const override;
	virtual vector<size_t> LocalWorkSize() const override;
};

} /* namespace Helper */
} /* namespace ATML */

#endif /* ATML_TEST_OPENCLDEVICETEST_TESTKERNEL_H_ */
