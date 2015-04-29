/*
 * TestKernel.cpp
 *
 *  Created on: Apr 29, 2015
 *      Author: Mikael
 */

#include "TestKernel.h"

namespace ATML {
namespace Helper {

TestKernel::TestKernel() {

}

TestKernel::~TestKernel() {

}

string TestKernel::ProgramName() const
{
	return "TestKernel";
}

void TestKernel::SetInput1(shared_ptr<OpenCLMemory> input1)
{
	memoryArguments.push_back(make_tuple(0, input1));
}

void TestKernel::SetInput2(shared_ptr<OpenCLMemory> input2)
{
	memoryArguments.push_back(make_tuple(1, input2));
}

void TestKernel::SetOutput(shared_ptr<OpenCLMemory> output)
{
	memoryArguments.push_back(make_tuple(2, output));
}

void TestKernel::SetMemorySize(size_t size)
{
	this->size = size;
}

string TestKernel::ProgramCode() const
{
	auto program = R"(__kernel void Test(const __global float* input1, const __global float* input2, __global float* output)
	{
		const int index = get_global_id(0);
		output[index] = input1[index] * input2[index];
	})";

	string stringProgram(program);

	return stringProgram;
}

string TestKernel::KernelName() const
{
	return "Test";
}

vector<tuple<int, shared_ptr<OpenCLMemory>>> TestKernel::GetMemoryArguments() const
{
	return memoryArguments;
}

vector<tuple<int, size_t, void*>> TestKernel::GetOtherArguments() const
{
	return vector<tuple<int, size_t, void*>>();
}

vector<size_t> TestKernel::GlobalWorkSize() const
{
	vector<size_t> result;
	result.push_back(size);
	return result;
}

vector<size_t> TestKernel::LocalWorkSize() const
{
	return vector<size_t>();
}

} /* namespace Helper */
} /* namespace ATML */
