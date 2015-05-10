/*
 * TestKernel.cpp
 *
 *  Created on: Apr 29, 2015
 *      Author: Mikael
 */

#include "TestKernel.h"
#include "OpenCLHelper/OpenCLUtility.h"

const string TestKernel::programCode = R"(__kernel void Test(const __global float* input1, const __global float* input2, __global float* output)
	{
		const int index = get_global_id(0);
		output[index] = input1[index] * input2[index];
	})";

TestKernel::TestKernel()
{

}

TestKernel::~TestKernel()
{

}

string TestKernel::ProgramName() const
{
	return "TestKernel";
}

void TestKernel::SetInput1(shared_ptr<OpenCLMemory> input1)
{
	auto memory = input1->GetCLMemory();
	CheckOpenCLError(clSetKernelArg(GetKernel(), 0, sizeof(cl_mem), &memory), "Could not set the kernel argument");
}

void TestKernel::SetInput2(shared_ptr<OpenCLMemory> input2)
{
	auto memory = input2->GetCLMemory();
	CheckOpenCLError(clSetKernelArg(GetKernel(), 1, sizeof(cl_mem), &memory), "Could not set the kernel argument");
}

void TestKernel::SetOutput(shared_ptr<OpenCLMemory> output)
{
	auto memory = output->GetCLMemory();
	CheckOpenCLError(clSetKernelArg(GetKernel(), 2, sizeof(cl_mem), &memory), "Could not set the kernel argument");
}

void TestKernel::SetMemorySize(size_t size)
{
	globalWorkSize.clear();
	globalWorkSize.push_back(size);
}

vector<string> TestKernel::GetProgramCode() const
{
	vector<string> result;
	result.push_back(programCode);
	return result;
}

string TestKernel::GetCompilerOptions() const
{
	return string();
}

string TestKernel::KernelName() const
{
	return "Test";
}

const vector<size_t>& TestKernel::GlobalWorkSize() const
{
	return globalWorkSize;
}

const vector<size_t>& TestKernel::LocalWorkSize() const
{
	return localWorkSize;
}

