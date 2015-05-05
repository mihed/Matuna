/*
 * OpenCLDevice.cpp
 *
 *  Created on: Apr 26, 2015
 *      Author: Mikael
 */

#include "OpenCLDevice.h"
#include "OpenCLUtility.h"
#include <memory>
#include <stdexcept>

namespace ATML
{
namespace Helper
{

OpenCLDevice::OpenCLDevice(cl_context context, OpenCLDeviceInfo deviceInfo) :
		context(context), deviceInfo(deviceInfo)
{
	deviceID = deviceInfo.DeviceID();
	cl_int error;
	queue = clCreateCommandQueue(context, deviceID, 0, &error);
	CheckOpenCLError(error, "Could not initialize the command queue");
}

OpenCLDevice::OpenCLDevice(cl_context context, OpenCLDeviceInfo deviceInfo,
		cl_command_queue_properties properties) :
		context(context), deviceInfo(deviceInfo)
{
	deviceID = deviceInfo.DeviceID();
	cl_int error;
	queue = clCreateCommandQueue(context, deviceID, properties, &error);
	CheckOpenCLError(error, "Could not initialize the command queue");
}

OpenCLDevice::~OpenCLDevice()
{
	for (auto& programKeyValue : programsAndKernels)
	{
		auto& tuple = programKeyValue.second;
		auto& program = get<0>(tuple);
		auto& kernels = get<1>(tuple);

		for (auto& kernelKeyValue : kernels)
			if (kernelKeyValue.second)
				CheckOpenCLError(clReleaseKernel(kernelKeyValue.second),
						"Could not release the kernel");

		kernels.clear();

		if (program)
			CheckOpenCLError(clReleaseProgram(program),
					"Could not release the program");
	}

	programsAndKernels.clear();
	referenceCounter.clear();

	if (queue)
		CheckOpenCLError(clReleaseCommandQueue(queue),
				"Could not release the commande queue");
	if (context)
		CheckOpenCLError(clReleaseContext(context),
				"Could not release the context");
}

void OpenCLDevice::AddKernel(const OpenCLKernel* kernel)
{
	string programName = kernel->ProgramName();
	if (ProgramAdded(programName))
	{
		auto& programCounter = referenceCounter[programName];
		get<0>(programCounter)++;auto
		& kernelCounter = get<1>(programCounter);
		string kernelName = kernel->KernelName();
		if (KernelAdded(programName, kernelName))
			kernelCounter[kernelName]++;
		else
		{
			auto& programAndKernels = programsAndKernels[programName];
			auto& kernels = get<1>(programAndKernels);
			auto& program = get<0>(programAndKernels);
			cl_int errorCode;
			cl_kernel kernelToAdd = clCreateKernel(program, kernelName.c_str(),
					&errorCode);
			CheckOpenCLError(errorCode, "Could not create the kernel");
			pair<string, cl_kernel> keyValuePair(kernelName, kernelToAdd);

			kernelCounter.insert(make_pair<string, int>(string(kernelName), 1));
			kernels.insert(keyValuePair);
		}
	}
	else
	{
		cl_int error;
		string buildLog;
		string programCode = kernel->ProgramCode();
		auto rawBuffer = programCode.c_str();
		auto programLength = programCode.size();
		auto program = clCreateProgramWithSource(context, 1,
				(const char**) &rawBuffer, &programLength, &error);
		CheckOpenCLError(error, "Could not create the program from the source");

		//Now it's time to build the executables
		error = clBuildProgram(program, 1, &deviceID, NULL, NULL, NULL);

		if (error != CL_BUILD_PROGRAM_FAILURE && error != CL_SUCCESS)
			CheckOpenCLError(error, "Error when building the program");

		size_t logSize;
		CheckOpenCLError(
				clGetProgramBuildInfo(program, deviceID, CL_PROGRAM_BUILD_LOG,
						0, NULL, &logSize), "Could not get the build log");

		unique_ptr<char[]> buildLogBuffer(new char[logSize + 1]);
		CheckOpenCLError(
				clGetProgramBuildInfo(program, deviceID, CL_PROGRAM_BUILD_LOG,
						logSize, buildLogBuffer.get(), NULL),
				"Could not get the build log");

		buildLog = string(buildLogBuffer.get());

		cl_build_status buildStatus;
		CheckOpenCLError(clGetProgramBuildInfo(program, deviceID,
		CL_PROGRAM_BUILD_STATUS, sizeof(cl_build_status), &buildStatus, NULL),
				"Could not get the build status");
		if (buildStatus != CL_BUILD_SUCCESS)
		{
			stringstream stringStream;
			stringStream << "Build failed: \n" << buildLog.c_str() << "\n";
			throw OpenCLCompilationException(stringStream.str());
		}

		programsAndKernels.insert(
				make_pair<string,
						tuple<cl_program, unordered_map<string, cl_kernel>>>(
		string(programName),
		tuple<cl_program, unordered_map<string, cl_kernel>>(program, unordered_map<string, cl_kernel>())));

		referenceCounter.insert(
				make_pair<string, tuple<int, unordered_map<string, int>>>(
		string(programName),
		tuple<int, unordered_map<string, int>>(1, unordered_map<string, int>())));

		//Adding the kernel, we know that it couldn't be added since the program wasn't added

		auto& tuple = referenceCounter[programName];
		auto& kernelCounter = get<1>(tuple);
		string kernelName = kernel->KernelName();
		auto& programAndKernels = programsAndKernels[programName];
		auto& kernels = get<1>(programAndKernels);
		cl_int errorCode;
		cl_kernel kernelToAdd = clCreateKernel(program, kernelName.c_str(),
				&errorCode);
		CheckOpenCLError(errorCode, "Could not create the kernel");
		pair<string, cl_kernel> keyValuePair(kernelName, kernelToAdd);

		kernelCounter.insert(make_pair<string, int>(string(kernelName), 1));
		kernels.insert(keyValuePair);

	}
}
void OpenCLDevice::RemoveKernel(const OpenCLKernel* kernel)
{
	auto programName = kernel->ProgramName();
	auto kernelName = kernel->KernelName();
	if (!KernelAdded(programName, kernelName))
		throw invalid_argument("The kernel was not added to the device");

	auto& programCounter = referenceCounter[programName];
	get<0>(programCounter)--;auto
	& kernelCounter = get<1>(programCounter);
	if (kernelCounter[kernelName] > 1)
	{
		kernelCounter[kernelName]--;
		return;
	}

	auto& programAndKernels = programsAndKernels[programName];
	auto& kernels = get<1>(programAndKernels);
	auto& kernelToRemove = kernels[kernelName];
	if (kernelToRemove)
		CheckOpenCLError(clReleaseKernel(kernelToRemove),
				"Could not release the kernel");

	kernels.erase(kernelName);
	kernelCounter.erase(kernelName);

	//If the kernels are empty, we can remove the program as well
	if (kernels.size() != 0)
		return;

	if (get<0>(programCounter) != 0)
		throw runtime_error(
				"The program reference counter does not match the amount of referenced kernels on the program");

	auto& program = get<0>(programAndKernels);

	if (program)
		CheckOpenCLError(clReleaseProgram(program),
				"Could not release the program");

	programsAndKernels.erase(programName);
	referenceCounter.erase(programName);
}

void OpenCLDevice::ExecuteKernel(const OpenCLKernel* kernel, bool blocking)
{
	auto programName = kernel->ProgramName();
	auto kernelName = kernel->KernelName();

	if (!KernelAdded(programName, kernelName))
		throw invalid_argument("The kernel was not added to the device");

	auto& programAndKernels = programsAndKernels[programName];
	auto& kernels = get<1>(programAndKernels);
	auto& kernelToExecute = kernels[kernelName];

	auto memoryArguments = kernel->GetMemoryArguments();
	auto otherArguments = kernel->GetOtherArguments();

	for (auto& memoryArgument : memoryArguments)
	{
		auto memory = get<1>(memoryArgument);
		if (memory->owningDevice != this)
			throw invalid_argument(
					"The memory doesnt correspond to this device");
		CheckOpenCLError(
				clSetKernelArg(kernelToExecute, get<0>(memoryArgument),
						sizeof(cl_mem), &memory->memory),
				"Could not set the kernel argument");
	}
	for (auto& otherArgument : otherArguments)
		CheckOpenCLError(
				clSetKernelArg(kernelToExecute, get<0>(otherArgument),
						get<1>(otherArgument), get<2>(otherArgument)),
				"Could not set the kernel argument");

	auto globalWorkSize = kernel->GlobalWorkSize();
	auto localWorkSize = kernel->LocalWorkSize();

	auto globalDimensionSize = globalWorkSize.size();
	if (globalDimensionSize == 0)
		throw invalid_argument("You need to specify a global work size");

	if (localWorkSize.size() != 0)
	{
		if (localWorkSize.size() != globalDimensionSize)
			throw invalid_argument(
					"The dimension of the local and the global work size must be the same");

		for (size_t i = 0; i < globalDimensionSize; i++)
			if (globalWorkSize[i] % localWorkSize[i] != 0)
				throw invalid_argument(
						"The local work size is not divisble with the global work size");

		CheckOpenCLError(
				clEnqueueNDRangeKernel(queue, kernelToExecute,
						globalDimensionSize, NULL, globalWorkSize.data(),
						localWorkSize.data(), 0, NULL, NULL),
				"Could not enqueue the kernel to the device queue");
	}
	else
	{
		CheckOpenCLError(
				clEnqueueNDRangeKernel(queue, kernelToExecute,
						globalDimensionSize, NULL, globalWorkSize.data(), NULL,
						0, NULL, NULL),
				"Could not enqueue the kernel to the device queue");
	}

	if (blocking)
		clFinish(queue);

}

void OpenCLDevice::WaitForDeviceQueue()
{
	clFinish(queue);
}

bool OpenCLDevice::ProgramAdded(const string& programName)
{
	if (programsAndKernels.find(programName) == programsAndKernels.end())
		return false;
	else
		return true;
}
bool OpenCLDevice::KernelAdded(const string& programName,
		const string& kernelName)
{
	if (programsAndKernels.find(programName) == programsAndKernels.end())
		return false;
	else
	{
		auto& kernels = get<1>(programsAndKernels[programName]);
		if (kernels.find(kernelName) == kernels.end())
			return false;
		else
			return true;
	}
}

unique_ptr<OpenCLMemory> OpenCLDevice::CreateMemory(cl_mem_flags flags,
		size_t bytes)
{
	cl_int error;
	cl_mem memory = clCreateBuffer(context, flags, bytes, NULL, &error);
	CheckOpenCLError(error, "Could not create the OpenCL memory");
	return unique_ptr<OpenCLMemory>(new OpenCLMemory(memory, this, flags));
}

unique_ptr<OpenCLMemory> OpenCLDevice::CreateMemory(cl_mem_flags flags,
		size_t bytes, void* buffer)
{
	cl_int error;
	cl_mem memory = clCreateBuffer(context, flags, bytes, buffer, &error);
	CheckOpenCLError(error, "Could not create the OpenCL memory");
	return unique_ptr<OpenCLMemory>(new OpenCLMemory(memory, this, flags));
}

void OpenCLDevice::WriteMemory(OpenCLMemory* memory, size_t bytes, void* buffer,
		bool blockingCall)
{
	//We need to verify that this memory is fixed to this device
	//TODO: In the future, it's enough with context sharing
	if (memory->owningDevice != this)
		throw invalid_argument("The OpenCLMemory is not tied to the device");

	if (blockingCall)
		CheckOpenCLError(
				clEnqueueWriteBuffer(queue, memory->memory, CL_TRUE, 0, bytes,
						buffer, 0, NULL, NULL),
				"Could not write the buffer to the device");
	else
		CheckOpenCLError(
				clEnqueueWriteBuffer(queue, memory->memory, CL_FALSE, 0, bytes,
						buffer, 0, NULL, NULL),
				"Could not write the buffer to the device");
}

void OpenCLDevice::ReadMemory(OpenCLMemory* memory, size_t bytes, void* buffer,
		bool blockingCall)
{
	//We need to verify that this memory is fixed to this device
	//TODO: In the future, it's enough with context sharing
	if (memory->owningDevice != this)
		throw invalid_argument("The OpenCLMemory is not tied to the device");

	if (blockingCall)
		CheckOpenCLError(
				clEnqueueReadBuffer(queue, memory->memory, CL_TRUE, 0, bytes,
						buffer, 0, NULL, NULL),
				"Could not write the buffer to the device");
	else
		CheckOpenCLError(
				clEnqueueReadBuffer(queue, memory->memory, CL_FALSE, 0, bytes,
						buffer, 0, NULL, NULL),
				"Could not write the buffer to the device");
}

} /* namespace Helper */
} /* namespace ATML */
