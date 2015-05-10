/*
 * OpenCLDevice.cpp
 *
 *  Created on: Apr 26, 2015
 *      Author: Mikael
 */

#include "OpenCLDevice.h"
#include "OpenCLUtility.h"
#include <stdexcept>
#include <stdio.h>

namespace ATML
{
namespace Helper
{

OpenCLDevice::OpenCLDevice(const OpenCLContext* const context,
		const OpenCLDeviceInfo& deviceInfo,
		const vector<cl_command_queue>& queues) :
		context(context), deviceInfo(deviceInfo), queues(queues), deviceID(
				deviceInfo.DeviceID())
{
	if (queues.size() == 0)
		throw invalid_argument(
				"We cannot initialize a device without a device queue");
}

OpenCLDevice::~OpenCLDevice()
{
	for (auto& queue : queues)
		CheckOpenCLError(clReleaseCommandQueue(queue),
				"Could not release the command queue");
}

void OpenCLDevice::ExecuteKernel(const OpenCLKernel* kernel, int queueIndex,
		bool blocking)
{

	if (!kernel->ContextSet())
		throw invalid_argument(
				"The kernel has not been attached to any context");
	if (!kernel->KernelSet())
		throw invalid_argument(
				"The kernel has not been set. Make sure you have attached the kernel to the correct context");
	if (kernel->GetContext() != context)
		throw invalid_argument(
				"The kernel has not been attached to the same context as the device.");

	auto kernelToExecute = kernel->GetKernel();
	auto globalWorkSize = kernel->GlobalWorkSize();
	auto localWorkSize = kernel->LocalWorkSize();

	auto globalDimensionSize = globalWorkSize.size();
	if (globalDimensionSize == 0)
		throw invalid_argument("You need to specify a global work size");

	auto queue = queues[queueIndex];

	if (localWorkSize.size() != 0)
	{
		if (localWorkSize.size() != globalDimensionSize)
			throw invalid_argument(
					"The dimension of the local and the global work size must be the same");

		for (size_t i = 0; i < globalDimensionSize; i++)
			if (globalWorkSize[i] % localWorkSize[i] != 0)
				throw invalid_argument(
						"The local work size is not divisable with the global work size");

		CheckOpenCLError(
				clEnqueueNDRangeKernel(queue, kernelToExecute,
						globalDimensionSize, nullptr, globalWorkSize.data(),
						localWorkSize.data(), 0, nullptr, nullptr),
				"Could not enqueue the kernel to the device queue");
	}
	else
	{
		CheckOpenCLError(
				clEnqueueNDRangeKernel(queue, kernelToExecute,
						globalDimensionSize, nullptr, globalWorkSize.data(),
						nullptr, 0, nullptr, nullptr),
				"Could not enqueue the kernel to the device queue");
	}

	if (blocking)
		clFinish(queue);

}

void OpenCLDevice::WaitForDeviceQueue(int queueIndex)
{
	clFinish(queues[queueIndex]);
}

void OpenCLDevice::WriteMemory(OpenCLMemory* memory, size_t bytes, void* buffer,
		int queueIndex, bool blockingCall)
{
	if (memory->OwningContext() != context)
		throw invalid_argument("The OpenCLMemory is not tied to the context");

	if (blockingCall)
		CheckOpenCLError(
				clEnqueueWriteBuffer(queues[queueIndex], memory->GetCLMemory(),
				CL_TRUE, 0, bytes, buffer, 0, nullptr, nullptr),
				"Could not write the buffer to the device");
	else
		CheckOpenCLError(
				clEnqueueWriteBuffer(queues[queueIndex], memory->GetCLMemory(),
				CL_FALSE, 0, bytes, buffer, 0, nullptr, nullptr),
				"Could not write the buffer to the device");
}

void OpenCLDevice::ReadMemory(OpenCLMemory* memory, size_t bytes, void* buffer,
		int queueIndex, bool blockingCall)
{
	if (memory->OwningContext() != context)
		throw invalid_argument("The OpenCLMemory is not tied to the context");

	if (blockingCall)
		CheckOpenCLError(
				clEnqueueReadBuffer(queues[queueIndex], memory->GetCLMemory(),
				CL_TRUE, 0, bytes, buffer, 0, nullptr, nullptr),
				"Could not write the buffer to the device");
	else
		CheckOpenCLError(
				clEnqueueReadBuffer(queues[queueIndex], memory->GetCLMemory(),
				CL_FALSE, 0, bytes, buffer, 0, nullptr, nullptr),
				"Could not write the buffer to the device");
}

} /* namespace Helper */
} /* namespace ATML */
