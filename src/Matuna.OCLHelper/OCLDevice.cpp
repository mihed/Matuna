/*
* OCLDevice.cpp
*
*  Created on: Apr 26, 2015
*      Author: Mikael
*/

#include "OCLDevice.h"
#include "OCLUtility.h"
#include <stdexcept>
#include <stdio.h>

namespace Matuna
{
	namespace Helper
	{

		OCLDevice::OCLDevice(OCLContext* context,
			const OCLDeviceInfo& deviceInfo,
			const vector<cl_command_queue>& queues) : deviceInfo(deviceInfo)
		{
			if (queues.size() == 0)
				throw invalid_argument(
				"We cannot initialize a device without a device queue");

			this->context = context;
			this->queues = queues;
			this->deviceID = deviceInfo.DeviceID();
		}

		OCLDevice::~OCLDevice()
		{
			for (auto& queue : queues)
				CheckOCLError(clReleaseCommandQueue(queue),
				"Could not release the command queue");
		}

		void OCLDevice::ExecuteTask(const OCLKernel* kernel, int queueIndex,
			bool blocking)
		{
			if (!kernel->GetProgram()->ContextSet())
				throw invalid_argument(
				"The kernel has not been attached to any context");
			if (!kernel->KernelSet())
				throw invalid_argument(
				"The kernel has not been set. Make sure you have attached the kernel to the correct context");
			if (kernel->GetProgram()->GetContext() != context)
				throw invalid_argument(
				"The kernel has not been attached to the same context as the device.");

			auto kernelToExecute = kernel->GetKernel();
			auto queue = queues[queueIndex];

			CheckOCLError(clEnqueueTask(queue, kernelToExecute, 0, nullptr, nullptr),
				"Could not enqueue the kernel to the device queue");

			if (blocking)
				clFinish(queue);
		}

		void OCLDevice::ExecuteKernel(const OCLKernel* kernel, int queueIndex,
			bool blocking)
		{

			if (!kernel->GetProgram()->ContextSet())
				throw invalid_argument(
				"The kernel has not been attached to any context");
			if (!kernel->KernelSet())
				throw invalid_argument(
				"The kernel has not been set. Make sure you have attached the kernel to the correct context");
			if (kernel->GetProgram()->GetContext() != context)
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

				CheckOCLError(
					clEnqueueNDRangeKernel(queue, kernelToExecute,
					static_cast<cl_uint>(globalDimensionSize), nullptr, globalWorkSize.data(),
					localWorkSize.data(), 0, nullptr, nullptr),
					"Could not enqueue the kernel to the device queue");
			}
			else
			{
				CheckOCLError(
					clEnqueueNDRangeKernel(queue, kernelToExecute,
					static_cast<cl_uint>(globalDimensionSize), nullptr, globalWorkSize.data(),
					nullptr, 0, nullptr, nullptr),
					"Could not enqueue the kernel to the device queue");
			}

			if (blocking)
				clFinish(queue);

		}

		void OCLDevice::WaitForDeviceQueue(int queueIndex)
		{
			clFinish(queues[queueIndex]);
		}

		OCLKernelInfo OCLDevice::GetKernelInfo(const OCLKernel* kernel)
		{
			if (!kernel->GetProgram()->ContextSet())
				throw invalid_argument(
				"The kernel has not been attached to any context");
			if (!kernel->KernelSet())
				throw invalid_argument(
				"The kernel has not been set. Make sure you have attached the kernel to the correct context");
			if (kernel->GetProgram()->GetContext() != context)
				throw invalid_argument(
				"The kernel has not been attached to the same context as the device.");

			size_t workGroupSize;
			CheckOCLError(clGetKernelWorkGroupInfo(kernel->GetKernel(), deviceID,
				CL_KERNEL_WORK_GROUP_SIZE, sizeof(workGroupSize), &workGroupSize, nullptr),
				"Could not get the kernel work group info");

			size_t compileWorkGroupSizes[3];
			CheckOCLError(
				clGetKernelWorkGroupInfo(kernel->GetKernel(), deviceID,
				CL_KERNEL_COMPILE_WORK_GROUP_SIZE, sizeof(compileWorkGroupSizes),
				&compileWorkGroupSizes, nullptr),
				"Could not get the kernel work group info");

			vector<size_t> workGroupSizes;
			workGroupSizes.push_back(compileWorkGroupSizes[0]);
			workGroupSizes.push_back(compileWorkGroupSizes[1]);
			workGroupSizes.push_back(compileWorkGroupSizes[2]);

			cl_ulong localMemorySize;
			CheckOCLError(
				clGetKernelWorkGroupInfo(kernel->GetKernel(), deviceID,
				CL_KERNEL_LOCAL_MEM_SIZE, sizeof(localMemorySize), &localMemorySize,
				nullptr), "Could not get the kernel work group info");

			return OCLKernelInfo(workGroupSize, workGroupSizes, localMemorySize);
		}

		void OCLDevice::CopyCLMemory(OCLMemory* source, OCLMemory* destination,
			size_t sourceOffset, size_t destinationOffset, size_t bytes,
			int queueIndex, bool blockingCall)
		{
			if (source->OwningContext() != context)
				throw invalid_argument("The OCLMemory is not tied to the context");

			if (destination->OwningContext() != context)
				throw invalid_argument("The OCLMemory is not tied to the context");

			auto queue = queues[queueIndex];

			CheckOCLError(
				clEnqueueCopyBuffer(queue, source->GetCLMemory(),
				destination->GetCLMemory(), sourceOffset, destinationOffset,
				bytes, 0, nullptr, nullptr), "Could not copy the buffer");

			if (blockingCall)
				clFinish(queue);
		}

		void OCLDevice::WriteMemory(OCLMemory* memory, size_t bytes, void* buffer,
			int queueIndex, bool blockingCall)
		{
			if (memory->OwningContext() != context)
				throw invalid_argument("The OCLMemory is not tied to the context");

			if (blockingCall)
				CheckOCLError(
				clEnqueueWriteBuffer(queues[queueIndex], memory->GetCLMemory(),
				CL_TRUE, 0, bytes, buffer, 0, nullptr, nullptr),
				"Could not write the buffer to the device");
			else
				CheckOCLError(
				clEnqueueWriteBuffer(queues[queueIndex], memory->GetCLMemory(),
				CL_FALSE, 0, bytes, buffer, 0, nullptr, nullptr),
				"Could not write the buffer to the device");
		}

		void OCLDevice::ReadMemory(OCLMemory* memory, size_t bytes, void* buffer,
			int queueIndex, bool blockingCall)
		{
			if (memory->OwningContext() != context)
				throw invalid_argument("The OCLMemory is not tied to the context");

			if (blockingCall)
			{
				CheckOCLError(
					clEnqueueReadBuffer(queues[queueIndex], memory->GetCLMemory(),
					CL_TRUE, 0, bytes, buffer, 0, nullptr, nullptr),
					"Could not write the buffer to the device");
				WaitForDeviceQueue(queueIndex);
			}
			else
				CheckOCLError(
				clEnqueueReadBuffer(queues[queueIndex], memory->GetCLMemory(),
				CL_FALSE, 0, bytes, buffer, 0, nullptr, nullptr),
				"Could not write the buffer to the device");
		}

		OCLDeviceInfo OCLDevice::DeviceInfo() const
		{
			return deviceInfo;
		}


		cl_device_id OCLDevice::DeviceID() const
		{
			return deviceID;
		}


	} /* namespace Helper */
} /* namespace Matuna */
