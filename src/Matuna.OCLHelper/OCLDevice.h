/*
 * OCLDevice.h
 *
 *  Created on: Apr 26, 2015
 *      Author: Mikael
 */

#ifndef MATUNA_OCLHELPER_OCLDEVICE_H_
#define MATUNA_OCLHELPER_OCLDEVICE_H_

#include "OCLInclude.h"
#include <unordered_map>
#include <tuple>
#include <string>
#include <memory>

#include "OCLDeviceInfo.h"
#include "OCLKernel.h"
#include "OCLMemory.h"
#include "OCLKernelInfo.h"
#include "OCLDeviceConfig.h"

using namespace std;

namespace Matuna
{
namespace Helper
{

class OCLContext;

/**
 *@brief An OCL device on the system.
 *
 *This device is responsible for executing kernels and creating memory on the device.
 */
class OCLDevice
final
{
	private:
		const OCLContext* const context;
		vector<cl_command_queue> queues;
		cl_device_id deviceID;
		OCLDeviceInfo deviceInfo;
	public:
		/**
		 *@brief Instantiate the device from the given context and OCLDeviceInfo.
		 *
		 *An OCL context is created with one or more devices.
		 *Contexts are used by the OCL runtime for managing objects such as command-queues, memory, program and kernel objects and for executing kernels on one or more devices specified in the context.
		 *The default command queue is in order execution.
		 *
		 *@param context The context on which the device runs
		 *@param deviceInfo The OCLDeviceInfo containing the information about the device to be created.
		 */
		OCLDevice(const OCLContext* const context,
				const OCLDeviceInfo& deviceInfo,
				const vector<cl_command_queue>& queues);

		~OCLDevice();

		/**
		 *@brief Execute the kernel on the device
		 *
		 *@param kernel OCLKernel that is to be executed
		 *@param blocking True if the call should block the execution.
		 */
		void ExecuteKernel(const OCLKernel* kernel, int queueIndex = 0,
				bool blocking = true);


		void ExecuteTask(const OCLKernel* kernel, int queueIndex = 0,
			bool blocking = true);

		OCLKernelInfo GetKernelInfo(const OCLKernel* kernel);

		/**
		 *@brief Waits for the execution of all kernels to finish.
		 *
		 *
		 */
		void WaitForDeviceQueue(int queueIndex = 0);

		/**
		 *@brief Writes memory from the host device to the OCLDevice.
		 *
		 *@param memory The OCLMemory to be written to.
		 *@param bytes The number of bytes to be written to the OCLMemory.
		 *@param buffer The buffer to be written to the OCLMemory.
		 *@param blockingCall True if the call is blocking.
		 */
		void WriteMemory(OCLMemory* memory, size_t bytes, void* buffer,
				int queueIndex = 0, bool blockingCall = true);

		/**
		 *@brief Reads the memory from the device to the host device.
		 *
		 *@param memory The OCLMemory containing the memory
		 *@param bytes The amount of bytes to be written to the buffer.
		 *@param buffer The buffer into which the OCLMemory will write its memory.
		 *@param blockingCall True if the call is blocking.
		 */
		void ReadMemory(OCLMemory* memory, size_t bytes, void* buffer,
				int queueIndex = 0, bool blockingCall = true);

		void CopyCLMemory(OCLMemory* source, OCLMemory* destination,
			size_t sourceOffset, size_t destinationOffset, size_t bytes, int queueIndex = 0, bool blockingCall = true);

		/**
		 *@brief Returns the OCLDeviceInfo describing this device.
		 *
		 *
		 *@return OCLDeviceInfo
		 */
		OCLDeviceInfo DeviceInfo() const
		{
			return deviceInfo;
		}
		;

		cl_device_id DeviceID() const
		{
			return deviceID;
		}
		;
	};

	}
	/* namespace Helper */
	} /* namespace Matuna */

#endif /* MATUNA_OCLHELPER_OCLDEVICE_H_ */
