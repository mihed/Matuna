/*
 * OpenCLDevice.h
 *
 *  Created on: Apr 26, 2015
 *      Author: Mikael
 */

#ifndef ATML_OPENCLHELPER_OPENCLDEVICE_H_
#define ATML_OPENCLHELPER_OPENCLDEVICE_H_

#include <CL/cl.h>
#include <unordered_map>
#include <tuple>
#include <string>
#include <memory>

#include "OpenCLDeviceInfo.h"
#include "OpenCLKernel.h"

using namespace std;

namespace ATML {
namespace Helper {

class OpenCLDevice final{
private:
	cl_context context;
	cl_command_queue queue;
	cl_device_id deviceID;
	OpenCLDeviceInfo deviceInfo;
	unordered_map<string, tuple<int, unordered_map<string, int>>> referenceCounter;
	unordered_map<string, tuple<cl_program, unordered_map<string, cl_kernel>>> programsAndKernels;

public:
	OpenCLDevice(cl_context context, OpenCLDeviceInfo deviceInfo);

	OpenCLDevice(cl_context context, OpenCLDeviceInfo deviceInfo, cl_command_queue_properties properties);

	~OpenCLDevice();

	void AddKernel(const OpenCLKernel* kernel);
	void RemoveKernel(const OpenCLKernel* kernel);
	void ExecuteKernel(const OpenCLKernel* kernel, bool blocking = true);
	void WaitForDeviceQueue();

	unique_ptr<OpenCLMemory> CreateMemory(cl_mem_flags flags, size_t bytes);
	unique_ptr<OpenCLMemory> CreateMemory(cl_mem_flags flags, size_t bytes, void* buffer);

	void WriteMemory(OpenCLMemory* memory, size_t bytes, void* buffer, bool blockingCall = true);
	void ReadMemory(OpenCLMemory* memory, size_t bytes, void* buffer, bool blockingCall = true);

private:
	bool ProgramAdded(string programName);
	bool KernelAdded(string programName, string kernelName);
};

} /* namespace Helper */
} /* namespace ATML */

#endif /* ATML_OPENCLHELPER_OPENCLDEVICE_H_ */
