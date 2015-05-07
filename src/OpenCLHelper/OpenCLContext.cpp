/*
 * OpenCLContext.cpp
 *
 *  Created on: May 6, 2015
 *      Author: Mikael
 */

#include "OpenCLContext.h"
#include <stdio.h>
#include <sstream>

namespace ATML
{
namespace Helper
{

OpenCLContext::OpenCLContext(
		const vector<tuple<OpenCLDeviceConfig, OpenCLDeviceInfo>>& deviceConfigs)
{
	vector<cl_device_id> deviceIDs;
	for (auto& configAndInfo : deviceConfigs)
		deviceIDs.push_back(get<1>(configAndInfo).DeviceID());

	cl_int error;
	context = clCreateContext(0, deviceIDs.size(), deviceIDs.data(), NULL, NULL,
			&error);
	CheckOpenCLError(error, "Could not create the native OpenCL context");

	//Create the device queues and create devices
	for (auto& configAndInfo : deviceConfigs)
	{
		vector<cl_command_queue> queues;
		for (auto& propertyFlag : get<0>(configAndInfo).GetCommandQueues())
		{
			cl_command_queue queue = clCreateCommandQueue(context,
					get<1>(configAndInfo).DeviceID(), propertyFlag, &error);
			queues.push_back(queue);
			CheckOpenCLError(error, "Could not create the device queues");
		}

		devices.push_back(
				unique_ptr<OpenCLDevice>(
						new OpenCLDevice(this, get<1>(configAndInfo), queues)));

	}
}

OpenCLContext::~OpenCLContext()
{
	for (auto& stringAndProgram : programs)
		CheckOpenCLError(clReleaseProgram(stringAndProgram.second),
				"Could not release the programs");

	programs.clear();
}

unique_ptr<OpenCLMemory> OpenCLContext::CreateMemory(cl_mem_flags flags,
		size_t bytes) const
{
	cl_int error;
	cl_mem memory = clCreateBuffer(context, flags, bytes, NULL, &error);
	CheckOpenCLError(error, "Could not create the OpenCL memory");
	return unique_ptr<OpenCLMemory>(
			new OpenCLMemory(memory, this, flags, bytes));
}

unique_ptr<OpenCLMemory> OpenCLContext::CreateMemory(cl_mem_flags flags,
		size_t bytes, void* buffer) const
{
	cl_int error;
	cl_mem memory = clCreateBuffer(context, flags, bytes, buffer, &error);
	CheckOpenCLError(error, "Could not create the OpenCL memory");
	return unique_ptr<OpenCLMemory>(
			new OpenCLMemory(memory, this, flags, bytes));
}

vector<OpenCLDevice*> OpenCLContext::GetDevices() const
{
	vector<OpenCLDevice*> result;
	for (auto& device : devices)
		result.push_back(device.get());

	return result;
}

void OpenCLContext::AddProgramFromSource(const string& programName,
		const string& compilerOptions, const vector<string>& programCodeFiles,
		const vector<OpenCLDevice*>& devices)
{
	cl_int error;

	vector<const char*> rawProgramFiles;
	vector<size_t> rawProgramLengths;
	for (auto& programCode : programCodeFiles)
	{
		rawProgramFiles.push_back(programCode.c_str());
		rawProgramLengths.push_back(programCode.size());
	}

	cl_program program = clCreateProgramWithSource(context, rawProgramFiles.size(),
			rawProgramFiles.data(), rawProgramLengths.data(), &error);
	CheckOpenCLError(error, "Could not create the program from the source");

	vector<cl_device_id> deviceIDs;
	for (auto device : devices)
		deviceIDs.push_back(device->DeviceID());

	error = clBuildProgram(program, deviceIDs.size(), deviceIDs.data(),
			compilerOptions.c_str(),
			NULL, NULL);

	if (error != CL_BUILD_PROGRAM_FAILURE && error != CL_SUCCESS)
		CheckOpenCLError(error, "Error when building the program");

	for (auto& deviceID : deviceIDs)
	{
		size_t logSize;
		CheckOpenCLError(
				clGetProgramBuildInfo(program, deviceID, CL_PROGRAM_BUILD_LOG,
						0,
						NULL, &logSize), "Could not get the build log");

		unique_ptr<char[]> buildLogBuffer(new char[logSize + 1]);
		CheckOpenCLError(
				clGetProgramBuildInfo(program, deviceID, CL_PROGRAM_BUILD_LOG,
						logSize, buildLogBuffer.get(), NULL),
				"Could not get the build log");

		string buildLog = string(buildLogBuffer.get());

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
	}

	programs.insert(make_pair(programName, program));
}

void OpenCLContext::AddProgramFromBinary(const string& programName,
		const size_t* lengths, const unsigned char** binaries,
		const vector<OpenCLDevice*>& devices)
{

	vector<cl_device_id> deviceIDs;
	for (auto device : devices)
		deviceIDs.push_back(device->DeviceID());

	cl_int error;
	cl_int binaryStatus;
	auto program = clCreateProgramWithBinary(context, devices.size(),
			deviceIDs.data(), lengths, binaries, &binaryStatus, &error);

	CheckOpenCLError(error, "Could not create the program from the binaries");

	programs.insert(make_pair(programName, program));
}

void OpenCLContext::RemoveProgram(const string& programName)
{
	auto& program = programs[programName];
	CheckOpenCLError(clReleaseProgram(program),
			"Could not release the program");

	programs.erase(programName);
}

} /* namespace Helper */
} /* namespace ATML */
