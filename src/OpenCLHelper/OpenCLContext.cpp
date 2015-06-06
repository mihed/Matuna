/*
 * OpenCLContext.cpp
 *
 *  Created on: May 6, 2015
 *      Author: Mikael
 */

#include "OpenCLContext.h"
#include <stdio.h>
#include <sstream>

namespace Matuna
{
namespace Helper
{

OpenCLContext::OpenCLContext(const OpenCLPlatformInfo& platformInfo,
		const vector<tuple<OpenCLDeviceConfig, OpenCLDeviceInfo>>& deviceConfigs) :
		platformInfo(platformInfo)
{

	for (auto& configInfoTuple : deviceConfigs)
		if (get<0>(configInfoTuple).CommandQueueCount() == 0)
			throw invalid_argument(
					"We cannot create a context with devices that has no device queues");

	vector<cl_device_id> deviceIDs;
	for (auto& configAndInfo : deviceConfigs)
		deviceIDs.push_back(get<1>(configAndInfo).DeviceID());

	cl_int error;
	context = nullptr;
	context = clCreateContext(0, deviceIDs.size(), deviceIDs.data(), nullptr,
			nullptr, &error);
	CheckOpenCLError(error, "Could not create the native OpenCL context");

	//Create the device queues and create devices
	for (auto& configAndInfo : deviceConfigs)
	{
		vector<cl_command_queue> queues;
		for (auto& propertyFlag : get<0>(configAndInfo).GetCommandQueues())
		{
			cl_command_queue queue = clCreateCommandQueue(context,
					get<1>(configAndInfo).DeviceID(), propertyFlag, &error);

			if (error != CL_SUCCESS)
				for (auto queueToClean : queues)
					clReleaseCommandQueue(queueToClean);
			CheckOpenCLError(error, "Could not create the device queues");

			queues.push_back(queue);
		}

		devices.push_back(
				unique_ptr<OpenCLDevice>(
						new OpenCLDevice(this, get<1>(configAndInfo), queues)));

	}
}

OpenCLContext::~OpenCLContext()
{
	for (auto& stringAndMap : kernels)
		for (auto& stringAndKernel : stringAndMap.second)
			CheckOpenCLError(clReleaseKernel(stringAndKernel.second),
					"Could not release the kernels. If this happens, you need to review the how the resources are handled in the context");

	for (auto& stringAndProgram : programs)
		CheckOpenCLError(clReleaseProgram(stringAndProgram.second),
				"Could not release the programs. If this happens, you need to review the how the resources are handled in the context");

	//Correct order of cleanup. Don't change this unless you know what you are doing.
	kernels.clear();
	programs.clear();
	devices.clear();

	//Could happen that we have an exception in the constructor
	if (context)
		CheckOpenCLError(clReleaseContext(context),
				"Could not release the context");
}

bool OpenCLContext::ProgramAdded(const string& name) const
{
	return programs.find(name) != programs.end();
}

bool OpenCLContext::KernelAdded(const string& programName,
		const string& kernelName) const
{
	if (kernels.find(programName) == kernels.end())
		return false;

	auto& kernelMap = kernels.find(programName)->second;
	return kernelMap.find(kernelName) != kernelMap.end();
}

OpenCLPlatformInfo OpenCLContext::GetPlatformInfo() const
{
	return platformInfo;
}

unique_ptr<OpenCLMemory> OpenCLContext::CreateMemory(cl_mem_flags flags,
		size_t bytes) const
{
	cl_int error;
	cl_mem memory = clCreateBuffer(context, flags, bytes, nullptr, &error);
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
		const vector<OpenCLDevice*>& affectedDevices)
{

	if (programs.find(programName) != programs.end())
		throw invalid_argument("The program has already been added.");

	if (kernels.find(programName) != kernels.end())
		throw invalid_argument(
				"The program has already been added in the kernels. This is an indication that there's something wrong in the implementation");

	cl_int error;
	vector<const char*> rawProgramFiles;
	vector<size_t> rawProgramLengths;
	for (auto& programCode : programCodeFiles)
	{
		rawProgramFiles.push_back(programCode.c_str());
		rawProgramLengths.push_back(programCode.size());
	}

	cl_program program = clCreateProgramWithSource(context,
			rawProgramFiles.size(), rawProgramFiles.data(),
			rawProgramLengths.data(), &error);
	CheckOpenCLError(error, "Could not create the program from the source");

	vector<cl_device_id> deviceIDs;
	for (auto device : affectedDevices)
		deviceIDs.push_back(device->DeviceID());

	error = clBuildProgram(program, deviceIDs.size(), deviceIDs.data(),
			compilerOptions.c_str(), nullptr, nullptr);

	if (error != CL_BUILD_PROGRAM_FAILURE && error != CL_SUCCESS)
	{
		clReleaseProgram(program);
		CheckOpenCLError(error, "Error when building the program");
	}

	for (auto& deviceID : deviceIDs)
	{
		size_t logSize;

		error = clGetProgramBuildInfo(program, deviceID, CL_PROGRAM_BUILD_LOG,
				0, nullptr, &logSize);

		if (error != CL_SUCCESS)
			clReleaseProgram(program);
		CheckOpenCLError(error, "Could not get the build log");

		unique_ptr<char[]> buildLogBuffer(new char[logSize + 1]);
		error = clGetProgramBuildInfo(program, deviceID, CL_PROGRAM_BUILD_LOG,
				logSize, buildLogBuffer.get(), nullptr);

		if (error != CL_SUCCESS)
			clReleaseProgram(program);
		CheckOpenCLError(error, "Could not get the build log");

		string buildLog = string(buildLogBuffer.get());

		cl_build_status buildStatus;
		error = clGetProgramBuildInfo(program, deviceID,
		CL_PROGRAM_BUILD_STATUS, sizeof(cl_build_status), &buildStatus,
				nullptr);

		if (error != CL_SUCCESS)
			clReleaseProgram(program);
		CheckOpenCLError(error, "Could not get the build status");

		if (buildStatus != CL_BUILD_SUCCESS)
		{
			clReleaseProgram(program);
			stringstream stringStream;
			stringStream << "Build failed: \n" << buildLog.c_str() << "\n";
			throw OpenCLCompilationException(stringStream.str());
		}
	}

	kernels.insert(make_pair(programName, unordered_map<string, cl_kernel>()));
	programs.insert(make_pair(programName, program));
}

void OpenCLContext::AddProgramFromSource(OpenCLKernelProgram* kernelProgram,
		const vector<OpenCLDevice*>& affectedDevices)
{
	AddProgramFromSource(kernelProgram->ProgramName(),
			kernelProgram->GetCompilerOptions(),
			kernelProgram->GetProgramCode(), affectedDevices);
}

void OpenCLContext::AddKernel(OpenCLKernel* kernel)
{
	if (programs.find(kernel->ProgramName()) == programs.end())
		throw invalid_argument(
				"The program where the kernel belongs has not been added.");

	if (kernels.find(kernel->ProgramName()) == kernels.end())
		throw invalid_argument(
				"The program has not been added to the kernels. This is an indication that there's something wrong in the implementation");

	auto& program = programs[kernel->ProgramName()];

	cl_int errorCode;
	cl_kernel kernelToSet = clCreateKernel(program,
			kernel->KernelName().c_str(), &errorCode);
	CheckOpenCLError(errorCode, "Could not create the kernel");

	kernel->SetOCLKernel(kernelToSet);
	kernel->SetContext(this);

	if (!kernel->KernelSet())
	{
		clReleaseKernel(kernelToSet);
		throw runtime_error(
				"The context could not attach a native kernel to the OpenCLKernel");
	}

	if (!kernel->ContextSet())
	{
		clReleaseKernel(kernelToSet);
		throw runtime_error(
				"The context could not attach itself to the OpenCLKernel");
	}

	auto& kernelMap = kernels[kernel->ProgramName()];
	kernelMap.insert(make_pair(kernel->KernelName(), kernelToSet));
}

void OpenCLContext::RemoveProgram(const string& programName)
{
	if (programs.find(programName) == programs.end())
		throw invalid_argument("The program has not been added.");

	if (kernels.find(programName) == kernels.end())
		throw invalid_argument(
				"The program has not been added to the kernels. This is an indication that there's something wrong in the implementation");

	//Start with removing all the associated kernels
	for (auto& stringAndKernel : kernels[programName])
		CheckOpenCLError(clReleaseKernel(stringAndKernel.second),
				"Could not release the kernels. If this happens, you need to review the how the resources are handled in the context");
	kernels.erase(programName);

	auto& program = programs[programName];
	CheckOpenCLError(clReleaseProgram(program),
			"Could not release the program");
	programs.erase(programName);
}

void OpenCLContext::RemoveProgram(OpenCLKernelProgram* kernelProgram)
{
	RemoveProgram(kernelProgram->ProgramName());
}

void OpenCLContext::RemoveKernel(OpenCLKernel* kernel)
{
	if (programs.find(kernel->ProgramName()) == programs.end())
		throw invalid_argument("There's no program associated to the kernel.");

	if (kernels.find(kernel->ProgramName()) == kernels.end())
		throw invalid_argument(
				"There's no program associated to the kernel. This is an indication that there's something wrong in the implementation");

	auto& kernelMap = kernels[kernel->ProgramName()];

	if (kernelMap.find(kernel->KernelName()) == kernelMap.end())
		throw invalid_argument("The kernel has not been added.");

	CheckOpenCLError(clReleaseKernel(kernelMap[kernel->KernelName()]),
			"Could not remove the kernel.");

	kernelMap.erase(kernel->KernelName());
}

//void OpenCLContext::AddProgramFromBinary(const string& programName,
//	const vector<vector<unsigned char>>& binaries,
//	const vector<OpenCLDevice*>& affectedDevices)
//{

//	if (devices.size() != binaries.size())
//		throw invalid_argument("The number of devices must match the number of binaries");

//	vector<size_t> lengths;
//	for (auto& binary : binaries)
//		lengths.push_back(binary.size());

//	vector<cl_device_id> deviceIDs;
//	for (auto device : affectedDevices)
//		deviceIDs.push_back(device->DeviceID());

//	unsigned char** tempBinaries = new unsigned char*[binaries.size()];
//	for (size_t i = 0; i < binaries.size(); i++)
//	{
//		tempBinaries[i] = new unsigned char[lengths[i]];
//		memcpy(tempBinaries[i], binaries[i].data(), lengths[i]);
//	}

//	cl_int error;
//	vector<cl_int> binaryStatus;
//	binaryStatus.resize(devices.size());

//	//The const_cast works here since we are in control of the tempBinaries variable.
//	cl_program program = clCreateProgramWithBinary(context, devices.size(),
//		deviceIDs.data(), lengths.data(), (const unsigned char**) tempBinaries, binaryStatus.data(), &error);

//	for (size_t i = 0; i < binaries.size(); i++)
//		delete[] tempBinaries[i];
//	delete[] tempBinaries;

//	CheckOpenCLError(error, "Could not create the program from the binaries");

//	//Make sure that the binaries has been loaded properly to the devices
//	for (auto status : binaryStatus)
//	{
//		if (status != CL_SUCCESS)
//		{
//			if (status == CL_INVALID_VALUE)
//				throw invalid_argument("The arguments are not valid");
//			else if (status == CL_INVALID_BINARY)
//				throw invalid_argument("The binary is not valid");
//			else
//				throw runtime_error("Unknown error when loading the binaries");
//		}
//	}

//	programs.insert(make_pair(programName, program));
//}

//vector<vector<unsigned char>> OpenCLContext::GetBinaryProgram(const string& programName)
//{
//	if (programs.find(programName) == programs.end())
//		throw invalid_argument(
//		"The program has not been added.");

//	auto& program = programs[programName];

//	cl_uint deviceCount;
//	CheckOpenCLError(clGetProgramInfo(program, CL_PROGRAM_NUM_DEVICES, sizeof(cl_uint), &deviceCount, NULL), "Could not fetch the number of attached devices to this program");

//	vector<size_t> binarySizes;
//	binarySizes.resize(deviceCount);
//	CheckOpenCLError(clGetProgramInfo(program, CL_PROGRAM_BINARY_SIZES, deviceCount * sizeof(size_t), binarySizes.data(), NULL), "Could not retreive the binary sizes");

//	unsigned char** binaries = new unsigned char*[deviceCount];
//	for (size_t i = 0; i < deviceCount; i++)
//		binaries[i] = new unsigned char[binarySizes[i]];

//	auto error = clGetProgramInfo(program, CL_PROGRAM_BINARIES, deviceCount * sizeof(unsigned char*), binaries, NULL);
//	if (error != CL_SUCCESS)
//	{
//		for (size_t i = 0; i < deviceCount; i++)
//			delete[] binaries[i];

//		delete[] binaries;

//		throw OpenCLException(error, "Could not get the binaries");
//	}

//	vector<vector<unsigned char>> result;
//	result.resize(deviceCount);
//	for (size_t i = 0; i < deviceCount; i++)
//	{
//		result[i].resize(binarySizes[i]);
//		memcpy(result[i].data(), binaries[i], binarySizes[i]);
//	}

//	for (size_t i = 0; i < deviceCount; i++)
//		delete[] binaries[i];

//	delete[] binaries;

//	return result;
//}

} /* namespace Helper */
} /* namespace Matuna */
