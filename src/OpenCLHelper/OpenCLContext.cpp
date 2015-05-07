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
			for (auto& stringAndProgram : programs)
				CheckOpenCLError(clReleaseProgram(stringAndProgram.second),
				"Could not release the programs");

			programs.clear();

			//Make sure the devices clean up before deleting the context
			devices.clear();

			CheckOpenCLError(clReleaseContext(context), "Could not release the context");
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
			const vector<OpenCLDevice*>& affectedDevices)
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
			for (auto device : affectedDevices)
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

		void OpenCLContext::RemoveProgram(const string& programName)
		{
			if (programs.find(programName) == programs.end())
				throw invalid_argument(
				"The program has not been added.");

			auto& program = programs[programName];
			CheckOpenCLError(clReleaseProgram(program),
				"Could not release the program");

			programs.erase(programName);
		}

	} /* namespace Helper */
} /* namespace ATML */
