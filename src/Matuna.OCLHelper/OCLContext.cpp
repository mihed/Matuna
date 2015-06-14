/*
* OCLContext.cpp
*
*  Created on: May 6, 2015
*      Author: Mikael
*/

#include "OCLContext.h"
#include <stdio.h>
#include <sstream>

namespace Matuna
{
	namespace Helper
	{

		OCLContext::OCLContext(const OCLPlatformInfo& platformInfo,
			const vector<tuple<OCLDeviceConfig, OCLDeviceInfo>>& deviceConfigs) :
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
			context = clCreateContext(0, deviceIDs.size(), deviceIDs.data(), nullptr,
				nullptr, &error);
			CheckOCLError(error, "Could not create the native OCL context");

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
					CheckOCLError(error, "Could not create the device queues");

					queues.push_back(queue);
				}

				devices.push_back(
					unique_ptr<OCLDevice>(
					new OCLDevice(this, get<1>(configAndInfo), queues)));

			}
		}

		OCLContext::~OCLContext()
		{

			//Correct order of cleanup. Don't change this unless you know what you are doing.
			programs.clear();
			devices.clear();

			//Could happen that we have an exception in the constructor
			if (context)
				CheckOCLError(clReleaseContext(context),
				"Could not release the context");
		}

		OCLProgram* OCLContext::GetProgram(string name) const
		{
			auto tempIterator = programs.find(name);
			if (tempIterator == programs.end())
				throw runtime_error("The program has not been attached");

			return get<0>((*tempIterator).second).get();
		}

		bool OCLContext::ProgramAdded(const string& name) const
		{
			return programs.find(name) != programs.end();
		}

		OCLPlatformInfo OCLContext::GetPlatformInfo() const
		{
			return platformInfo;
		}

		unique_ptr<OCLMemory> OCLContext::CreateMemory(cl_mem_flags flags,
			size_t bytes) const
		{
			cl_int error;
			cl_mem memory = clCreateBuffer(context, flags, bytes, nullptr, &error);
			CheckOCLError(error, "Could not create the OCL memory");
			unique_ptr<OCLMemory> result(new OCLMemory(memory, this, flags, bytes));
			return move(result);
		}

		unique_ptr<OCLMemory> OCLContext::CreateMemory(cl_mem_flags flags,
			size_t bytes, void* buffer) const
		{
			cl_int error;
			cl_mem memory = clCreateBuffer(context, flags, bytes, buffer, &error);
			CheckOCLError(error, "Could not create the OCL memory");
			unique_ptr<OCLMemory> result(new OCLMemory(memory, this, flags, bytes));
			return move(result);
		}

		vector<OCLDevice*> OCLContext::GetDevices() const
		{
			vector<OCLDevice*> result;
			for (auto& device : devices)
				result.push_back(device.get());

			return result;
		}

		void OCLContext::AttachProgram(unique_ptr<OCLProgram> program, const vector<OCLDevice*>& affectedDevices)
		{
			if (programs.find(program->GetName()) != programs.end())
				throw invalid_argument("The program has already been added!");

			auto programCodeFiles = program->GetProgramCodeFiles();
			if (programCodeFiles.size() == 0)
				throw invalid_argument("The program contains no code");

			auto compilerOptions = program->GetCompilerOptions();

			cl_int error;
			vector<const char*> rawProgramFiles;
			vector<size_t> rawProgramLengths;
			for (auto& programCode : programCodeFiles)
			{
				rawProgramFiles.push_back(programCode.c_str());
				rawProgramLengths.push_back(programCode.size());
			}

			cl_program clProgram = clCreateProgramWithSource(context,
				rawProgramFiles.size(), rawProgramFiles.data(),
				rawProgramLengths.data(), &error);
			CheckOCLError(error, "Could not create the program from the source");

			vector<cl_device_id> deviceIDs;
			for (auto device : affectedDevices)
				deviceIDs.push_back(device->DeviceID());

			error = clBuildProgram(clProgram, deviceIDs.size(), deviceIDs.data(),
				compilerOptions.c_str(), nullptr, nullptr);

			if (error != CL_BUILD_PROGRAM_FAILURE && error != CL_SUCCESS)
			{
				clReleaseProgram(clProgram);
				CheckOCLError(error, "Error when building the program");
			}

			for (auto& deviceID : deviceIDs)
			{
				size_t logSize;

				error = clGetProgramBuildInfo(clProgram, deviceID, CL_PROGRAM_BUILD_LOG,
					0, nullptr, &logSize);

				if (error != CL_SUCCESS)
					clReleaseProgram(clProgram);
				CheckOCLError(error, "Could not get the build log");

				unique_ptr<char[]> buildLogBuffer(new char[logSize + 1]);
				error = clGetProgramBuildInfo(clProgram, deviceID, CL_PROGRAM_BUILD_LOG,
					logSize, buildLogBuffer.get(), nullptr);

				if (error != CL_SUCCESS)
					clReleaseProgram(clProgram);
				CheckOCLError(error, "Could not get the build log");

				string buildLog = string(buildLogBuffer.get());

				cl_build_status buildStatus;
				error = clGetProgramBuildInfo(clProgram, deviceID,
					CL_PROGRAM_BUILD_STATUS, sizeof(cl_build_status), &buildStatus,
					nullptr);

				if (error != CL_SUCCESS)
					clReleaseProgram(clProgram);
				CheckOCLError(error, "Could not get the build status");

				if (buildStatus != CL_BUILD_SUCCESS)
				{
					clReleaseProgram(clProgram);
					stringstream stringStream;
					stringStream << "Build failed: \n" << buildLog.c_str() << "\n";
					throw OCLCompilationException(stringStream.str());
				}
			}

			program->SetContext(this);
			program->SetProgram(clProgram);


			//TODO: This hash is completely wrong and needs fixing!
			unordered_set<OCLDevice*> deviceMap;
			for (auto device : affectedDevices)
			{
				if (deviceMap.find(device) != deviceMap.end())
					throw invalid_argument("The device vector contains duplicate devices");
				deviceMap.insert(device);
			}

			string programName = program->GetName();
			programs.insert(make_pair(programName, make_tuple(move(program), deviceMap)));
		}

		void OCLContext::DetachProgram(OCLProgram* program)
		{
			string programName = program->GetName();
			if (programs.find(programName) == programs.end())
				throw invalid_argument("We detach a program that has not been attached");

			programs.erase(programName);
		}

	} /* namespace Helper */
} /* namespace Matuna */
