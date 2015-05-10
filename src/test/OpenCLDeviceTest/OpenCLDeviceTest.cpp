/*
 * OpenCLDeviceTest.cpp
 *
 *  Created on: Apr 28, 2015
 *      Author: Mikael
 */
#define CATCH_CONFIG_MAIN
#include "catch/catch.hpp"
#include "OpenCLHelper/OpenCLDeviceHandler.h"
#include "TestKernel.h"
#include <memory>

using namespace ATML::Helper;

SCENARIO("Acquiring memory, writing memory, reading memory", "[OpenCLMemory][OpenCLDevice][OpenCLDeviceHandler]")
{
	auto platformInfos = OpenCLDeviceHandler::GetPlatformInfos();
	if (platformInfos.size() == 0)
	{
		WARN(
			"No platforms are detected. "
			"This is either because you are running "
			"a system without OCL drivers or that we have a bug in the GetPlatformInfo() function.");
	}

	INFO("Initializing buffers")
		const size_t bufferSize = 10000;
	unique_ptr<cl_float[]> inputBuffer(new cl_float[bufferSize]);
	unique_ptr<cl_float[]> outputBuffer(new cl_float[bufferSize]);
	auto rawInputBuffer = inputBuffer.get();
	for (int i = 0; i < bufferSize; i++)
		rawInputBuffer[i] = static_cast<int>(i);

	GIVEN("All the contexts with all devices with a single device queue")
	{
		WHEN("Creating the memory with the context")
		{
			THEN("We should be able to read the memory from every device inside the context")
			{
				for (auto& platformInfo : platformInfos)
				{
					INFO("Creating the context");
					auto context = OpenCLDeviceHandler::GetContext(platformInfo);
					INFO("Creating memory inside the context");
					auto inputMemory = context->CreateMemory(CL_MEM_READ_ONLY, sizeof(cl_float) * bufferSize);
					for (auto device : context->GetDevices())
					{
						INFO("Writing memory");
						device->WriteMemory(inputMemory.get(), sizeof(cl_float) * bufferSize, inputBuffer.get());
						auto rawOutputBuffer = outputBuffer.get();
						INFO("Reading the memory from the device")
							device->ReadMemory(inputMemory.get(), sizeof(cl_float) * bufferSize, outputBuffer.get());
						for (int i = 0; i < bufferSize; i++)
							CHECK(rawInputBuffer[i] == rawOutputBuffer[i]);
					}
				}
			}
		}

	}
}


SCENARIO("Making sure that we get exception when using memory from different contexts", "[OpenCLMemory][OpenCLDevice][OpenCLDeviceHandler]")
{
	auto platformInfos = OpenCLDeviceHandler::GetPlatformInfos();
	if (platformInfos.size() == 0)
	{
		WARN(
			"No platforms are detected. "
			"This is either because you are running "
			"a system without OCL drivers or that we have a bug in the GetPlatformInfo() function.");
	}

	int dummy = 100;
	if (platformInfos.size() == 1)
	{
		WARN(
			"Not enough OCL platforms on the system to perform memory exception test");
	}
	else
	{
		INFO("Creating all the available contexts");
		vector<unique_ptr<OpenCLContext>> contexts;
		for (auto& platformInfo : platformInfos)
			contexts.push_back(move(OpenCLDeviceHandler::GetContext(platformInfo)));

		vector<vector<OpenCLDevice*>> devices;
		for (auto& context : contexts)
			devices.push_back(context->GetDevices());

		WHEN("Creating memory on separate contexts"){
			vector<unique_ptr<OpenCLMemory>> memories;
			vector<cl_float> dummyBuffer;
			dummyBuffer.resize(100);
			for (auto& context : contexts)
				memories.push_back(context->CreateMemory(CL_MEM_READ_ONLY, sizeof(cl_float) * dummyBuffer.size()));

			auto count = memories.size();
			CHECK(count >= 1);
			CHECK(devices.size() == count);

			THEN("We must have an exception when reading from devices in a different context")
			{
				for (int i = 1; i < count; i++)
				{
					auto firstDevice = devices[i - 1][0];
					auto secondDevice = devices[i][0];
					auto& firstMemory = memories[i - 1];
					auto& secondMemory = memories[i];

					CHECK_THROWS(firstDevice->ReadMemory(secondMemory.get(), sizeof(cl_float) * dummyBuffer.size(), dummyBuffer.data()));
					CHECK_THROWS(secondDevice->ReadMemory(firstMemory.get(), sizeof(cl_float) * dummyBuffer.size(), dummyBuffer.data()));
				}
			}
		}
	}
}
SCENARIO("Creating kernels from the context", "[OpenCLDeviceHandler][OpenCLContext][OpenCLKernel][OpenCLKernelProgram]")
{
	auto platformInfos = OpenCLDeviceHandler::GetPlatformInfos();
	if (platformInfos.size() == 0)
	{
		WARN(
			"No platforms are detected. "
			"This is either because you are running "
			"a system without OCL drivers or that we have a bug in the GetPlatformInfo() function.");
	}
	GIVEN("A OpenCLKernelProgram implementation")
	{
		TestKernel kernel;
		WHEN("Removing a program that has not been added")
		{
			THEN("We must have an exception")
			{
				for (auto& platformInfo : platformInfos)
				{
					auto context = OpenCLDeviceHandler::GetContext(platformInfo);
					CHECK_THROWS(context->RemoveProgram(kernel.ProgramName()));
				}
			}
		}
		WHEN("Removing a program that has been added")
		{
			THEN("We must not have exception")
			{
				for (auto& platformInfo : platformInfos)
				{
					auto context = OpenCLDeviceHandler::GetContext(platformInfo);
					context->AddProgramFromSource(kernel.ProgramName(), kernel.GetCompilerOptions(), kernel.GetProgramCode(), context->GetDevices());
					context->RemoveProgram(kernel.ProgramName());
				}
			}
		}
		WHEN("Removing a program that has been added by the create program kernel command")
		{
			THEN("We must not have exception")
			{
				for (auto& platformInfo : platformInfos)
				{
					//TODO:
					//auto context = OpenCLDeviceHandler::GetContext(platformInfo);
					//auto testKernel = context->CreateOpenCLKernelProgram<TestKernel>(context->GetDevices());
					//INFO("Release the kernel before the program to be rigorous");
					//testKernel.reset();
					//context->RemoveProgram(kernel.ProgramName());
				}
			}
		}
		WHEN("Removing a program that has been added by the create kernel command")
		{
			THEN("We must have an exception")
			{
				for (auto& platformInfo : platformInfos)
				{
					//TODO:
					//auto context = OpenCLDeviceHandler::GetContext(platformInfo);
					//CHECK_THROWS(context->CreateOpenCLKernel<TestKernel>());
					//CHECK_THROWS(context->RemoveProgram(kernel.ProgramName()));
				}
			}
		}

		/*WHEN("Adding a binary source file that we have previously added")
		{
		THEN("We must not have exception")
		{
		for (auto& platformInfo : platformInfos)
		{
		cout << platformInfo.GetString().c_str() << endl;

		auto context = OpenCLDeviceHandler::GetContext(platformInfo);
		vector<OpenCLDevice*> oneDevice;
		oneDevice.push_back(context->GetDevices()[0]);
		auto testKernel = context->CreateOpenCLKernelProgram<TestKernel>(oneDevice);

		INFO("Fetching the binary program");
		auto binaries = context->GetBinaryProgram(testKernel->ProgramName());

		INFO("Removing the old program so that we can re add with the binaries");
		context->RemoveProgram(testKernel->ProgramName());

		INFO("Adding a binary program");
		context->AddProgramFromBinary(testKernel->ProgramName(), binaries, oneDevice);
		}
		}
		}*/
	}
}

SCENARIO("Executing an OCL kernel", "[OpenCLDevice][OpenCLDeviceHandler][OpenCLKernel][OpenCLMemory]")
{
	auto platformInfos = OpenCLDeviceHandler::GetPlatformInfos();
	if (platformInfos.size() == 0)
	{
		WARN(
			"No platforms are detected. "
			"This is either because you are running "
			"a system without OCL drivers or that we have a bug in the GetPlatformInfo() function.");
	}

	INFO("Initializing the buffers");
	const size_t bufferSize = 1000;
	unique_ptr<cl_float[]> inputBuffer(new cl_float[bufferSize]);
	unique_ptr<cl_float[]> outputBuffer(new cl_float[bufferSize]);
	auto rawInputBuffer = inputBuffer.get();
	for (int i = 0; i < bufferSize; i++)
		rawInputBuffer[i] = i;

	auto epsilon = numeric_limits<float>::epsilon();


	for (auto& platformInfo : platformInfos)
	{
		INFO("Creating the context");
		auto context = OpenCLDeviceHandler::GetContext(platformInfo);

		INFO("Making sure that we get an exception without adding");
		CHECK_THROWS(context->RemoveProgram(""));

		INFO("Creating the necessary memory");
		shared_ptr<OpenCLMemory> input1Memory(move(context->CreateMemory(CL_MEM_READ_ONLY, sizeof(cl_float) * bufferSize)));
		shared_ptr<OpenCLMemory> input2Memory(move(context->CreateMemory(CL_MEM_READ_ONLY, sizeof(cl_float) * bufferSize)));
		shared_ptr<OpenCLMemory> outputMemory(move(context->CreateMemory(CL_MEM_WRITE_ONLY, sizeof(cl_float) * bufferSize)));

		//TODO:
		/*auto kernel = context->CreateOpenCLKernelProgram<TestKernel>(context->GetDevices());

		kernel->SetInput1(input1Memory);
		kernel->SetInput2(input2Memory);
		kernel->SetOutput(outputMemory);
		kernel->SetMemorySize(bufferSize);
		CHECK(kernel->KernelSet());
		kernel->SetArguments();
		CHECK(kernel->ArgumentsSet());

		for (auto device : context->GetDevices())
		{

			INFO("Writing to the input memory");
			device->WriteMemory(input1Memory.get(), sizeof(cl_float) * bufferSize, inputBuffer.get());
			device->WriteMemory(input2Memory.get(), sizeof(cl_float) * bufferSize, inputBuffer.get());

			INFO("Making sure that the memory is correctly written");
			auto rawOutputBuffer = outputBuffer.get();
			device->ReadMemory(input1Memory.get(), sizeof(cl_float) * bufferSize, outputBuffer.get());
			for (int i = 0; i < bufferSize; i++)
				CHECK(rawInputBuffer[i] == rawOutputBuffer[i]);

			device->ReadMemory(input2Memory.get(), sizeof(cl_float) * bufferSize, outputBuffer.get());
			for (int i = 0; i < bufferSize; i++)
				CHECK(rawInputBuffer[i] == rawOutputBuffer[i]);

			device->ExecuteKernel(kernel.get());

			//Let us now read the output memory
			device->ReadMemory(outputMemory.get(), sizeof(cl_float) * bufferSize, outputBuffer.get());
			auto rawOutputPointer = outputBuffer.get();
			for (int i = 0; i < bufferSize; i++)
			{
				auto calculatedResult = cl_float(i) * cl_float(i);
				auto difference = abs(calculatedResult - rawOutputPointer[i]);
				CHECK(difference <= epsilon);
			}
		}

		//Release the memories before
		input1Memory.reset();
		input2Memory.reset();
		outputMemory.reset();

		//Release the kernel before the context to be rigorous
		kernel.reset();
		*/

	}
}
