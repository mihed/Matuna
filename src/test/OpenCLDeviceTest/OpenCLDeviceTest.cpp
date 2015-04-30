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

SCENARIO("Fetching the OpenCLDevices from the OpenCLDeviceHandler", "[OpenCLDevice][OpenCLDeviceHandler]") {
	INFO("Getting the device information");
	auto deviceInfos = OpenCLDeviceHandler::GetDeviceInfos();
	INFO("Getting the platform information");
	auto platformInfos = OpenCLDeviceHandler::GetPlatformInfos();

	GIVEN("All of the devices"){
		auto allDevices = OpenCLDeviceHandler::GetDevices();

		if (allDevices.size() == 0)
			WARN("Could not retreive any devices, make sure you have an OCL capable system");

		WHEN("Fetching devices with all platform informations")
		{
			auto platformInfoDevices = OpenCLDeviceHandler::GetDevices(platformInfos);
			THEN("The amount of devices must be equal")
			{
				REQUIRE(allDevices.size() == platformInfoDevices.size());
			}
		}
		WHEN("Fetching devices with single platform infos")
		{
			size_t size2 = 0;
			for (auto& platformInfo : platformInfos)
			{
				auto devices = OpenCLDeviceHandler::GetDevices(platformInfo);
				size2 += devices.size();
			}
			THEN("Make sure that the count is correct")
			{
				REQUIRE(allDevices.size() == size2);
			}
		}
		WHEN("Fetching devices with single device info")
		{
			size_t size2 = 0;
			for (auto& deviceInfo : deviceInfos)
			{
				size2++;
				auto device = OpenCLDeviceHandler::GetDevices(deviceInfo);
			}
			THEN("Make sure that thec count is correct")
			{
				REQUIRE(allDevices.size() == size2);
			}
		}
		WHEN("Fetching devices with all device infos")
		{
			auto deviceInfoDevices = OpenCLDeviceHandler::GetDevices(deviceInfos);
			THEN("The amount of devices must be equal")
			{
				REQUIRE(allDevices.size() == deviceInfoDevices.size());
			}
		}
	}
}

SCENARIO("Acquiring memory, writing memory, reading memory", "[OpenCLMemory][OpenCLDevice][OpenCLDeviceHandler]")
{
	INFO("Getting all devices")
		auto devices = OpenCLDeviceHandler::GetDevices();

	if (devices.size() == 0)
		WARN("No OCL devices found, make sure your system supports OpenCL.");
	else
	{

		INFO("Initializing buffers")
			const size_t bufferSize = 10000;
		unique_ptr<cl_float[]> inputBuffer(new cl_float[bufferSize]);
		unique_ptr<cl_float[]> outputBuffer(new cl_float[bufferSize]);
		auto rawInputBuffer = inputBuffer.get();
		for (int i = 0; i < bufferSize; i++)
			rawInputBuffer[i] = static_cast<int>(i);
		GIVEN("All of the devices")
		{
			for (auto& device : devices)
			{
				WHEN("Writing to OCL memory")
				{
					auto inputMemory = device->CreateMemory(CL_MEM_READ_ONLY, sizeof(cl_float) * bufferSize);
					CHECK(inputMemory.get());
					device->WriteMemory(inputMemory.get(), sizeof(cl_float) * bufferSize, inputBuffer.get());
					THEN("The read memory must be equal")
					{
						auto rawOutputBuffer = outputBuffer.get();
						device->ReadMemory(inputMemory.get(), sizeof(cl_float) * bufferSize, outputBuffer.get());
						for (int i = 0; i < bufferSize; i++)
							CHECK(rawInputBuffer[i] == rawOutputBuffer[i]);
					}
				}
			}
		}
	}
}

SCENARIO("Making sure that we get exception when using memory from different devices", "[OpenCLMemory][OpenCLDevice][OpenCLDeviceHandler]")
{
	INFO("Getting all devices");
	auto devices = OpenCLDeviceHandler::GetDevices();

	if (devices.size() == 0)
		WARN("No OCL devices found, make sure your system supports OpenCL.");
	else if (devices.size() == 1)
	{
		WARN("Not enought OCL devices on the system to perform this test");
	}
	else
	{
		auto& firstDevice = devices[0];
		auto& secondDevice = devices[1];

		auto firstMemory = firstDevice->CreateMemory(CL_MEM_READ_WRITE, sizeof(cl_float) * 100);

		CHECK_THROWS(secondDevice->ReadMemory(firstMemory.get(), sizeof(cl_float) * 100, 0));
	}
}

SCENARIO("Adding removing kernels from OCL devices", "[OpenCLDevice][OpenCLDeviceHandler][OpenCLKernel]")
{
	INFO("Getting all devices");
	auto devices = OpenCLDeviceHandler::GetDevices();

	if (devices.size() == 0)
		WARN("No OCL devices found, make sure your system supports OpenCL.");

	TestKernel kernel;
	GIVEN("OpenCLKernel and the devices")
	{
		WHEN("Removing a kernel that was not added")
		{
			THEN("We must have an exception")
			{
				INFO("Looping through all the devices");
				for (auto& device : devices)
				{
					CHECK_THROWS(device->RemoveKernel(&kernel));
				}
			}
		}
		WHEN("Adding a kernel for the first time")
		{
			THEN("We can remove it safely")
			{
				INFO("Looping through all the devices");
				for (auto& device : devices)
				{
					device->AddKernel(&kernel);
					device->RemoveKernel(&kernel);
				}
			}
		}

		WHEN("Adding multiple kernels")
		{
			INFO("Looping through all the devices");
			for (auto& device : devices)
				for (int i = 0; i < 10; i++)
					device->AddKernel(&kernel);
			THEN("We should be able to remove them without exception except for the last one")
			{
				INFO("Looping through all the devices");
				for (auto& device : devices)
				{
					for (int i = 0; i < 10; i++)
						device->RemoveKernel(&kernel);

					INFO("Removing one too many")
						CHECK_THROWS(device->RemoveKernel(&kernel));
				}
			}
		}
	}
}


SCENARIO("Executing an OCL kernel", "[OpenCLDevice][OpenCLDeviceHandler][OpenCLKernel][OpenCLMemory]")
{
	INFO("Getting all devices");
	auto devices = OpenCLDeviceHandler::GetDevices();

	if (devices.size() == 0)
		WARN("No OCL devices found, make sure your system supports OpenCL.");

	INFO("Initializing the buffers");
	const size_t bufferSize = 1000;
	unique_ptr<cl_float[]> inputBuffer(new cl_float[bufferSize]);
	unique_ptr<cl_float[]> outputBuffer(new cl_float[bufferSize]);
	auto rawInputBuffer = inputBuffer.get();
	for (int i = 0; i < bufferSize; i++)
		rawInputBuffer[i] = i;

	auto epsilon = numeric_limits<float>::epsilon();

	GIVEN("OpenCLKernel and the devices")
	{
		for (auto& device : devices)
		{
			TestKernel kernel;
			INFO("Creating the necessary memory");
			auto input1Memory = shared_ptr<OpenCLMemory>(move(device->CreateMemory(CL_MEM_READ_ONLY, sizeof(cl_float) * bufferSize)));
			auto input2Memory = shared_ptr<OpenCLMemory>(move(device->CreateMemory(CL_MEM_READ_ONLY, sizeof(cl_float) * bufferSize)));
			auto outputMemory = shared_ptr<OpenCLMemory>(move(device->CreateMemory(CL_MEM_WRITE_ONLY, sizeof(cl_float) * bufferSize)));

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

			kernel.SetInput1(input1Memory);
			kernel.SetInput2(input2Memory);
			kernel.SetOutput(outputMemory);
			kernel.SetMemorySize(bufferSize);
			device->AddKernel(&kernel);
			device->ExecuteKernel(&kernel);

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
	}
}