/*
 * OCLDeviceTest.cpp
 *
 *  Created on: Apr 28, 2015
 *      Author: Mikael
 */
#define CATCH_CONFIG_MAIN
#include "catch/catch.hpp"
#include "Matuna.OCLHelper/OCLHelper.h"
#include <memory>

using namespace Matuna::Helper;

SCENARIO("Acquiring memory, writing memory, reading memory", "[OCLMemory][OCLDevice][OCLHelper]")
{
	auto platformInfos = OCLHelper::GetPlatformInfos();
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
	for (size_t i = 0; i < bufferSize; i++)
		rawInputBuffer[i] = static_cast<cl_float>(i);

	GIVEN("All the contexts with all devices with a single device queue")
	{
		WHEN("Creating the memory with the context")
		{
			THEN("We should be able to read the memory from every device inside the context")
			{
				for (auto& platformInfo : platformInfos)
				{
					INFO("Creating the context");
					auto context = OCLHelper::GetContext(platformInfo);
					INFO("Creating memory inside the context");
					auto inputMemory = context->CreateMemory(CL_MEM_READ_ONLY, sizeof(cl_float) * bufferSize);
					for (auto device : context->GetDevices())
					{
						INFO("Writing memory");
						device->WriteMemory(inputMemory.get(), sizeof(cl_float) * bufferSize, inputBuffer.get());
						auto rawOutputBuffer = outputBuffer.get();
						INFO("Reading the memory from the device")
							device->ReadMemory(inputMemory.get(), sizeof(cl_float) * bufferSize, outputBuffer.get());
						for (size_t i = 0; i < bufferSize; i++)
							CHECK(rawInputBuffer[i] == rawOutputBuffer[i]);
					}
				}
			}
		}

	}
}


SCENARIO("Making sure that we get exception when using memory from different contexts", "[OCLMemory][OCLDevice][OCLHelper]")
{
	auto platformInfos = OCLHelper::GetPlatformInfos();
	if (platformInfos.size() == 0)
	{
		WARN(
			"No platforms are detected. "
			"This is either because you are running "
			"a system without OCL drivers or that we have a bug in the GetPlatformInfo() function.");
	}

	if (platformInfos.size() == 1)
	{
		WARN(
			"Not enough OCL platforms on the system to perform memory exception test");
	}
	else
	{
		INFO("Creating all the available contexts");
		vector<unique_ptr<OCLContext>> contexts;
		for (auto& platformInfo : platformInfos)
			contexts.push_back(move(OCLHelper::GetContext(platformInfo)));

		vector<vector<OCLDevice*>> devices;
		for (auto& context : contexts)
			devices.push_back(context->GetDevices());

		WHEN("Creating memory on separate contexts"){
			vector<unique_ptr<OCLMemory>> memories;
			vector<cl_float> dummyBuffer;
			dummyBuffer.resize(100);
			for (auto& context : contexts)
				memories.push_back(context->CreateMemory(CL_MEM_READ_ONLY, sizeof(cl_float) * dummyBuffer.size()));

			auto count = memories.size();
			CHECK(count >= size_t(1));
			CHECK(devices.size() == count);

			THEN("We must have an exception when reading from devices in a different context")
			{
				for (size_t i = 1; i < count; i++)
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
