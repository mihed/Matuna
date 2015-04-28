/*
 * OpenCLDeviceTest.cpp
 *
 *  Created on: Apr 28, 2015
 *      Author: Mikael
 */
#define CATCH_CONFIG_MAIN
#include "catch/catch.hpp"
#include "OpenCLHelper/OpenCLDeviceHandler.h"

using namespace ATML::Helper;

SCENARIO("Fetching the OpenCLDevices from the OpenCLDeviceHandler", "[OpenCLDevice][OpenCLDeviceHandler]") {
	OpenCLDeviceHandler handler;
	INFO("Getting the device information");
	auto deviceInfos = handler.GetDeviceInfos();
	INFO("Getting the platform information");
	auto platformInfos = handler.GetPlatformInfos();

	GIVEN("All of the devices"){
	auto allDevices = handler.GetDevices();
	WHEN("Fetching devices with all platform informations")
	{
		auto platformInfoDevices = handler.GetDevices(platformInfos);
		THEN("The amount of devices must be equal")
		{
			REQUIRE(allDevices.size() == platformInfoDevices.size());
		}
	}
	WHEN("Fetching devices with single platform infos")
	{
		size_t size2 = 0;
		for(auto& platformInfo : platformInfos)
		{
			auto devices = handler.GetDevices(platformInfo);
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
			auto device = handler.GetDevices(deviceInfo);
		}
		THEN("Make sure that thec count is correct")
		{
			REQUIRE(allDevices.size() == size2);
		}
	}
	WHEN("Fetching devices with all device infos")
	{
		auto deviceInfoDevices = handler.GetDevices(deviceInfos);
		THEN("The amount of devices must be equal")
		{
			REQUIRE(allDevices.size() == deviceInfoDevices.size());
		}
	}
}
}

SCENARIO("Acquiring memory from a device", "[OpenCLMemory][OpenCLDevice][OpenCLDeviceHandler]")
{
	OpenCLDeviceHandler handler;
	INFO("Getting all devices")
	auto devices = handler.GetDevices();

	//TODO: Finish test
}
