/*
 * OpenCLHelperTest.cpp
 *
 *  Created on: Apr 28, 2015
 *      Author: Mikael
 */
#define CATCH_CONFIG_MAIN
#include "catch/catch.hpp"
#include "OpenCLHelper/OpenCLHelper.h"
#include <stdio.h>

using namespace ATML::Helper;

SCENARIO("Fetching device and platform information", "[PlatformInfo]")
{
	WHEN("Getting platform informations"){
		auto platformInfos = OpenCLHelper::GetPlatformInfos();
		if (platformInfos.size() == 0)
		{
			WARN("No platforms are detected. "
				"This is either because you are running "
				"a system without OCL drivers or that we have a bug in the GetPlatformInfo() function.");
		}
		THEN("We should have a vector with a lot of information that is printable")
		{
			for (auto& info : platformInfos)
				cout << info.GetString().c_str() << endl;
		}
	}
}

SCENARIO("Fetching device information", "[DeviceInfo]")
{
	auto platformInfos = OpenCLHelper::GetPlatformInfos();
	if (platformInfos.size() == 0)
	{
		WARN(
			"No platforms are detected. "
			"This is either because you are running "
			"a system without OCL drivers or that we have a bug in the GetPlatformInfo() function.");
	}
	WHEN("Fetching the device info from the platform infos"){
		size_t size1;
		THEN("The device infos should be printable and working")
		{
			for (auto& platformInfo : platformInfos)
			{
				auto deviceInfos = OpenCLHelper::GetDeviceInfos(platformInfo);
				size1 = deviceInfos.size();
				for (auto& info : deviceInfos)
					cout << info.GetString().c_str() << endl;
			}
		}
	}
}

SCENARIO("Fetching the context from the devicehandler", "[OpenCLContext][OpenCLHelper]")
{
	INFO("Fetching the platform infos");
	auto platformInfos = OpenCLHelper::GetPlatformInfos();
	if (platformInfos.size() == 0)
	{
		WARN(
			"No platforms are detected. "
			"This is either because you are running "
			"a system without OCL drivers or that we have a bug in the GetPlatformInfo() function.");
	}
	for (auto& platformInfo : platformInfos)
	{
		WHEN("Creating the context from a platform info with standard arguments"){
			auto context = OpenCLHelper::GetContext(platformInfo);
			THEN("We should be able to get the devices from the context")
			{
				auto devices = context->GetDevices();
				CHECK(devices.size() == context->DeviceCount());
			}
		}
		WHEN("Creating the context with 2 device queues")
		{
			auto context = OpenCLHelper::GetContext(platformInfo, 2);
			THEN("We should be able to get the devices from the context")
			{
				auto devices = context->GetDevices();
				CHECK(devices.size() == context->DeviceCount());
			}
		}
		WHEN("Creating the context using custom configurations")
		{
			auto deviceInfos = OpenCLHelper::GetDeviceInfos(platformInfo);
			vector<tuple<OpenCLDeviceConfig, OpenCLDeviceInfo>> configs;
			for (auto& deviceInfo : deviceInfos)
			{
				OpenCLDeviceConfig config;
				config.AddCommandQueue();
				configs.push_back(make_tuple(config, deviceInfo));
			}
			auto context = OpenCLHelper::GetContext(platformInfo, configs);
			THEN("We should be able to get the devices from the context")
			{
				auto devices = context->GetDevices();
				CHECK(devices.size() == context->DeviceCount());
			}
		}
	}
}
