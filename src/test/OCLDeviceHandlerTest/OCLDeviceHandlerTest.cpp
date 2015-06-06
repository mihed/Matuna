/*
 * OCLHelperTest.cpp
 *
 *  Created on: Apr 28, 2015
 *      Author: Mikael
 */
#define CATCH_CONFIG_MAIN
#include "catch/catch.hpp"
#include "OCLHelper/OCLHelper.h"
#include <stdio.h>

using namespace Matuna::Helper;

SCENARIO("Fetching device and platform information", "[PlatformInfo]")
{
	WHEN("Getting platform informations"){
		auto platformInfos = OCLHelper::GetPlatformInfos();
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
	auto platformInfos = OCLHelper::GetPlatformInfos();
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
				auto deviceInfos = OCLHelper::GetDeviceInfos(platformInfo);
				size1 = deviceInfos.size();
				for (auto& info : deviceInfos)
					cout << info.GetString().c_str() << endl;
			}
		}
	}
}

SCENARIO("Fetching the context from the devicehandler", "[OCLContext][OCLHelper]")
{
	INFO("Fetching the platform infos");
	auto platformInfos = OCLHelper::GetPlatformInfos();
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
			auto context = OCLHelper::GetContext(platformInfo);
			THEN("We should be able to get the devices from the context")
			{
				auto devices = context->GetDevices();
				CHECK(devices.size() == context->DeviceCount());
			}
		}
		WHEN("Creating the context with 2 device queues")
		{
			auto context = OCLHelper::GetContext(platformInfo, 2);
			THEN("We should be able to get the devices from the context")
			{
				auto devices = context->GetDevices();
				CHECK(devices.size() == context->DeviceCount());
			}
		}
		WHEN("Creating the context using custom configurations")
		{
			auto deviceInfos = OCLHelper::GetDeviceInfos(platformInfo);
			vector<tuple<OCLDeviceConfig, OCLDeviceInfo>> configs;
			for (auto& deviceInfo : deviceInfos)
			{
				OCLDeviceConfig config;
				config.AddCommandQueue();
				configs.push_back(make_tuple(config, deviceInfo));
			}
			auto context = OCLHelper::GetContext(platformInfo, configs);
			THEN("We should be able to get the devices from the context")
			{
				auto devices = context->GetDevices();
				CHECK(devices.size() == context->DeviceCount());
			}
		}
	}
}
