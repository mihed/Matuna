/*
 * OpenCLDeviceHandlerTest.cpp
 *
 *  Created on: Apr 28, 2015
 *      Author: Mikael
 */
#define CATCH_CONFIG_MAIN
#include "catch/catch.hpp"
#include "OpenCLHelper/OpenCLDeviceHandler.h"

using namespace ATML::Helper;

SCENARIO("Fetching device and platform information", "[PlatformInfo]") {
	OpenCLDeviceHandler deviceHandler;
	WHEN("Getting platform informations"){
	auto platformInfos = deviceHandler.GetPlatformInfos();
	if (platformInfos.size() == 0)
	{
		WARN("No platforms are detected. "
				"This is either because you are running "
				"a system without OCL drivers or that we have a bug in the GetPlatformInfo() function.");
	}
	THEN("We should have a vector with a lot of information that is printable") {
		for(auto& info : platformInfos)
		printf(info.GetString().c_str());
	}
}
}

SCENARIO("Fetching device information", "[DeviceInfo]") {
	OpenCLDeviceHandler deviceHandler;
	auto platformInfos = deviceHandler.GetPlatformInfos();
	if (platformInfos.size() == 0) {
		WARN(
				"No platforms are detected. "
						"This is either because you are running "
						"a system without OCL drivers or that we have a bug in the GetPlatformInfo() function.");
	}
	WHEN("Fetching the device info from the platform infos"){
	size_t size1;
	THEN("The device infos should be printable and working") {
		auto deviceInfos = deviceHandler.GetDeviceInfos(platformInfos);
		size1 = deviceInfos.size();
		for(auto& info : deviceInfos)
		printf(info.GetString().c_str());
	}
	THEN("Manual fetching per platform should give the same answer")
	{
		size_t size2 = 0;
		for(auto& platformInfo : platformInfos)
		{
			auto deviceInfos = deviceHandler.GetDeviceInfos(platformInfo);
			size2 += deviceInfos.size();
			for(auto& info : deviceInfos)
			printf(info.GetString().c_str());
		}

		REQUIRE(size1 == size2);
	}
}
}
