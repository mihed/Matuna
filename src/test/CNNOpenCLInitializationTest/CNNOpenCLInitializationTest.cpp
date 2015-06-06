/*
 * CNNOpenCLInitializationTest.cpp
 *
 *  Created on: May 9, 2015
 *      Author: Mikael
 */

#define CATCH_CONFIG_MAIN
#include "catch/catch.hpp"
#include "CNNOpenCL/CNNOpenCL.h"
#include "OpenCLHelper/OpenCLHelper.h"
#include "OpenCLHelper/OpenCLContext.h"
#include "CNN/PerceptronLayerConfig.h"
#include "CNN/LayerDescriptions.h"
#include "CNN/StandardOutputLayerConfig.h"
#include <vector>
#include <memory>

using namespace std;
using namespace Matuna::MachineLearning;
using namespace Matuna::Helper;

SCENARIO("Creating a CNNOpenCL network", "[CNNOpenCL][OpenCLContext]")
{
	INFO("Getting the platform infos");
	auto platformInfos = OpenCLHelper::GetPlatformInfos();

	if (platformInfos.size() == 0)
	{
		WARN(
			"No platforms are detected. "
			"This is either because you are running "
			"a system without OCL drivers or that we have a bug in the GetPlatformInfo() function.");
	}

	INFO("Creating a OCL Context");
	auto deviceInfos = OpenCLHelper::GetDeviceInfos(platformInfos[0]);

	INFO("Creating a suitable CNNConfig");

	vector<LayerDataDescription> dataDescriptions;
	LayerDataDescription desc1;
	desc1.Height = 1;
	desc1.Width = 1;
	desc1.Units = 100;
	dataDescriptions.push_back(desc1);

	LayerDataDescription desc2;
	desc2.Height = 1;
	desc2.Width = 1;
	desc2.Units = 100;
	dataDescriptions.push_back(desc2);

	unique_ptr<CNNConfig> config(new CNNConfig(dataDescriptions));
	unique_ptr<ForwardBackPropLayerConfig> config1(new PerceptronLayerConfig(18));
	unique_ptr<ForwardBackPropLayerConfig> config2(new PerceptronLayerConfig(120));
	unique_ptr<ForwardBackPropLayerConfig> config3(new PerceptronLayerConfig(14));
	unique_ptr<OutputLayerConfig> oConfig(new StandardOutputLayerConfig());
	config->SetOutputConfig(move(oConfig));
	config->AddToBack(move(config1));
	config->AddToBack(move(config2));
	config->AddToBack(move(config3));

	INFO("Creating a CNNOpenCL<cl_float> network from the config");
	CNNOpenCL<cl_float> network(deviceInfos, move(config));
	CHECK(network.Interlocked());
}
