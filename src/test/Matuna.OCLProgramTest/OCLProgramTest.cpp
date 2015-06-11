/*
* OCLProgramTest.cpp
*
*  Created on: Jun 10, 2015
*      Author: Mikael
*/

#define CATCH_CONFIG_MAIN
#include "catch/catch.hpp"
#include "OCLTestKernel.h"
#include "OCLTestSourceKernel.h"
#include "OCLMatunaTestKernel.h"
#include "Matuna.OCLHelper/OCLHelper.h"
#include <memory>

using namespace Matuna::Helper;

SCENARIO("Testing to compile a program by defining its source path and no associated kernels")
{
	auto platformInfos = OCLHelper::GetPlatformInfos();
	for (auto& platformInfo : platformInfos)
	{
		auto context =  OCLHelper::GetContext(platformInfo);
		auto program = new OCLProgram();
		program->AddProgramPath(OCLProgram::DefaultSourceLocation + "ConvolutionKernel.cl");
		program->AddIncludePath(OCLProgram::DefaultSourceLocation);
		program->SetName("Testprogram");
		auto program2 = new OCLProgram();
		program2->SetName("Testprogram");
		context->AttachProgram(unique_ptr<OCLProgram>(program), context->GetDevices());
		CHECK_THROWS(context->AttachProgram(unique_ptr<OCLProgram>(program2), context->GetDevices()));
		context->DetachProgram(program);
		CHECK_THROWS(context->DetachProgram(program));
	}
}

SCENARIO("Testing a simple kernel")
{
	auto platformInfos = OCLHelper::GetPlatformInfos();
	for (auto& platformInfo : platformInfos)
	{
		auto context =  OCLHelper::GetContext(platformInfo);
		auto program = new OCLProgram();
		program->SetName("Testprogram");
		program->AddProgramPath(OCLProgram::DefaultSourceLocation + "ConvolutionKernel.cl");
		program->AddIncludePath(OCLProgram::DefaultSourceLocation);
		auto testKernel = new OCLTestKernel("ConvolutionKernel");
		auto testKernel2 = new OCLTestKernel("ConvolutionKernel");
		program->AttachKernel(unique_ptr<OCLKernel>(testKernel));
		INFO("We cannot attach a kernel of the same name. This would imply two definitions of a single kernel.");
		CHECK_THROWS(program->AttachKernel(unique_ptr<OCLKernel>(testKernel2)));
		program->DetachKernel(testKernel);
		testKernel = new OCLTestKernel("ConvolutionKernel");
		INFO("Now the test kernel should work fine");
		program->AttachKernel(unique_ptr<OCLKernel>(testKernel));
		context->AttachProgram(unique_ptr<OCLProgram>(program), context->GetDevices());
		context->DetachProgram(program);
	}
}

SCENARIO("Testing a simple source kernel")
{
	auto platformInfos = OCLHelper::GetPlatformInfos();
	for (auto& platformInfo : platformInfos)
	{
		auto context =  OCLHelper::GetContext(platformInfo);
		auto program = new OCLProgram();
		program->SetName("Testprogram");
		auto testKernel = new OCLTestSourceKernel();
		program->AttachKernel(unique_ptr<OCLKernel>(testKernel));
		context->AttachProgram(unique_ptr<OCLProgram>(program), context->GetDevices());
	}
}

SCENARIO("Testing a source kernel with Matuna extensions")
{
	auto platformInfos = OCLHelper::GetPlatformInfos();
	for (auto& platformInfo : platformInfos)
	{
		auto context =  OCLHelper::GetContext(platformInfo);
		auto program = new OCLProgram();
		program->SetName("Testprogram");
		auto testKernel = new OCLMatunaTestKernel();
		program->AttachKernel(unique_ptr<OCLKernel>(testKernel));
		context->AttachProgram(unique_ptr<OCLProgram>(program), context->GetDevices());
	}
}

SCENARIO("Testing a source kernel with Matuna extensions disabled")
{
	auto platformInfos = OCLHelper::GetPlatformInfos();
	for (auto& platformInfo : platformInfos)
	{
		auto context =  OCLHelper::GetContext(platformInfo);
		auto program = new OCLProgram();
		program->SetEnableMatunaScript(false);
		program->SetName("Testprogram");
		auto testKernel = new OCLMatunaTestKernel();
		program->AttachKernel(unique_ptr<OCLKernel>(testKernel));
		context->AttachProgram(unique_ptr<OCLProgram>(program), context->GetDevices());
	}
}


