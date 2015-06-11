
#define CATCH_CONFIG_MAIN
#include "catch/catch.hpp"
#include "Matuna.OCLHelper/OCLHelper.h"
#include "Matuna.OCLHelper/OCLProgram.h"
#include "Matuna.OCLConvNet/LayerKernel.h"
#include <memory>

using namespace Matuna::Helper;
using namespace Matuna::MachineLearning;

SCENARIO("Testing and executing a layer kernel")
{

	auto platformInfos = OCLHelper::GetPlatformInfos();
	for (auto& platformInfo : platformInfos)
	{
		auto context =  OCLHelper::GetContext(platformInfo);
		auto program = new OCLProgram();
		program->SetName("Testprogram");
		LayerKernel<cl_float>* kernel = new LayerKernel<cl_float>();

		int memoryCount = 100;
		cl_float scalar = 10;
		auto memory = context->CreateMemory(CL_MEM_READ_WRITE, memoryCount * sizeof(cl_float));

		kernel->SetKernelName("DivideByScalarKernel");
		string path = OCLProgram::DefaultSourceLocation + "LayerTestKernel.cl";
		kernel->AddSourcePath(path);
		kernel->AddIncludePath(OCLProgram::DefaultSourceLocation);
		kernel->AddGlobalSize(memoryCount);
		kernel->AddDefineSubsitute(path, "OFFSET_SCALAR", to_string(10)); 

		program->AttachKernel(unique_ptr<OCLKernel>(kernel));
		context->AttachProgram(unique_ptr<OCLProgram>(program), context->GetDevices());
		kernel->SetMemoryArg(memory.get(), 0);
		kernel->SetRealArg(scalar, 1);

		for(auto device: context->GetDevices())
			device->ExecuteKernel(kernel);

		auto program2 = new OCLProgram();
		program->SetName("Testprogram2");
		LayerKernel<cl_float>* kernel2 = new LayerKernel<cl_float>();

		kernel2->SetKernelName("DivideByScalarKernel");
		kernel2->AddSourcePath(path);
		kernel2->AddIncludePath(OCLProgram::DefaultSourceLocation);
		kernel2->AddGlobalSize(memoryCount);
		kernel2->AddDefineSubsitute(path, "OFFSET_SCALAR", to_string(10));
		kernel2->AddDefine(path, "USE_OFFSET");

		program2->AttachKernel(unique_ptr<OCLKernel>(kernel2));
		context->AttachProgram(unique_ptr<OCLProgram>(program2), context->GetDevices());
		kernel2->SetMemoryArg(memory.get(), 0);
		kernel2->SetRealArg(scalar, 1);

		for(auto device: context->GetDevices())
			device->ExecuteKernel(kernel2);
	}
}