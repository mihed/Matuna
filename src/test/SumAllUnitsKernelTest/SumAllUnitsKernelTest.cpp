/*
 * SumAllUnitsKernelTest.cpp
 *
 *  Created on: May 20, 2015
 *      Author: Mikael
 */

#define CATCH_CONFIG_MAIN
#include "catch/catch.hpp"
#include "OpenCLHelper/OpenCLHelper.h"
#include "CNNOpenCL/SumAllUnitsKernel.h"
#include "Math/Matrix.h"
#include <memory>
#include <random>
#include <type_traits>

using namespace ATML::Helper;
using namespace ATML::Math;
using namespace ATML::MachineLearning;

SCENARIO("Summing all input units")
{
	random_device tempDevice;
	mt19937 mt(tempDevice());
	uniform_int_distribution<int> dimensionGenerator(1, 100);

	WHEN("Summing images without offset or padding")
	{
		auto platformInfos = OpenCLHelper::GetPlatformInfos();
		for (auto& platformInfo : platformInfos)
		{
			auto context = OpenCLHelper::GetContext(platformInfo);
			auto devices = context->GetDevices();
			for (auto device : devices)
			{
				vector<OpenCLDevice*> oneDeviceVector;
				oneDeviceVector.push_back(device);

				int numberOfUnits = dimensionGenerator(mt);
				int unitWidth = dimensionGenerator(mt);
				int unitHeight = dimensionGenerator(mt);

				INFO("Creating all the input units");
				vector<Matrix<float>> units;
				for (int i = 0; i < numberOfUnits; i++)
					units.push_back(Matrix<float>::RandomNormal(unitHeight, unitWidth));

				INFO("Putting the input units into contiguous memory");
				unique_ptr<float[]> contiguousMemory(new float[unitHeight * unitWidth * numberOfUnits]);
				auto rawPointer = contiguousMemory.get();
				for (int i = 0; i < numberOfUnits; i++)
				{
					auto& unit = units[i];
					memcpy(rawPointer, unit.Data, unit.ElementCount());
					rawPointer += unit.ElementCount();
				}

				INFO("Calculating the expected result by matrix addition");
				Matrix<float> manualResult = Matrix<float>::Zeros(unitHeight, unitWidth);
				for (auto& unit : units)
					manualResult += unit;

				INFO("Creating and compiling the kernel");
				SumAllUnitsKernel<float> kernel(unitWidth, unitHeight, numberOfUnits,
					0, 0, 0, unitWidth, unitHeight,
					0, 0, unitWidth, unitHeight);
				context->AddProgramFromSource(&kernel, oneDeviceVector);
				context->AddKernel(&kernel);

				auto outputSize = sizeof(float) * unitHeight * unitWidth;
				auto inputSize = outputSize * numberOfUnits;
				auto inputMemory = context->CreateMemory(CL_MEM_READ_ONLY, inputSize);
				auto outputMemory = context->CreateMemory(CL_MEM_WRITE_ONLY, outputSize);
				device->WriteMemory(inputMemory.get(), inputSize, contiguousMemory.get());

				kernel.SetInput(inputMemory.get());
				kernel.SetOutput(outputMemory.get());

				device->ExecuteKernel(&kernel);

				Matrix<float> sumKernelResult(unitHeight, unitWidth);
				device->ReadMemory(outputMemory.get(), outputSize, sumKernelResult.Data);

				auto difference = manualResult - sumKernelResult;
				auto absDifference = difference.Norm2Square() / difference.ElementCount();
				cout << "Difference: " << absDifference << endl;
				CHECK(absDifference < 1E-14);
			}
		}
	}

	WHEN("Summing images with offset and no padding")
	{
		auto platformInfos = OpenCLHelper::GetPlatformInfos();
		for (auto& platformInfo : platformInfos)
		{
			auto context = OpenCLHelper::GetContext(platformInfo);
			auto devices = context->GetDevices();
			for (auto device : devices)
			{
			}
		}
	}

	WHEN("Summing images with padding and no offset")
	{
		auto platformInfos = OpenCLHelper::GetPlatformInfos();
		for (auto& platformInfo : platformInfos)
		{
			auto context = OpenCLHelper::GetContext(platformInfo);
			auto devices = context->GetDevices();
			for (auto device : devices)
			{
			}
		}
	}

	WHEN("Summing images with padding and offset")
	{
		auto platformInfos = OpenCLHelper::GetPlatformInfos();
		for (auto& platformInfo : platformInfos)
		{
			auto context = OpenCLHelper::GetContext(platformInfo);
			auto devices = context->GetDevices();
			for (auto device : devices)
			{
			}
		}
	}
}



