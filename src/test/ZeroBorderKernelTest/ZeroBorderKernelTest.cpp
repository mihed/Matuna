/*
 * ZeroBorderKernelTest.cpp
 *
 *  Created on: May 23, 2015
 *      Author: Mikael
 */

#define CATCH_CONFIG_MAIN
#include "catch/catch.hpp"
#include "OpenCLHelper/OpenCLHelper.h"
#include "CNNOpenCL/ZeroBorderKenel.h"
#include "Math/Matrix.h"
#include <memory>
#include <random>
#include <type_traits>

using namespace ATML::Helper;
using namespace ATML::Math;
using namespace ATML::MachineLearning;

SCENARIO("Adding a zero border around the data")
{
	random_device tempDevice;
	mt19937 mt(tempDevice());
	uniform_int_distribution<int> dimensionGenerator(1, 100);
	uniform_int_distribution<int> unitGenerator(1, 40);
	uniform_int_distribution<int> paddingGenerator(1, 20);

	WHEN("Adding a zero border to a random input with random border")
	{
		for (int dummy = 0; dummy < 2; dummy++)
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

					int width = dimensionGenerator(mt);
					int height = dimensionGenerator(mt);
					int unitsCount = unitGenerator(mt);

					int borderSize = paddingGenerator(mt);
					int widthOffset = borderSize + paddingGenerator(mt);
					int heightOffset = borderSize + paddingGenerator(mt);
					int unitOffset = paddingGenerator(mt);

					int widthPadding = borderSize + paddingGenerator(mt);
					int heightPadding = borderSize + paddingGenerator(mt);
					int unitPadding = paddingGenerator(mt);

					int totalUnits = unitsCount + unitOffset + unitPadding;
					int totalHeight = height + heightOffset + heightPadding;
					int totalWidth = width + widthPadding + widthOffset;
					int totalElementsPerUnit = totalHeight * totalWidth;

					INFO("Creating all the input units");
					vector<Matrixf> units;
					for (int i = 0; i < totalUnits; i++)
						units.push_back(Matrixf::RandomNormal(totalHeight, totalWidth));

					unique_ptr<float[]> inputMemory(new float[totalUnits * totalHeight * totalWidth]);
					for (int i = 0; i < totalUnits; i++)
						memcpy(inputMemory.get() + i * totalElementsPerUnit, units[i].Data, sizeof(float) * totalElementsPerUnit);

					int borderStartUp = heightOffset - borderSize;
					int borderStartDown = heightOffset + height;
					int borderStartLeft = widthOffset - borderSize;
					int borderStartRight = widthOffset + width;

					INFO("Creating the zero border kernel");
					ZeroBorderKenel<float> kernel(width, height, unitsCount,
						borderStartLeft, borderStartRight, borderStartUp,
						borderStartDown, borderSize, totalWidth, totalHeight, unitOffset);

					INFO("Initializing the compiler options");
					kernel.InitializeCompilerOptions();

					INFO("Adding the kernel as a program");
					context->AddProgramFromSource(&kernel, oneDeviceVector);
					context->AddKernel(&kernel);

					INFO("Creating the OCL memory");
					auto oclInputMemory = context->CreateMemory(CL_MEM_READ_WRITE, totalElementsPerUnit * totalUnits * sizeof(float));
					device->WriteMemory(oclInputMemory.get(), oclInputMemory->ByteSize(), inputMemory.get());
					device->WaitForDeviceQueue(0);

					INFO("Setting the OCL memory to the kernel");
					kernel.SetInputOutput(oclInputMemory.get());

					INFO("Executing the kernel");
					device->ExecuteKernel(&kernel);
					device->WaitForDeviceQueue(0);
					device->ReadMemory(oclInputMemory.get(), oclInputMemory->ByteSize(), inputMemory.get());
					device->WaitForDeviceQueue(0);

					vector<Matrixf> rawKernelResult;
					for (int i = 0; i < totalUnits; i++)
						rawKernelResult.push_back(Matrixf(totalHeight, totalWidth, inputMemory.get() + i * totalElementsPerUnit));

					INFO("Comparing the results");
					for (int i = unitOffset; i < (unitOffset + unitsCount); i++)
					{
						auto manualResult = units[i].GetSubMatrix(heightOffset, widthOffset, height, width).AddZeroBorder(borderSize);
						auto kernelResult = rawKernelResult[i].GetSubMatrix(borderStartUp, borderStartLeft, height + 2 * borderSize, width + 2 * borderSize);

						for (int j = 0; j < manualResult.ElementCount(); j++)
							CHECK(manualResult.Data[j] == kernelResult.Data[j]);
					}
				}
			}
		}
	}
}


