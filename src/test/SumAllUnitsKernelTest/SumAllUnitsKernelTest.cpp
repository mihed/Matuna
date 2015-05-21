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
					memcpy(rawPointer, unit.Data, unit.ElementCount() * sizeof(float));
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

				kernel.InitializeCompilerOptions();

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

			//cout << "Platform name: " << context->GetPlatformInfo().PlatformName() << endl;
			//if (context->GetPlatformInfo().PlatformName().find("Intel") == string::npos)
			//	continue;

			auto devices = context->GetDevices();
			for (auto device : devices)
			{

				vector<OpenCLDevice*> oneDeviceVector;
				oneDeviceVector.push_back(device);

				int dataUnits = dimensionGenerator(mt);
				int dataWidth = dimensionGenerator(mt);
				int dataHeight = dimensionGenerator(mt);
				int inputWidthOffset = dimensionGenerator(mt);
				int inputHeightOffset = dimensionGenerator(mt);
				int inputUnitOffset = dimensionGenerator(mt);
				int outputWidthOffset = dimensionGenerator(mt);
				int outputHeightOffset = dimensionGenerator(mt);

				INFO("Creating all the input units");
				int totalInputHeight = dataHeight + inputHeightOffset;
				int totalInputWidth = dataWidth + inputWidthOffset;
				int totalInputUnits = dataUnits + inputUnitOffset;
				vector<Matrix<float>> units;
				for (int i = 0; i < totalInputUnits; i++)
					units.push_back(Matrix<float>::RandomNormal(totalInputHeight, totalInputWidth));

				INFO("Putting the input units into contiguous memory");
				unique_ptr<float[]> contiguousMemory(new float[totalInputHeight * totalInputWidth * totalInputUnits]);
				auto rawPointer = contiguousMemory.get();
				for (int i = 0; i < totalInputUnits; i++)
				{
					auto& unit = units[i];
					memcpy(rawPointer, unit.Data, unit.ElementCount() * sizeof(float));
					rawPointer += unit.ElementCount();
				}

				INFO("Calculating the expected result by matrix addition");
				Matrix<float> manualResult = Matrix<float>::Zeros(totalInputHeight, totalInputWidth);
				for (int i = inputUnitOffset; i < totalInputUnits; i++)
					manualResult += units[i];

				Matrix<float> manualSubMatrix = manualResult.GetSubMatrix(inputHeightOffset, inputWidthOffset, dataHeight, dataWidth);

				INFO("Creating and compiling the kernel");
				SumAllUnitsKernel<float> kernel(dataWidth, dataHeight, dataUnits,
					inputWidthOffset, inputHeightOffset, inputUnitOffset, dataWidth + inputWidthOffset, dataHeight + inputHeightOffset,
					outputWidthOffset, outputHeightOffset, outputWidthOffset + dataWidth, outputHeightOffset + dataHeight);

				kernel.InitializeCompilerOptions();

				context->AddProgramFromSource(&kernel, oneDeviceVector);
				context->AddKernel(&kernel);

				auto outputSize = sizeof(float) * (outputWidthOffset + dataWidth) * (outputHeightOffset + dataHeight);
				auto inputSize = sizeof(float) * totalInputHeight * totalInputWidth * totalInputUnits;
				auto inputMemory = context->CreateMemory(CL_MEM_READ_ONLY, inputSize);
				auto outputMemory = context->CreateMemory(CL_MEM_WRITE_ONLY, outputSize);
				device->WriteMemory(inputMemory.get(), inputSize, contiguousMemory.get());

				kernel.SetInput(inputMemory.get());
				kernel.SetOutput(outputMemory.get());

				device->ExecuteKernel(&kernel);

				Matrix<float> sumKernelResult(outputHeightOffset + dataHeight, outputWidthOffset + dataWidth);
				device->ReadMemory(outputMemory.get(), outputSize, sumKernelResult.Data, 0, true);
				Matrix<float> kernelSubMatrix = sumKernelResult.GetSubMatrix(outputHeightOffset, outputWidthOffset, dataHeight, dataWidth);

				auto difference = manualSubMatrix - kernelSubMatrix;
				auto absDifference = difference.Norm2Square() / difference.ElementCount();
				cout << "Difference: " << absDifference << endl;
				if (absDifference >= 1E-14)
				{

					//cout << "Kernel Sub matrix: \n" << kernelSubMatrix.GetString() << endl;

					cout << "Failed: " << endl;

					cout << "Device Info" << endl;
					cout << device->DeviceInfo().GetString() << endl;

					cout << "Data units: " << dataUnits << endl;
					cout << "Data width: " << dataWidth << endl;
					cout << "Data height: " << dataHeight << endl;
					cout << "inputWidthOffset	" << inputWidthOffset << endl;
					cout << "inputHeightOffset	" << inputHeightOffset << endl;
					cout << "inputUnitOffset	" << inputUnitOffset << endl;
					cout << "outputWidthOffset	" << outputWidthOffset << endl;
					cout << "outputHeightOffset	" << outputHeightOffset << endl;

				}
				CHECK(absDifference < 1E-14);
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
				vector<OpenCLDevice*> oneDeviceVector;
				oneDeviceVector.push_back(device);

				int dataUnits = dimensionGenerator(mt);
				int dataWidth = dimensionGenerator(mt);
				int dataHeight = dimensionGenerator(mt);
				int inputWidthPadding = dimensionGenerator(mt);
				int inputHeightPadding = dimensionGenerator(mt);
				int inputUnitPadding = dimensionGenerator(mt);
				int outputWidthPadding = dimensionGenerator(mt);
				int outputHeightPadding = dimensionGenerator(mt);

				INFO("Creating all the input units");
				int totalInputHeight = dataHeight + inputHeightPadding;
				int totalInputWidth = dataWidth + inputWidthPadding;
				int totalInputUnits = dataUnits + inputUnitPadding;
				vector<Matrix<float>> units;
				for (int i = 0; i < totalInputUnits; i++)
					units.push_back(Matrix<float>::RandomNormal(totalInputHeight, totalInputWidth));

				INFO("Putting the input units into contiguous memory");
				unique_ptr<float[]> contiguousMemory(new float[totalInputHeight * totalInputWidth * totalInputUnits]);
				auto rawPointer = contiguousMemory.get();
				for (int i = 0; i < totalInputUnits; i++)
				{
					auto& unit = units[i];
					memcpy(rawPointer, unit.Data, unit.ElementCount() * sizeof(float));
					rawPointer += unit.ElementCount();
				}

				INFO("Calculating the expected result by matrix addition");
				Matrix<float> manualResult = Matrix<float>::Zeros(totalInputHeight, totalInputWidth);
				for (int i = 0; i < dataUnits; i++)
					manualResult += units[i];

				Matrix<float> manualSubMatrix = manualResult.GetSubMatrix(0, 0, dataHeight, dataWidth);

				INFO("Creating and compiling the kernel");
				SumAllUnitsKernel<float> kernel(dataWidth, dataHeight, dataUnits,
					0, 0, 0, dataWidth + inputWidthPadding, dataHeight + inputHeightPadding,
					0, 0, outputWidthPadding + dataWidth, outputHeightPadding + dataHeight);

				kernel.InitializeCompilerOptions();

				context->AddProgramFromSource(&kernel, oneDeviceVector);
				context->AddKernel(&kernel);

				auto outputSize = sizeof(float) * (outputWidthPadding + dataWidth) * (outputHeightPadding + dataHeight);
				auto inputSize = sizeof(float) * totalInputHeight * totalInputWidth * totalInputUnits;
				auto inputMemory = context->CreateMemory(CL_MEM_READ_ONLY, inputSize);
				auto outputMemory = context->CreateMemory(CL_MEM_WRITE_ONLY, outputSize);
				device->WriteMemory(inputMemory.get(), inputSize, contiguousMemory.get());

				kernel.SetInput(inputMemory.get());
				kernel.SetOutput(outputMemory.get());

				device->ExecuteKernel(&kernel);

				Matrix<float> sumKernelResult(outputHeightPadding + dataHeight, outputWidthPadding + dataWidth);
				device->ReadMemory(outputMemory.get(), outputSize, sumKernelResult.Data, 0, true);
				Matrix<float> kernelSubMatrix = sumKernelResult.GetSubMatrix(0, 0, dataHeight, dataWidth);

				auto difference = manualSubMatrix - kernelSubMatrix;
				auto absDifference = difference.Norm2Square() / difference.ElementCount();
				cout << "Difference: " << absDifference << endl;
				CHECK(absDifference < 1E-14);
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
				vector<OpenCLDevice*> oneDeviceVector;
				oneDeviceVector.push_back(device);

				int dataUnits = dimensionGenerator(mt);
				int dataWidth = dimensionGenerator(mt);
				int dataHeight = dimensionGenerator(mt);

				int inputWidthOffset = dimensionGenerator(mt);
				int inputHeightOffset = dimensionGenerator(mt);
				int inputUnitOffset = dimensionGenerator(mt);
				int outputWidthOffset = dimensionGenerator(mt);
				int outputHeightOffset = dimensionGenerator(mt);

				int inputWidthPadding = dimensionGenerator(mt);
				int inputHeightPadding = dimensionGenerator(mt);
				int inputUnitPadding = dimensionGenerator(mt);
				int outputWidthPadding = dimensionGenerator(mt);
				int outputHeightPadding = dimensionGenerator(mt);

				INFO("Creating all the input units");
				int totalInputHeight = dataHeight + inputHeightPadding + inputHeightOffset;
				int totalInputWidth = dataWidth + inputWidthPadding + inputWidthOffset;
				int totalInputUnits = dataUnits + inputUnitPadding + inputUnitOffset;
				vector<Matrix<float>> units;
				for (int i = 0; i < totalInputUnits; i++)
					units.push_back(Matrix<float>::RandomNormal(totalInputHeight, totalInputWidth));

				INFO("Putting the input units into contiguous memory");
				unique_ptr<float[]> contiguousMemory(new float[totalInputHeight * totalInputWidth * totalInputUnits]);
				auto rawPointer = contiguousMemory.get();
				for (int i = 0; i < totalInputUnits; i++)
				{
					auto& unit = units[i];
					memcpy(rawPointer, unit.Data, unit.ElementCount() * sizeof(float));
					rawPointer += unit.ElementCount();
				}

				INFO("Calculating the expected result by matrix addition");
				Matrix<float> manualResult = Matrix<float>::Zeros(totalInputHeight, totalInputWidth);
				for (int i = inputUnitOffset; i < (dataUnits + inputUnitOffset); i++)
					manualResult += units[i];

				Matrix<float> manualSubMatrix = manualResult.GetSubMatrix( inputHeightOffset, inputWidthOffset, dataHeight, dataWidth);

				INFO("Creating and compiling the kernel");
				SumAllUnitsKernel<float> kernel(dataWidth, dataHeight, dataUnits,
					inputWidthOffset, inputHeightOffset, inputUnitOffset, totalInputWidth, totalInputHeight,
					outputWidthOffset, outputHeightOffset, outputWidthPadding + dataWidth + outputWidthOffset, outputHeightPadding + dataHeight + outputHeightOffset);

				kernel.InitializeCompilerOptions();

				context->AddProgramFromSource(&kernel, oneDeviceVector);
				context->AddKernel(&kernel);

				auto outputSize = sizeof(float) * (outputWidthPadding + dataWidth + outputWidthOffset) * (outputHeightPadding + dataHeight + outputHeightOffset);
				auto inputSize = sizeof(float) * totalInputHeight * totalInputWidth * totalInputUnits;
				auto inputMemory = context->CreateMemory(CL_MEM_READ_ONLY, inputSize);
				auto outputMemory = context->CreateMemory(CL_MEM_WRITE_ONLY, outputSize);
				device->WriteMemory(inputMemory.get(), inputSize, contiguousMemory.get());

				kernel.SetInput(inputMemory.get());
				kernel.SetOutput(outputMemory.get());

				device->ExecuteKernel(&kernel);

				Matrix<float> sumKernelResult(outputHeightPadding + dataHeight + outputHeightOffset, outputWidthPadding + dataWidth + outputWidthOffset);
				device->ReadMemory(outputMemory.get(), outputSize, sumKernelResult.Data, 0, true);
				Matrix<float> kernelSubMatrix = sumKernelResult.GetSubMatrix(outputHeightOffset, outputWidthOffset, dataHeight, dataWidth);

				auto difference = manualSubMatrix - kernelSubMatrix;
				auto absDifference = difference.Norm2Square() / difference.ElementCount();
				cout << "Difference: " << absDifference << endl;
				CHECK(absDifference < 1E-14);
			}
		}
	}
}



